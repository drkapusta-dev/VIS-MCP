"""VIS-MCP server (v0) — multi-model fan-out.

Core function: ask once, fan out to N models concurrently, return their answers
side by side (optionally a light merged summary). Goal: stop being the courier
between AI models.

Boundaries (see CLAUDE.md): VIS emits evidence/decision-support only; it never
calls a finality layer and never imports an evidence-backend's code. The sink
seam below is a dormant no-op in v0 (v2 wires it to the evidence backend over a
data boundary).
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from datetime import date

from pydantic import ValidationError

import vis_case
from vis_schemas import AdvisoryResponse, AskAllResult, ModelConfig

# =============================================================================
# CONFIG (env-only keys; fail-closed on missing env)
# =============================================================================
CONFIG_PATH = Path(os.getenv("VIS_CONFIG", Path(__file__).parent / "config.yaml"))


def _interpolate(obj: Any) -> Any:
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        env = obj[2:-1]
        val = os.getenv(env)
        if val is None:
            raise RuntimeError(f"Missing required env var: {env}")
        return val
    if isinstance(obj, dict):
        return {k: _interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v) for v in obj]
    return obj


def load_config() -> Dict[str, Any]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    # Only interpolate api_keys lazily (so the server can list models without all keys set).
    raw["models"] = [ModelConfig.model_validate(m).model_dump() for m in raw.get("models", [])]
    return raw


CONFIG = load_config()
DEFAULTS = CONFIG.get("defaults", {})
MODELS: List[ModelConfig] = [ModelConfig.model_validate(m) for m in CONFIG["models"]]


def _key(provider: str) -> str:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")).get("api_keys", {})
    return _interpolate(raw.get(provider, ""))  # raises if env missing


# =============================================================================
# MULTI-PROVIDER CLIENT (shared v0/v1 seam)
# =============================================================================
class AdvisoryClient:
    def __init__(self) -> None:
        self.http = httpx.AsyncClient(timeout=300.0)

    async def close(self) -> None:
        await self.http.aclose()

    async def ask(self, m: ModelConfig, system: str, user: str) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        if m.provider in ("openrouter", "xai"):
            url = (
                "https://openrouter.ai/api/v1/chat/completions"
                if m.provider == "openrouter"
                else "https://api.x.ai/v1/chat/completions"
            )
            headers = {"Authorization": f"Bearer {_key(m.provider)}", "Content-Type": "application/json"}
            payload: Dict[str, Any] = {"model": m.model_id, "messages": messages, "max_tokens": m.max_tokens}
            if m.temperature is not None:
                payload["temperature"] = m.temperature
            resp = await self.http.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"].get("content") or ""
        if m.provider == "anthropic":
            headers = {
                "x-api-key": _key("anthropic"),
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
            payload = {"model": m.model_id, "messages": [{"role": "user", "content": user}], "max_tokens": m.max_tokens}
            if system:
                payload["system"] = system
            if m.temperature is not None:
                payload["temperature"] = m.temperature
            resp = await self.http.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
        raise ValueError(f"Unknown provider: {m.provider}")


client = AdvisoryClient()


async def _ask_one(m: ModelConfig, system: str, user: str) -> AdvisoryResponse:
    t0 = time.monotonic()
    try:
        content = await client.ask(m, system, user)
        return AdvisoryResponse(model_id=m.id, ok=True, content=content,
                                latency_ms=int((time.monotonic() - t0) * 1000))
    except Exception as e:  # one model failing must not sink the rest
        return AdvisoryResponse(model_id=m.id, ok=False, error=f"{type(e).__name__}: {e}",
                                latency_ms=int((time.monotonic() - t0) * 1000))


async def ask_all(question: str, system: str = "", only: Optional[List[str]] = None,
                  synthesize: bool = False) -> AskAllResult:
    models = [m for m in MODELS if (only is None or m.id in only)]
    responses = await asyncio.gather(*[_ask_one(m, system, question) for m in models])
    summary = None
    if synthesize:
        ok = [r for r in responses if r.ok]
        if ok:
            merged = "\n\n".join(f"## {r.model_id}\n{r.content}" for r in ok)
            synth_model = next((m for m in models if m.id in {r.model_id for r in ok}), None)
            if synth_model is not None:
                summary = await client.ask(
                    synth_model,
                    "You merge multiple model answers into one concise synthesis: agreements, "
                    "disagreements, and the strongest combined recommendation. No new claims.",
                    f"Question:\n{question}\n\nAnswers:\n{merged}",
                )
    return AskAllResult(question=question, responses=list(responses), summary=summary)


# =============================================================================
# SINK SEAM (dormant in v0; v2 wires this to the evidence backend over a boundary)
# =============================================================================
async def emit_to_sink(payload: Dict[str, Any]) -> Optional[str]:
    """v0 no-op. v2: emit an external-trust-artifact-reference dict to the evidence
    backend (HTTP/file) — never an import, never a finality-layer call."""
    return None


# =============================================================================
# MCP SERVER
# =============================================================================
app = Server("vis-mcp")


@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="ask-all",
            description="Ask one question to all configured AI models concurrently; return their "
                        "answers side by side (optionally a merged summary). Stops manual copy-paste.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "system": {"type": "string", "default": ""},
                    "only": {"type": "array", "items": {"type": "string"},
                             "description": "Optional model ids to limit to (e.g. ['opus','grok'])."},
                    "synthesize": {"type": "boolean", "default": False,
                                   "description": "Also return one model-merged summary — a reading aid, "
                                                  "never the VIS synthesis stage (that is code)."},
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="list-models",
            description="List the configured models (id, name, provider, slug).",
            inputSchema={"type": "object", "properties": {}},
        ),
        *VIS_TOOLS,
    ]


# =============================================================================
# VIS pipeline tools (v1 step 2: Quick VIS). Models draft; code validates, gates and hashes.
# =============================================================================
_WHO = {
    "by": {"type": "string", "description": "Who hands this in: a person's name, or the client's id when it drafts as staff."},
    "role": {"type": "string", "enum": ["human", "staff", "red_team"],
             "description": "'human' only for what a person typed; a model's draft is never 'human'."},
    "family": {"type": "string", "description": "Model family of the submitter ('human' for a person)."},
}

VIS_TOOLS = [
    Tool(
        name="vis-start",
        description="Open a VIS case (one decision). The charter: case_id, mode ('quick' or 'standard'), intent "
                    "{why, what, by_when, constraints, statement ≤ 2 sentences}, question {decision_maker, action, "
                    "by_when, constraints, outcome}, roles {decision_owner, recommender}, deadline (YYYY-MM-DD), "
                    "reversibility ('reversible' | 'irreversible' | 'severe', set by the decision owner), "
                    "reversibility_set_by (= decision_owner), statements [{text, label fact|assumption|interpretation, "
                    "source_ref}], tags. Loads the lessons pack.",
        inputSchema={"type": "object", "properties": {
            "charter": {"type": "object"},
            "staff_model": {"type": "string", "description": "Model id that is the staff's voice (default from config)."},
            **_WHO,
        }, "required": ["charter", "by", "role", "family"]},
    ),
    Tool(
        name="vis-step",
        description="Fill one stage of a case. Either hand in `payload` (with by/role/family), or set `draft_with` "
                    "to model ids and VIS asks those models, validates their JSON and stores it marked with who "
                    "answered. Stages: hypotheses, hypotheses_add, assumptions, estimate, red_team, evidence, "
                    "human_hand (the decision owner only), prose (the brief's text). Returns the stage's checks.",
        inputSchema={"type": "object", "properties": {
            "case_id": {"type": "string"},
            "stage": {"type": "string", "enum": list(vis_case.STAGES)},
            "payload": {"description": "The stage object (or list) when handed in."},
            "draft_with": {"type": "array", "items": {"type": "string"},
                           "description": "Model ids to draft with. For red_team, default = config red_team."},
            "draft_role": {"type": "string", "enum": ["staff", "red_team"], "default": "staff"},
            **_WHO,
        }, "required": ["case_id", "stage"]},
    ),
    Tool(
        name="vis-gate",
        description="Run the gate (pure code): checks every stage, the 70–100 confidence range and the reversibility "
                    "bar; returns the allowed GO / LEARN / STOP, the recommendation, failures and warnings.",
        inputSchema={"type": "object", "properties": {"case_id": {"type": "string"}}, "required": ["case_id"]},
    ),
    Tool(
        name="vis-decide",
        description="Record the decision owner's call (GO / LEARN / STOP). Only the decision owner decides; a "
                    "choice other than the recommendation is recorded as an override.",
        inputSchema={"type": "object", "properties": {
            "case_id": {"type": "string"},
            "choice": {"type": "string", "enum": ["GO", "LEARN", "STOP"]},
            "by": {"type": "string"}, "reason": {"type": "string"},
        }, "required": ["case_id", "choice", "by", "reason"]},
    ),
    Tool(
        name="vis-brief",
        description="Assemble the Decision Brief (numbers from code, prose from the 'prose' stage) and issue the "
                    "DecisionReceipt. Needs the after-action review date.",
        inputSchema={"type": "object", "properties": {
            "case_id": {"type": "string"},
            "aar_date": {"type": "string", "description": "YYYY-MM-DD"},
        }, "required": ["case_id", "aar_date"]},
    ),
    Tool(
        name="vis-show",
        description="Show a case (state, gate, who answered, what is next), or list the cases.",
        inputSchema={"type": "object", "properties": {"case_id": {"type": "string"}}},
    ),
]

STORE = vis_case.CaseStore()
# Drafts are long structured answers and some models reason first: give them room.
DRAFT_MAX_TOKENS = int(DEFAULTS.get("draft_max_tokens", 16000))


async def _draft_ask(m: ModelConfig, system: str, user: str) -> str:
    return await client.ask(m.model_copy(update={"max_tokens": max(m.max_tokens, DRAFT_MAX_TOKENS)}), system, user)


def _model(model_id: str) -> ModelConfig:
    for m in MODELS:
        if m.id == model_id:
            return m
    raise ValueError(f"unknown model {model_id!r}; see list-models")


def _errors(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "Refused — the object does not validate:\n" + "\n".join(
            f"- {'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:12])
    if isinstance(exc, KeyError):
        return f"Refused — missing argument {exc}"
    return f"Refused — {exc}"


async def vis_call(name: str, args: Dict[str, Any]) -> str:
    if name == "vis-start":
        staff_id = args.get("staff_model") or DEFAULTS.get("staff_model")
        if not staff_id:
            raise ValueError("no staff model: pass staff_model or set defaults.staff_model in config.yaml")
        record = vis_case.start_case(STORE, args["charter"], vis_case.model_family(_model(staff_id)),
                                     by=args["by"], role=args["role"], family=args["family"])
        return vis_case.summary(record)
    if name == "vis-step":
        case_id, stage = args["case_id"], args["stage"]
        if args.get("draft_with") or (stage == "red_team" and "payload" not in args):
            role = args.get("draft_role") or ("red_team" if stage == "red_team" else "staff")
            ids = args.get("draft_with") or (DEFAULTS.get("red_team") if role == "red_team" else
                                             [DEFAULTS.get("staff_model")])
            models = [_model(i) for i in ids if i]
            lines = await vis_case.draft_stage(STORE, case_id, stage, models, role, _draft_ask)
            return "\n".join(lines) + "\n\n" + vis_case.summary(STORE.load(case_id))
        if "payload" not in args:
            raise ValueError("hand in a payload, or set draft_with")
        record, feedback = vis_case.submit_stage(STORE, case_id, stage, args["payload"], by=args["by"],
                                                 role=args["role"], family=args["family"])
        return ("Checks: " + "; ".join(feedback) + "\n\n" if feedback else "") + vis_case.summary(record)
    if name == "vis-gate":
        return vis_case.summary(vis_case.gate_case(STORE, args["case_id"]))
    if name == "vis-decide":
        record = vis_case.decide_case(STORE, args["case_id"], args["choice"], by=args["by"], reason=args["reason"])
        return vis_case.summary(record)
    if name == "vis-brief":
        _, page, path = vis_case.brief_case(STORE, args["case_id"], date.fromisoformat(args["aar_date"]))
        return page + f"\n_Saved in the private case folder as {path.name}._"
    if name == "vis-show":
        if args.get("case_id"):
            return vis_case.summary(STORE.load(args["case_id"]))
        cases = STORE.list()
        return "\n".join(f"- {c}" for c in cases) or "(no cases yet)"
    raise ValueError(f"unknown tool {name}")


def _render(result: AskAllResult) -> str:
    lines = [f"# ask-all — {len(result.responses)} models\n", f"**Question:** {result.question}\n"]
    for r in result.responses:
        head = f"## {r.model_id}" + ("" if r.ok else " — ERROR")
        body = r.content if r.ok else (r.error or "unknown error")
        lat = f"  _({r.latency_ms} ms)_" if r.latency_ms is not None else ""
        lines.append(f"{head}{lat}\n\n{body}\n")
    if result.summary:
        lines.append(f"---\n\n## Synthesis\n\n{result.summary}\n")
    return "\n".join(lines)


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    if name == "ask-all":
        result = await ask_all(
            question=arguments["question"],
            system=arguments.get("system", ""),
            only=arguments.get("only"),
            synthesize=bool(arguments.get("synthesize", False)),
        )
        return CallToolResult(content=[TextContent(type="text", text=_render(result))])
    if name == "list-models":
        text = "\n".join(f"- `{m.id}` — {m.name} ({m.provider}: {m.model_id}, family {vis_case.model_family(m)})"
                         for m in MODELS)
        return CallToolResult(content=[TextContent(type="text", text=text or "(no models)")])
    if name.startswith("vis-"):
        try:
            text = await vis_call(name, arguments)
        except (ValueError, ValidationError, KeyError) as exc:
            text = _errors(exc)
        return CallToolResult(content=[TextContent(type="text", text=text)])
    return CallToolResult(content=[TextContent(type="text", text=f"Unknown tool: {name}")])


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
