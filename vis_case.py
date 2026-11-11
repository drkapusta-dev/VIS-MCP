"""VIS-MCP case store and stage handlers (v1, step 2: Quick VIS over MCP).

A case is one decision. It lives as one JSON file in a private local folder
(``VIS_CASES_DIR``, default ``~/.vis/cases``) — never in this repository.

Two ways a stage gets filled:

- **submitted** — someone hands in the stage object (a person, or the MCP
  client acting as staff); ``by`` and ``role`` say who;
- **drafted** — VIS asks a configured model, parses its JSON, validates it, and
  stores it. Code stamps who answered (model id, family, role) on the object;
  a model can never claim to be a human or someone else.

Either way the object passes the same schema, and the gate is pure code
(``vis_core.run_gate``). Changing any stage clears the gate, decision, brief
and receipt that rested on it.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import Field

import vis_core as core
from vis_schemas import (
    Assumption,
    BriefProse,
    CaseFile,
    CaseRecord,
    Charter,
    Evidence,
    FailureScenario,
    BiasFlag,
    HumanHand,
    Hypothesis,
    Lesson,
    LessonsPack,
    LogEntry,
    ModelConfig,
    ProbabilityEstimate,
    RedTeamReport,
    Recommendation,
    Strict,
)

Ask = Callable[[ModelConfig, str, str], Awaitable[str]]

STAGES = ("hypotheses", "hypotheses_add", "assumptions", "estimate", "red_team", "evidence",
          "human_hand", "prose")
DRAFTABLE = ("hypotheses", "hypotheses_add", "assumptions", "estimate", "red_team", "prose")
_SAFE_ID = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
_RESERVED = {"lessons_pack"}
_FAMILY_ALIASES = {"x-ai": "xai", "moonshotai": "moonshot", "meta-llama": "meta", "mistralai": "mistral"}


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def default_cases_dir() -> Path:
    return Path(os.getenv("VIS_CASES_DIR", str(Path.home() / ".vis" / "cases"))).expanduser()


def model_family(m: ModelConfig) -> str:
    if m.family:
        return m.family
    raw = m.model_id.split("/")[0] if "/" in m.model_id else m.provider
    return _FAMILY_ALIASES.get(raw, raw)


# =============================================================================
# The store
# =============================================================================
class CaseStore:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else default_cases_dir()

    def _path(self, case_id: str, suffix: str = ".json") -> Path:
        if not case_id or case_id in _RESERVED or any(c not in _SAFE_ID for c in case_id):
            raise ValueError(f"case id {case_id!r}: use letters, digits, '-' and '_' only")
        return self.root / f"{case_id}{suffix}"

    def _write(self, path: Path, text: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".tmp-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def exists(self, case_id: str) -> bool:
        return self._path(case_id).exists()

    def load(self, case_id: str) -> CaseRecord:
        path = self._path(case_id)
        if not path.exists():
            raise ValueError(f"no case {case_id!r}")
        return CaseRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, record: CaseRecord) -> None:
        self._write(self._path(record.case.charter.case_id), record.model_dump_json(indent=2))

    def write_text(self, case_id: str, suffix: str, text: str) -> Path:
        path = self._path(case_id, suffix)
        self._write(path, text)
        return path

    def list(self) -> List[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.json")
                      if p.stem not in _RESERVED and not p.name.startswith("."))

    def lessons_pack(self) -> LessonsPack:
        path = self.root / "lessons_pack.json"
        if not path.exists():
            return LessonsPack()
        return LessonsPack.model_validate_json(path.read_text(encoding="utf-8"))


# =============================================================================
# Stage payloads
# =============================================================================
class Hypotheses(Strict):
    items: List[Hypothesis] = Field(min_length=1)


class Assumptions(Strict):
    items: List[Assumption] = Field(min_length=1)


class EvidenceList(Strict):
    items: List[Evidence] = Field(min_length=1)


class EstimateDraft(Strict):
    percent: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1)


class RedTeamDraft(Strict):
    premortem: str = Field(min_length=1)
    strongest_argument: str = Field(min_length=1)
    failure_scenarios: List[FailureScenario] = Field(min_length=2)
    mitigations: List[str] = Field(min_length=1)
    biases: List[BiasFlag] = Field(default_factory=list)


_DRAFT_TARGET = {
    "hypotheses": Hypotheses, "hypotheses_add": Hypotheses, "assumptions": Assumptions,
    "estimate": EstimateDraft, "red_team": RedTeamDraft, "prose": BriefProse,
}


def _items(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list):
        return {"items": payload}
    if isinstance(payload, dict) and "items" in payload:
        return payload
    return {"items": [payload]}


def _invalidate(record: CaseRecord, stage: str) -> Dict[str, Any]:
    if stage == "prose":
        return {"brief": None, "receipt": None}
    return {"diagnostics": None, "gate": None, "decision": None, "brief": None, "receipt": None}


# =============================================================================
# Start, submit
# =============================================================================
def start_case(store: CaseStore, charter: Dict[str, Any], staff_family: str, by: str, role: str,
               family: str, now: Optional[datetime] = None) -> CaseRecord:
    now = now or now_utc()
    ch = Charter.model_validate(charter)
    if store.exists(ch.case_id):
        raise ValueError(f"case {ch.case_id!r} already exists")
    case, _ = core.load_lessons(CaseFile(charter=ch, staff_family=staff_family), store.lessons_pack())
    record = CaseRecord(case=case, log=[
        LogEntry(at=now, stage="frame", by=by, role=role, family=family),
        LogEntry(at=now, stage="load", by="code", role="code", family="code",
                 note=f"lessons loaded: {', '.join(case.lessons_loaded) or 'none'}"),
    ])
    store.save(record)
    return record


def submit_stage(store: CaseStore, case_id: str, stage: str, payload: Any, by: str, role: str,
                 family: str, model: Optional[str] = None, note: str = "",
                 now: Optional[datetime] = None) -> Tuple[CaseRecord, List[str]]:
    """Validate and store one stage object. Returns the record and the stage's own check results."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; one of {', '.join(STAGES)}")
    now = now or now_utc()
    record = store.load(case_id)
    case = record.case
    feedback: List[str] = []
    update: Dict[str, Any] = {}

    if stage in ("hypotheses", "hypotheses_add"):
        data = _items(payload)
        data = {"items": [{**item, "proposed_by": by} for item in data["items"]]}   # who answered
        new = Hypotheses.model_validate(data).items
        hyps = new if stage == "hypotheses" else list(case.hypotheses) + new
        case = case.model_copy(update={"hypotheses": hyps})
        feedback = core.check_hypotheses(hyps, case.charter.mode)
    elif stage == "assumptions":
        items = Assumptions.model_validate(_items(payload)).items
        case = case.model_copy(update={"assumptions": items})
        feedback = core.check_assumptions(items, case.charter.mode)
    elif stage == "evidence":
        items = EvidenceList.model_validate(_items(payload)).items
        case = case.model_copy(update={"evidence": items})
    elif stage == "estimate":
        percent = payload["percent"] if isinstance(payload, dict) else payload
        estimate = ProbabilityEstimate(by=by, family=family, role=role, percent=percent)
        if role == "red_team" and family == case.staff_family:
            feedback.append(f"{by}: the staff's family ({family}) does not count as red team")
        others = [e for e in case.estimates if e.by != by]
        case = case.model_copy(update={"estimates": others + [estimate]})
    elif stage == "red_team":
        report = RedTeamReport.model_validate({**dict(payload), "scorer_id": by, "family": family})
        reports = [r for r in case.red_team if r.scorer_id != by] + [report]
        case = case.model_copy(update={"red_team": reports})
        known = {e.id for e in case.evidence} | {a.id for a in case.assumptions}
        feedback = [f for f in core.check_red_team(reports, case.staff_family, known)
                    if not f.startswith("red team: reports from")]
        if family == case.staff_family:
            feedback.append(f"{by}: the staff's family ({family}) does not count as red team")
    elif stage == "human_hand":
        if role != "human":
            raise ValueError("only a human gives the human hand")
        case = case.model_copy(update={"human_hand": HumanHand(name=by, at=now)})
    elif stage == "prose":
        update["prose"] = BriefProse.model_validate(payload)

    update.update(_invalidate(record, stage))
    update["case"] = case
    update["log"] = list(record.log) + [LogEntry(at=now, stage=stage, by=by, role=role, family=family,
                                                 model=model, note=note)]
    record = record.model_copy(update=update)
    store.save(record)
    return record, feedback


# =============================================================================
# Drafting by a model
# =============================================================================
_STAGE_RULES = {
    "hypotheses": (
        "Generate 3–6 competing hypotheses (options), simplest first: fewest explicit assumptions. Include the "
        "current / preferred option (kind 'preferred'), at least one alternative, a minimal-change / do-nothing / "
        "delay option (kind 'minimal_change'), and an aggressive or asymmetric option when relevant. For each: its "
        "explicit assumptions, ONE probe variable (a single variable, no 'and'), and at least one kill test — the "
        "fastest inexpensive test that could falsify it — with a reject_if threshold. Ids H1, H2, …"
    ),
    "hypotheses_add": (
        "Add 1–2 hypotheses from OUTSIDE the current frame that the team has not considered (kind 'wildcard' or "
        "'alternative'), each with assumptions, one probe variable and a kill test. Continue the ids after the "
        "existing ones."
    ),
    "assumptions": (
        "List the key hidden assumptions behind the preferred option — at least 3 for Standard VIS; for Quick VIS "
        "at least the biggest one; more is welcome. Mark as critical the 1–2 that would damage the recommendation "
        "most if wrong. Each critical one needs a kill indicator: an observable signal, a threshold, a monitoring "
        "cadence and the pre-set action (GO, LEARN or STOP). Ids A1, A2, …"
    ),
    "estimate": (
        "State the probability, 0–100, that the preferred option (or, if named, the least-disproved hypothesis) "
        "holds, and the reason in one sentence. Decisions are taken in the 70–100 range; never inflate a number "
        "to reach it."
    ),
    "red_team": (
        "Assume the preferred option failed badly. Give the premortem (what most likely caused it), the strongest "
        "argument against the option, two or more failure scenarios each with the signal that would show it "
        "developing, and mitigations. Flag a bias (confirmation, availability, overconfidence, anchoring, "
        "mirror_imaging, sunk_cost, status_quo) only where you can cite an evidence or assumption id from the case; "
        "otherwise flag none. Optimise for discovering what the team missed, not for agreement."
    ),
    "prose": (
        "Write the brief's prose fields only. Never state a number, the leader or the recommendation — code fills "
        "those. Fields: main_risks, recommendation_reason (why the recommendation the gate allows makes sense), "
        "next_action (action, owner, deadline YYYY-MM-DD, success_metric, kill_criterion), ooda_trigger "
        "('reassess when …'), least_disproved_reason (Standard VIS only; else an empty string)."
    ),
}


def _context(record: CaseRecord, lessons: Sequence[Lesson]) -> str:
    case = record.case
    ch = case.charter
    lines = [
        f"Mode: {ch.mode} VIS · reversibility: {ch.reversibility} · deadline {ch.deadline.isoformat()}",
        f"Commander's intent: {ch.intent.statement}",
        f"Decision question: {ch.question.text()}",
        f"Constraints: {', '.join(ch.intent.constraints)}",
    ]
    for s in ch.statements:
        lines.append(f"- {s.label}: {s.text}" + (f" [{s.source_ref}]" if s.source_ref else ""))
    for h in core.occam_order(case.hypotheses):
        lines.append(f"{h.id} ({h.kind}) {h.title} — assumptions: {'; '.join(h.assumptions) or 'none'}")
    for a in case.assumptions:
        lines.append(f"{a.id} {a.text} (importance {a.importance}, confidence {a.confidence}"
                     + (", critical" if a.critical else "") + ")")
    for e in case.evidence:
        lines.append(f"{e.id} {e.statement} (reliability {e.reliability}, channel {e.channel})")
    if record.gate is not None:
        g = record.gate
        lines.append(f"Gate (code): {g.recommendation}, confidence {g.confidence} % against the {g.bar} % bar; "
                     f"allowed {', '.join(g.allowed)}; warnings: {'; '.join(g.warnings) or 'none'}")
    if lessons:
        lines.append("Lessons from earlier decisions that apply:")
        lines += [f"- [{l.stage}] {l.text}" for l in lessons]
    return "\n".join(lines)


def build_prompt(record: CaseRecord, stage: str, role: str, lessons: Sequence[Lesson]) -> Tuple[str, str]:
    who = "the staff" if role == "staff" else "a red-team analyst"
    schema = _DRAFT_TARGET[stage].model_json_schema()
    system = (
        f"You are {who} of VIS (Vanguard Intelligence System), a decision-support method. AI structures and "
        "challenges; humans decide — never write that AI decided. Separate facts from assumptions; never present "
        f"an assumption as a fact.\n\n{_STAGE_RULES[stage]}\n\n"
        "Answer with JSON only — no prose before or after — that validates against this JSON Schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    return system, _context(record, lessons)


def extract_json(text: str) -> Any:
    """The JSON object or array in a model's answer (tolerates a thinking block or code fences)."""
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>", 1)[1].strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rstrip()
        if t.endswith("```"):
            t = t[:-3]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    end = max(t.rfind("}"), t.rfind("]"))
    if not starts or end <= min(starts):
        raise ValueError("no JSON in the answer")
    return json.loads(t[min(starts):end + 1])


def _short(exc: Exception, limit: int = 400) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[:limit] + " …"


RETRY_PAUSE_S = 3.0
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def _http_status(exc: Exception) -> Optional[int]:
    return getattr(getattr(exc, "response", None), "status_code", None)


def _parse_draft(stage: str, model: ModelConfig, answer: Optional[str]):
    if not answer or not answer.strip():
        raise ValueError("empty answer — the model may have spent its whole token budget on reasoning")
    data = extract_json(answer)
    if stage in ("hypotheses", "hypotheses_add"):
        data = _items(data)
        data = {"items": [{**item, "proposed_by": model.id} for item in data["items"]]}
    return _DRAFT_TARGET[stage].model_validate(data)


async def _draft_one(model: ModelConfig, stage: str, system: str, user: str, ask: Ask) -> Tuple[Optional[dict], str, str]:
    """Two attempts at most. A rate limit or server error is retried once after a pause; an answer
    that does not parse or validate is retried once with the error shown to the model."""
    prompt, error = user, ""
    for attempt in (1, 2):
        try:
            answer = await ask(model, system, prompt)
        except Exception as exc:                   # one model failing must not sink the rest
            error = _short(exc)
            if attempt == 1 and _http_status(exc) in _RETRY_STATUS:
                await asyncio.sleep(RETRY_PAUSE_S)
                continue
            return None, "", error
        try:
            draft = _parse_draft(stage, model, answer)
            note = draft.reason if isinstance(draft, EstimateDraft) else ""
            return draft.model_dump(mode="json"), note, ""
        except Exception as exc:
            error = _short(exc)
            prompt = (f"{user}\n\nYour previous answer did not validate:\n{error}\n\n"
                      "Answer again with the corrected JSON only.")
    return None, "", error


async def draft_stage(store: CaseStore, case_id: str, stage: str, models: Sequence[ModelConfig],
                      role: str, ask: Ask, now: Optional[datetime] = None) -> List[str]:
    """Ask each model to draft the stage; store every answer that validates, marked with who answered."""
    if stage not in DRAFTABLE:
        raise ValueError(f"stage {stage!r} is not drafted by a model; one of {', '.join(DRAFTABLE)}")
    if role not in ("staff", "red_team"):
        raise ValueError("a model drafts as staff or red_team, never as a human")
    if not models:
        raise ValueError("no model to draft with")
    record = store.load(case_id)
    if role == "red_team":
        refused = [m.id for m in models if model_family(m) == record.case.staff_family]
        models = [m for m in models if model_family(m) != record.case.staff_family]
        if not models:
            raise ValueError(f"every model asked is of the staff's family ({record.case.staff_family})")
    else:
        refused = []
    lessons = [l for l in core.lessons_for(store.lessons_pack(), record.case.charter.tags)
               if l.id in record.case.lessons_loaded]
    system, user = build_prompt(record, stage, role, lessons)
    results = await asyncio.gather(*[_draft_one(m, stage, system, user, ask) for m in models])
    lines = [f"{m}: refused — the staff's family does not count as red team" for m in refused]
    for model, (payload, note, error) in zip(models, results):
        if payload is None:
            lines.append(f"{model.id}: not stored — {error}")
            continue
        body = payload.get("items") if stage in ("hypotheses", "hypotheses_add", "assumptions") else payload
        _, feedback = submit_stage(store, case_id, stage, body, by=model.id, role=role,
                                   family=model_family(model), model=model.model_id, note=note, now=now)
        lines.append(f"{model.id}: stored" + (f" — {'; '.join(feedback)}" if feedback else ""))
    return lines


# =============================================================================
# Gate, decision, brief
# =============================================================================
def gate_case(store: CaseStore, case_id: str, now: Optional[datetime] = None) -> CaseRecord:
    now = now or now_utc()
    record = store.load(case_id)
    diagnostics, gate = core.run_gate(record.case)
    record = record.model_copy(update={
        "diagnostics": diagnostics, "gate": gate, "decision": None, "brief": None, "receipt": None,
        "log": list(record.log) + [LogEntry(at=now, stage="gate", by="code", role="code", family="code",
                                            note=f"{gate.recommendation} at {gate.confidence} %")],
    })
    store.save(record)
    return record


def decide_case(store: CaseStore, case_id: str, choice: Recommendation, by: str, reason: str,
                now: Optional[datetime] = None) -> CaseRecord:
    now = now or now_utc()
    record = store.load(case_id)
    if record.gate is None:
        raise ValueError("run the gate first")
    decision = core.decide(record.gate, record.case.charter, choice, by=by, at=now, reason=reason)
    record = record.model_copy(update={
        "decision": decision, "brief": None, "receipt": None,
        "log": list(record.log) + [LogEntry(at=now, stage="decision", by=by, role="human", family="human",
                                            note=f"{choice}" + (" (override)" if decision.override else ""))],
    })
    store.save(record)
    return record


def brief_case(store: CaseStore, case_id: str, aar_date: date,
               now: Optional[datetime] = None) -> Tuple[CaseRecord, str, Path]:
    now = now or now_utc()
    record = store.load(case_id)
    if record.gate is None:
        raise ValueError("run the gate first")
    if record.prose is None:
        raise ValueError("the brief's prose is missing (stage 'prose')")
    brief = core.assemble_brief(record.case, record.diagnostics, record.gate, record.prose, aar_date)
    receipt = core.issue_receipt(record.case, record.diagnostics, record.gate, brief, created_at=now,
                                 decision=record.decision)
    page = core.render_brief(brief) + f"\nReceipt: `{receipt.receipt_hash}`\n"
    path = store.write_text(case_id, ".brief.md", page)
    record = record.model_copy(update={
        "brief": brief, "receipt": receipt,
        "log": list(record.log) + [LogEntry(at=now, stage="brief", by="code", role="code", family="code",
                                            note=f"receipt {receipt.receipt_hash[:12]}")],
    })
    store.save(record)
    return record, page, path


# =============================================================================
# Reading a case
# =============================================================================
def next_steps(record: CaseRecord) -> List[str]:
    case = record.case
    todo: List[str] = []
    if len(case.hypotheses) < core.HYPOTHESES_MIN:
        todo.append("hypotheses (3–6, with a minimal-change option)")
    if not any(a.critical for a in case.assumptions):
        todo.append("assumptions (at least the biggest one, marked critical, with its kill indicator)")
    if not any(e.role in ("staff", "human") for e in case.estimates):
        todo.append("estimate (the staff's or the decision owner's percent)")
    if case.charter.mode == "standard":
        todo.append("Standard VIS stages (evidence, blinded matrices, red team) — step 4")
    if record.gate is None:
        todo.append("gate")
    if record.prose is None:
        todo.append("prose (for the brief)")
    if record.brief is None:
        todo.append("brief (with its after-action date)")
    return todo


def summary(record: CaseRecord) -> str:
    case = record.case
    ch = case.charter
    lines = [
        f"# {ch.case_id} — {ch.mode} VIS · {ch.reversibility} · decision owner {ch.roles.decision_owner}",
        f"**Question:** {ch.question.text()}",
        f"**Hypotheses:** {len(case.hypotheses)} · **assumptions:** {len(case.assumptions)} · "
        f"**evidence:** {len(case.evidence)} · **estimates:** {len(case.estimates)} · "
        f"**red team:** {len(case.red_team)} · **lessons loaded:** {', '.join(case.lessons_loaded) or 'none'}",
    ]
    if record.gate is not None:
        g = record.gate
        lines.append(f"**Gate:** {g.recommendation} · {g.confidence} % ({g.headroom:+d} against {g.bar} %) · "
                     f"allowed {', '.join(g.allowed)}")
        lines += [f"- failure: {f}" for f in g.failures] + [f"- warning: {w}" for w in g.warnings]
    if record.decision is not None:
        d = record.decision
        lines.append(f"**Decision:** {d.choice} by {d.by}" + (" — override" if d.override else ""))
    if record.receipt is not None:
        lines.append(f"**Receipt:** `{record.receipt.receipt_hash}`")
    todo = next_steps(record)
    if todo:
        lines.append("**Next:** " + "; ".join(todo))
    lines.append("**Who answered:**")
    lines += [f"- {e.at.isoformat()} {e.stage}: {e.by} ({e.role}{', ' + e.model if e.model else ''})"
              + (f" — {e.note}" if e.note else "") for e in record.log[-12:]]
    return "\n".join(lines)
