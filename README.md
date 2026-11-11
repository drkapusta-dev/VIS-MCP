# VIS-MCP

**Experimental MCP server for structured, bias-resistant decision deliberation.**

Status: **experimental · synthetic · not production.** This repo is intentionally isolated from any
production codebase.

## What it is
The VIS decision method (Vanguard Intelligence System) as an enforced pipeline. Models draft and score;
**pure code** decides every form and every number:

`Load lessons → Frame → Occam → Evidence → ACH (blinded) → Assumptions & kill indicators → Red Team →
Gate (GO / LEARN / STOP) → Decision Brief + DecisionReceipt → After action → Lessons pack`

- **Occam generates, ACH eliminates.** Hypotheses are ordered simplest-first (fewest explicit assumptions)
  in code; they die by inconsistent evidence. The least-disproved hypothesis leads, never the most
  supported one; a tie goes to the simpler hypothesis.
- **Blinding** is a deterministic, seeded, recorded permutation of hypothesis labels; scorers never see
  which option is preferred. Re-attachment to true labels is pure code.
- **Decisions are taken in the 70–100 % confidence range: 70 is the minimum, more is better.** The
  confidence number is the estimate itself; code never bends it. The brief shows how far above the bar
  a decision stands.
- **The reversibility gate:** reversible decisions → GO allowed at ≥ 70 %; irreversible or severe →
  ≥ 80 % **and** the decision owner's recorded hand. A failed gate or a tripped kill indicator takes GO
  out of the allowed set. Weak spots (fewer than 3 independent channels, a critical assumption held at
  low confidence, an unresolved item that could flip the leader) block GO on an irreversible decision and
  are shown as warnings on a reversible one.
- **Evidence weights:** one high-reliability item weighs as much as three low ones (H 3 · M 2 · L 1).
- **After action:** every brief carries its review date; accepted lessons form a versioned pack that the
  next decision loads. Lessons add checks and warnings; they never move a threshold.

## What exists today
| Part | State |
|---|---|
| `ask-all`, `list-models` (MCP tools) | live — one question to N models concurrently |
| `vis_schemas.py` — typed objects of every stage | built, tested |
| `vis_core.py` — Occam order, blinding, ACH scoring and its four diagnostics, gates, reversibility bar, brief, receipt hash, lessons pack | built, tested, no model calls |
| `vis_case.py` + MCP tools `vis-start`, `vis-step`, `vis-gate`, `vis-decide`, `vis-brief`, `vis-show` — **Quick VIS** over a private case file | built, tested (models faked in tests) |
| Standard and Red Team modes over MCP (evidence register, blinded matrices) | **not yet** (core exists) |
| A dedicated staff-voice provider | **not yet** — the staff is the configured `defaults.staff_model`, marked as such |
| Emitting the receipt to an evidence backend | dormant seam (`emit_to_sink` is a no-op) |

`ask-all synthesize=true` lets one model merge the answers. That is a convenience for reading red-team
answers side by side — never the synthesis stage, which is code.

## Architecture boundary (binding)
VIS produces **decision-support evidence** only. It:
- emits a `DecisionReceipt` as an **external evidence-artifact reference** to a pluggable evidence backend
  **over a data boundary** (HTTP / port / file) — it does **not** import the backend's code;
- **never** calls a settlement / finality layer;
- treats a reference as **not** a verification (a reference is carried, not attested).

VIS assesses; the decision owner decides. Nothing VIS writes says "AI decided".

## Claim discipline
VIS output is **decision-support ("readiness") evidence, not legal / audit / qualified certification.**
No certification, compliance, or finality claims in code, docs, tests, or commit messages.

## Layout
```
VIS-MCP/
├── server.py             # MCP server (ask-all, list-models, vis-* tools; evidence-seam emit, dormant)
├── vis_schemas.py        # Pydantic single source of truth (every stage + DecisionReceipt + canonical_json)
├── vis_core.py           # the pure-code pipeline core: ordering, blinding, scoring, gates, receipt, lessons
├── vis_case.py           # the case store and stage handlers (submit or draft, gate, decide, brief)
├── config.yaml           # env-only keys; model list; fail-closed
├── personas/             # stage personas (to come with the pipeline tools)
├── docs/EVIDENCE_SEAM.md # the reference-dict contract VIS emits (data boundary, not a code import)
├── tests/                # no network
└── pyproject.toml
```

## Quick VIS over MCP
1. `vis-start` — the charter (intent, question, roles, deadline, reversibility set by the decision owner). Loads the
   lessons pack. Quick VIS is for reversible decisions only.
2. `vis-step` — `hypotheses`, `assumptions`, `estimate`, optionally `red_team`, then `prose`. Hand a stage in
   (`payload` + `by` / `role` / `family`) or let models draft it (`draft_with`); a draft is stored only if it
   validates, and code stamps who answered. A model's draft is never `human`.
3. `vis-gate` — pure code: the allowed GO / LEARN / STOP, the confidence against its bar, failures and warnings.
4. `vis-decide` — the decision owner's call; anything other than the recommendation is recorded as an override.
5. `vis-brief` — the one-page brief and the `DecisionReceipt`, with the after-action review date.

Cases live in a private local folder: `VIS_CASES_DIR` (default `~/.vis/cases`, created 0700, files 0600), never in
this repo. The lessons pack is `lessons_pack.json` in the same folder.

## Tests
```
uv run --no-project --with pytest --with "pydantic>=2.6" --with pyyaml --with "mcp>=1.27,<2" --with httpx python -m pytest -q
```

## Secrets
API keys come from environment variables only. Never commit keys; `config.yaml` uses `${ENV_VAR}`
interpolation and fails closed if a variable is missing.
