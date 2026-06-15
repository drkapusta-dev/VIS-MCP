# VIS-MCP

**Experimental MCP server for structured, bias-resistant decision deliberation.**

Status: **experimental · synthetic · not production.** This repo is intentionally isolated from any
production codebase.

## What it is
An enforced decision pipeline exposed as MCP tools:

`Frame → Hypotheses → ACH (blinded) → Red Team → Synthesis (GO / LEARN / STOP)`

Each stage emits a typed object. Hypotheses are **blinded** (deterministic, seeded, recorded permutation)
before the Analysis-of-Competing-Hypotheses stage to reduce anchoring bias, then re-attached to true labels
**in pure code** (never by a model). The final brief becomes a `DecisionReceipt`.

## Architecture boundary (binding)
VIS produces **decision-support evidence** only. It:
- emits a `DecisionReceipt` as an **external evidence-artifact reference** to a pluggable evidence backend
  **over a data boundary** (HTTP / port / file) — it does **not** import the backend's code;
- **never** calls a settlement / finality layer;
- treats a reference as **not** a verification (a reference is carried, not attested).

## Claim discipline
VIS output is **decision-support ("readiness") evidence, not legal / audit / qualified certification.**
No certification, compliance, or finality claims in code, docs, tests, or commit messages.

## Layout
```
VIS-MCP/
├── server.py            # MCP server (deliberation pipeline + evidence-seam emit)
├── vis_schemas.py       # Pydantic single source of truth (stages + DecisionReceipt + Blinding + canonical_json)
├── config.yaml          # env-only keys; per-model structured-output mode; fail-closed
├── personas/            # stage personas (framer, hypothesis-generator, ach-analyst, red-team, synthesis-engine)
├── docs/EVIDENCE_SEAM.md# the reference-dict contract VIS emits (data boundary, not a code import)
└── pyproject.toml
```

## Secrets
API keys come from environment variables only. Never commit keys; `config.yaml` uses `${ENV_VAR}`
interpolation and fails closed if a variable is missing.
