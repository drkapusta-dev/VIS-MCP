# CLAUDE.md — VIS-MCP working agreement (binding)

This is an **experimental** repo, isolated on purpose. Any agent working here must hold these boundaries.

## Boundaries
1. **VIS emits evidence; it is not an authority.** Output is decision-support, never a verdict.
2. **Never call a finality / settlement layer** from this repo.
3. **Talk to the evidence backend only by emitting a reference object (a dict) over a boundary**
   (HTTP / port / file). **Do NOT import the backend's code** into this repo — that would re-couple the
   experiment to a production codebase. The seam is a data contract (see `docs/EVIDENCE_SEAM.md`), not an import.
4. **Schema is single-source** (`vis_schemas.py`). No regex validation; typed objects only.
5. **Blinding is deterministic + recorded; re-attachment is pure code** (never an LLM). The model never makes
   the final pass/fail — code restores true labels before synthesis.
6. **Claim discipline:** no qualified / legal / certification / production-finality claims anywhere.
7. **Secrets via environment only.** Never commit keys; `config.yaml` is `${ENV_VAR}`-only, fail-closed.

## Build-forward
Minimal primitive → prove with tests → narrow scope → fail closed → no defensive boundary essays.

## Note on visibility
This repo is currently public. Keep internal/production architecture names, private paths, and internal
contracts OUT of committed files until/unless the repo is made private.
