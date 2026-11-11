# Evidence Seam — the reference VIS emits (data boundary, not a code import)

VIS hands its `DecisionReceipt` to a pluggable evidence backend by emitting a single **reference object**
(a JSON-serializable dict) across a boundary. VIS does not import backend code and does not call any
finality layer.

## Reference object (minimum contract)
- `schema_id` — fixed external-artifact-reference identifier (provided by the backend's contract).
- `schema_version` — required, non-empty.
- `source_system` — required; for this repo: `"vis"`.
- at least one of `receipt_id` / `receipt_hash` — v1 sends the `DecisionReceipt`'s own `receipt_hash`
  (sha256 over the whole receipt, which carries the brief hash and the hash of every stage object);
  `vis_core.to_reference()` builds the object and refuses a receipt whose hash does not verify.
- `metadata` — a dict (carry the full `DecisionReceipt` here) when present.
- any field ending in `_ref` must **not** be an absolute local path.

## Boundary rules
- **Reference ≠ verification.** Absent an explicit verifier result, the reference is carried, not attested.
- The actual submit/persist is implemented on the **backend** side (behind the boundary), not here.
- Canonical hashing used for `*_hash` fields must match the backend's canonicalization byte-for-byte
  (sorted keys, compact separators, non-ASCII preserved, `null` kept — do not drop null fields).
