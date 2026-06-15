"""VIS-MCP schemas — single source of truth (v0).

v0 carries only what the multi-model fan-out needs, plus the canonical-hashing
seam so v1 (decision pipeline) and v2 (evidence anchoring) extend this same file
without rework. Canonical JSON is byte-identical to the evidence backend's rule
(sorted keys, compact separators, ensure_ascii=False, null NOT dropped).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

Provider = Literal["openrouter", "xai", "anthropic"]


class ModelConfig(BaseModel):
    id: str                       # short handle, e.g. "opus"
    name: str                     # human name
    provider: Provider
    model_id: str                 # provider catalog slug (CONFIRM against live catalog)
    temperature: Optional[float] = 0.3
    max_tokens: int = 4096
    timeout: int = 120
    # v1 seam (unused in v0): "native" | "tool_use" | "prompt_parse"
    structured_output: Literal["native", "tool_use", "prompt_parse"] = "native"


class AdvisoryResponse(BaseModel):
    model_id: str                 # the ModelConfig.id that answered
    ok: bool
    content: str = ""
    error: Optional[str] = None
    latency_ms: Optional[int] = None


class AskAllResult(BaseModel):
    question: str
    responses: List[AdvisoryResponse]
    summary: Optional[str] = None


# ---- canonical hashing seam (matches the evidence backend byte-for-byte) ----
def _to_jsonable(data: Any) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    return data


def canonical_json(data: Any) -> str:
    return json.dumps(
        _to_jsonable(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
