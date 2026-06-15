"""v0 smoke tests — no network. Validates schemas, config parse, canonical-json parity."""
import os
from pathlib import Path

from vis_schemas import (
    AdvisoryResponse, AskAllResult, ModelConfig, canonical_json, sha256_hex,
)


def test_schemas_roundtrip():
    r = AdvisoryResponse(model_id="opus", ok=True, content="hi", latency_ms=12)
    assert AdvisoryResponse.model_validate_json(r.model_dump_json()) == r
    res = AskAllResult(question="q", responses=[r])
    assert res.responses[0].ok


def test_canonical_json_parity():
    # sorted keys, compact separators, ensure_ascii=False, null kept (not dropped)
    cj = canonical_json({"b": 1, "a": None, "z": "č"})
    assert cj == '{"a":null,"b":1,"z":"č"}'
    assert len(sha256_hex(cj)) == 64


def test_config_parses():
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert "models" in raw and len(raw["models"]) >= 1
    # model entries validate as ModelConfig shape (env not interpolated here)
    for m in raw["models"]:
        ModelConfig.model_validate(m)
