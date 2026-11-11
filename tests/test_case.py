"""v1 step 2 — Quick VIS over the case store. Models are faked; no network."""
import asyncio
import json
import stat
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

import vis_case
import vis_core as core
from vis_schemas import ModelConfig

T0 = datetime(2026, 11, 11, 14, 0, tzinfo=timezone.utc)
OWNER = "Decision Owner"

STAFF = ModelConfig(id="qwen", name="Qwen", provider="openrouter", model_id="qwen/qwen3.7-max")
SAME_FAMILY = ModelConfig(id="qwen-b", name="Qwen B", provider="openrouter", model_id="qwen/qwen3.6-plus")
RED_A = ModelConfig(id="opus", name="Opus", provider="openrouter", model_id="anthropic/claude-opus-4.8")
RED_B = ModelConfig(id="grok", name="Grok", provider="xai", model_id="grok-4.3")
BROKEN = ModelConfig(id="broken", name="Broken", provider="openrouter", model_id="mistralai/mistral-large-2512")


def charter(case_id="quick-001"):
    return {
        "case_id": case_id, "mode": "quick",
        "intent": {"why": "Demand is unproven.", "what": "A course decision.", "by_when": "Q1",
                   "constraints": ["budget 20k"],
                   "statement": "Find the fastest low-risk way to validate demand before committing resources."},
        "question": {"decision_maker": "the school", "action": "launch a short online course",
                     "by_when": "31 March", "constraints": ["budget 20k"], "outcome": "a paying first cohort"},
        "roles": {"decision_owner": OWNER, "recommender": "Recommender"},
        "deadline": "2026-12-01", "reversibility": "reversible", "reversibility_set_by": OWNER,
        "statements": [{"text": "40 people asked last year.", "label": "fact", "source_ref": "crm:report-2025"}],
        "tags": ["courses"],
    }


HYPOTHESES = {"items": [
    {"id": "H1", "title": "Full course in Q1", "kind": "preferred", "assumptions": ["demand at full price", "faculty"],
     "probe_variable": "price", "kill_tests": [{"text": "pre-sell", "variable": "price", "reject_if": "< 10 sales"}],
     "proposed_by": "human"},                                           # a model claiming to be human: overwritten
    {"id": "H2", "title": "Paid pilot", "kind": "alternative", "assumptions": ["pilot predicts demand"],
     "probe_variable": "signups", "kill_tests": [{"text": "pilot", "variable": "signups", "reject_if": "< 12"}],
     "proposed_by": "x"},
    {"id": "H3", "title": "Wait until Q2", "kind": "minimal_change", "assumptions": [],
     "probe_variable": "waitlist", "kill_tests": [{"text": "waitlist", "variable": "waitlist", "reject_if": "< 20"}],
     "proposed_by": "x"},
]}
RED = {"premortem": "Only friends bought.", "strongest_argument": "Hall talk is not demand.",
       "failure_scenarios": [{"text": "friends only", "watch_signal": "no strangers"},
                             {"text": "price shock", "watch_signal": "checkout drop"}],
       "mitigations": ["sell to strangers first"], "biases": []}
PROSE = {"main_risks": ["friends-only demand"], "recommendation_reason": "a cheap reversible test",
         "next_action": {"action": "open the pilot", "owner": "Recommender", "deadline": "2026-12-15",
                         "success_metric": "12 strangers pay", "kill_criterion": "< 12 by 15 Jan"},
         "ooda_trigger": "reassess when sign-ups stall", "least_disproved_reason": ""}


async def fake_ask(model, system, user):
    if model.id == "broken":
        return "I think the plan is fine."
    if "Generate 3–6 competing hypotheses" in system:
        return "<think>simplest first {</think>```json\n" + json.dumps(HYPOTHESES) + "\n```"
    if "State the probability" in system:
        return json.dumps({"percent": 76, "reason": "a pilot is cheap and reversible"})
    if "Assume the preferred option failed badly" in system:
        return "Here is my answer: " + json.dumps(RED)
    if "Write the brief's prose fields only" in system:
        return json.dumps(PROSE)
    raise AssertionError("unexpected prompt")


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def store(tmp_path):
    return vis_case.CaseStore(tmp_path / "cases")


def start(store, case_id="quick-001"):
    return vis_case.start_case(store, charter(case_id), staff_family="qwen", by="Recommender", role="human",
                               family="human", now=T0)


def test_quick_vis_end_to_end(store):
    start(store)
    out = run(vis_case.draft_stage(store, "quick-001", "hypotheses", [STAFF], "staff", fake_ask, now=T0))
    assert out == ["qwen: stored"]
    record = store.load("quick-001")
    assert {h.proposed_by for h in record.case.hypotheses} == {"qwen"}        # code stamps who answered

    assumptions = [{"id": "A1", "text": "the pilot predicts demand", "importance": "H", "confidence": "M",
                    "owner": "Recommender", "critical": True,
                    "kill_indicator": {"observable": "sign-ups", "threshold": "< 12 by 15 Jan",
                                       "cadence": "weekly", "action": "STOP"}}]
    _, feedback = vis_case.submit_stage(store, "quick-001", "assumptions", assumptions, by="Recommender",
                                        role="human", family="human", now=T0)
    assert feedback == []

    run(vis_case.draft_stage(store, "quick-001", "estimate", [STAFF], "staff", fake_ask, now=T0))
    red = run(vis_case.draft_stage(store, "quick-001", "red_team", [RED_A, RED_B, SAME_FAMILY, BROKEN],
                                   "red_team", fake_ask, now=T0))
    assert "qwen-b: refused — the staff's family does not count as red team" in red
    assert any(line.startswith("broken: not stored") for line in red)
    assert {"opus: stored", "grok: stored"} <= set(red)

    record = vis_case.gate_case(store, "quick-001", now=T0)
    assert record.gate.recommendation == "GO" and record.gate.confidence == 76 and record.gate.headroom == 6

    record = vis_case.decide_case(store, "quick-001", "GO", by=OWNER, reason="agree", now=T0)
    assert record.decision.override is False
    run(vis_case.draft_stage(store, "quick-001", "prose", [STAFF], "staff", fake_ask, now=T0))
    record, page, path = vis_case.brief_case(store, "quick-001", date(2026, 11, 20), now=T0)
    assert core.verify_receipt(record.receipt) and record.receipt.decision.choice == "GO"
    assert "Based on the available evidence, VIS assesses GO at 76 %" in page
    assert path.read_text(encoding="utf-8") == page
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700

    summary = vis_case.summary(store.load("quick-001"))
    assert "qwen (staff, qwen/qwen3.7-max)" in summary and record.receipt.receipt_hash in summary


def test_a_change_clears_what_rested_on_it(store):
    test_quick_vis_end_to_end(store)
    vis_case.submit_stage(store, "quick-001", "estimate", 72, by=OWNER, role="human", family="human", now=T0)
    record = store.load("quick-001")
    assert record.gate is None and record.decision is None and record.brief is None and record.receipt is None
    assert record.prose is not None                                          # prose survives; numbers do not


def test_who_may_do_what(store):
    start(store)
    with pytest.raises(ValueError):
        vis_case.submit_stage(store, "quick-001", "human_hand", {}, by="qwen", role="staff", family="qwen")
    with pytest.raises(ValueError):
        run(vis_case.draft_stage(store, "quick-001", "hypotheses", [STAFF], "human", fake_ask))
    with pytest.raises(ValueError):
        run(vis_case.draft_stage(store, "quick-001", "red_team", [SAME_FAMILY], "red_team", fake_ask))
    vis_case.gate_case(store, "quick-001", now=T0)
    with pytest.raises(ValueError):
        vis_case.decide_case(store, "quick-001", "GO", by="qwen", reason="r")


def test_store_refuses_bad_ids_duplicates_and_bad_objects(store):
    with pytest.raises(ValueError):
        store.load("../etc/passwd")
    with pytest.raises(ValueError):
        vis_case.start_case(store, charter("lessons_pack"), "qwen", "R", "human", "human")
    start(store)
    with pytest.raises(ValueError):
        start(store)
    with pytest.raises(ValidationError):
        vis_case.submit_stage(store, "quick-001", "assumptions",
                              [{"id": "A1", "text": "t", "importance": "H", "confidence": "M", "owner": "o",
                                "critical": True}], by="R", role="human", family="human")
    assert store.load("quick-001").case.assumptions == []                     # nothing stored
    assert store.list() == ["quick-001"]


def test_extract_json():
    assert vis_case.extract_json('{"a": 1}') == {"a": 1}
    assert vis_case.extract_json("```json\n[1, 2]\n```") == [1, 2]
    assert vis_case.extract_json("<think>{x}</think> {\"a\": 2}") == {"a": 2}
    assert vis_case.extract_json('Sure: {"a": {"b": 3}} done') == {"a": {"b": 3}}
    with pytest.raises(ValueError):
        vis_case.extract_json("no json here")


def test_lessons_reach_the_prompt(store):
    receipt_hash = "a" * 64
    lesson = core.accept_lesson(
        core.Lesson(id="L1", source_receipt_hash=receipt_hash, stage="occam", applies_to=["courses"],
                    kind="check", text="Count strangers, not alumni."), OWNER, T0)
    pack = core.add_to_pack(core.LessonsPack(), [lesson])
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "lessons_pack.json").write_text(pack.model_dump_json(), encoding="utf-8")
    record = start(store)
    assert record.case.lessons_loaded == ["L1"] and record.case.lessons_pack_hash
    seen = {}

    async def spy(model, system, user):
        seen["user"] = user
        return await fake_ask(model, system, user)

    run(vis_case.draft_stage(store, "quick-001", "hypotheses", [STAFF], "staff", spy, now=T0))
    assert "Count strangers, not alumni." in seen["user"]


def test_model_family():
    assert vis_case.model_family(STAFF) == "qwen"
    assert vis_case.model_family(RED_B) == "xai"
    assert vis_case.model_family(ModelConfig(id="l", name="L", provider="openrouter",
                                             model_id="meta-llama/llama-4-maverick")) == "meta"


def test_server_lists_the_vis_tools():
    import server
    names = {t.name for t in run(server.list_tools())}
    assert {"ask-all", "list-models", "vis-start", "vis-step", "vis-gate", "vis-decide", "vis-brief",
            "vis-show"} <= names


# ----------------------------------------------------------------------------- drafting robustness (live run, 11 Nov)


class _Status(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.response = type("R", (), {"status_code": code})()


def _scripted(answers):
    calls = []

    async def ask(model, system, user):
        calls.append(user)
        item = answers[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item
    return ask, calls


def test_a_cut_or_empty_answer_is_retried_once_with_the_error(store, monkeypatch):
    monkeypatch.setattr(vis_case, "RETRY_PAUSE_S", 0)
    start(store)
    cut = json.dumps(HYPOTHESES)[:300]                                   # cut mid-JSON, as at the token limit
    ask, calls = _scripted([cut, json.dumps(HYPOTHESES)])
    out = run(vis_case.draft_stage(store, "quick-001", "hypotheses", [STAFF], "staff", ask, now=T0))
    assert out == ["qwen: stored"] and len(calls) == 2
    assert "Your previous answer did not validate" in calls[1]
    ask, calls = _scripted([None, ""])
    out = run(vis_case.draft_stage(store, "quick-001", "hypotheses", [STAFF], "staff", ask, now=T0))
    assert len(calls) == 2 and "empty answer" in out[0]
    assert len(store.load("quick-001").case.hypotheses) == 3            # the earlier good draft stays


def test_rate_limits_are_retried_once_and_refusals_are_not(store, monkeypatch):
    monkeypatch.setattr(vis_case, "RETRY_PAUSE_S", 0)
    start(store)
    ask, calls = _scripted([_Status(429), json.dumps(HYPOTHESES)])
    assert run(vis_case.draft_stage(store, "quick-001", "hypotheses", [STAFF], "staff", ask, now=T0)) == ["qwen: stored"]
    ask, calls = _scripted([_Status(403), json.dumps(HYPOTHESES)])
    out = run(vis_case.draft_stage(store, "quick-001", "hypotheses", [STAFF], "staff", ask, now=T0))
    assert len(calls) == 1 and "403" in out[0]
