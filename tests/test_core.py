"""v1 step 1 — the pure-code core. No network, no model.

One synthetic Standard case runs through every gate; each test bends one thing.
"""
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

import vis_core as core
from vis_schemas import (
    AfterAction, Assumption, BiasFlag, BriefProse, CaseFile, Charter, DecisionQuestion,
    Evidence, FailureScenario, HumanHand, Hypothesis, Intent, KillIndicator, KillTest,
    Lesson, LessonsPack, NextAction, ProbabilityEstimate, RedTeamReport, Roles,
    ScorerMatrix, Statement, content_hash,
)

T0 = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
OWNER = "Decision Owner"

# ----------------------------------------------------------------------------- fixtures


def charter(reversibility="reversible", mode="standard", set_by=OWNER, **kw):
    return Charter(
        case_id="case-001", mode=mode,
        intent=Intent(why="Demand is unproven.", what="A validated course decision.", by_when="Q1",
                      constraints=["budget 20k"],
                      statement="Find the fastest low-risk way to validate demand before committing resources."),
        question=DecisionQuestion(decision_maker="the school", action="launch a short online course",
                                  by_when="31 March", constraints=["budget 20k", "two faculty"],
                                  outcome="a paying first cohort"),
        roles=Roles(decision_owner=OWNER, recommender="Recommender"),
        deadline=date(2026, 12, 1), reversibility=reversibility, reversibility_set_by=set_by,
        statements=[Statement(text="Last year 40 people asked for it.", label="fact", source_ref="crm:report-2025")],
        tags=["courses"], **kw,
    )


def kt(variable="price"):
    return [KillTest(text=f"test {variable}", variable=variable, reject_if="< 20 % convert")]


def hypotheses():
    return [
        Hypothesis(id="H1", title="Full course in Q1", kind="preferred",
                   assumptions=["demand at full price", "faculty free", "platform ready"],
                   probe_variable="price", kill_tests=kt(), proposed_by="staff"),
        Hypothesis(id="H2", title="Paid pilot, one cohort", kind="alternative",
                   assumptions=["pilot predicts demand"], probe_variable="pilot signups",
                   kill_tests=kt("pilot signups"), proposed_by="staff"),
        Hypothesis(id="H3", title="Keep the programme; decide in Q2", kind="minimal_change",
                   assumptions=[], probe_variable="waitlist", kill_tests=kt("waitlist"), proposed_by="staff"),
        Hypothesis(id="H4", title="Three languages at once", kind="aggressive",
                   assumptions=["a", "b", "c", "d"], probe_variable="translation cost",
                   kill_tests=kt("translation cost"), proposed_by="red-a"),
    ]


def evidence(e5_targets="H2", groups=None):
    g = groups or ["crm", "survey", "finance", "interviews", "competitor"]
    return [
        Evidence(id="E1", statement="40 requests", source_ref="crm:requests", cutoff=date(2026, 11, 1),
                 reliability="H", channel="crm", independence_group=g[0]),
        Evidence(id="E2", statement="survey: 30 % would pay", source_ref="survey:2026-10", cutoff=date(2026, 10, 30),
                 reliability="M", channel="survey", independence_group=g[1]),
        Evidence(id="E3", statement="faculty booked to May", source_ref="finance:plan", cutoff=date(2026, 11, 2),
                 reliability="H", channel="finance", independence_group=g[2]),
        Evidence(id="E4", statement="alumni say yes in the hall", observer="R", observed_on=date(2026, 11, 3),
                 context="alumni evening", cutoff=date(2026, 11, 3), reliability="L", channel="interviews",
                 independence_group=g[3]),
        Evidence(id="E5", statement="a rival's pilot", source_ref="web:rival", cutoff=date(2026, 11, 4),
                 reliability="M", channel="competitor", independence_group=g[4],
                 designed_to_disconfirm=e5_targets),
    ]


TRUE = {
    "E1": {"H1": "C", "H2": "C", "H3": "I", "H4": "I"},
    "E2": {"H1": "I", "H2": "C", "H3": "N", "H4": "I"},
    "E3": {"H1": "I", "H2": "N", "H3": "C", "H4": "I"},
    "E4": {"H1": "C", "H2": "C", "H3": "N", "H4": "C"},
    "E5": {"H1": "N", "H2": "N", "H3": "N", "H4": "N"},
}


def with_cell(cells, eid, hid, value):
    out = {e: dict(row) for e, row in cells.items()}
    out[eid][hid] = value
    return out


def blinded(cells, blinding, scorer, family, role):
    return ScorerMatrix(scorer_id=scorer, family=family, role=role,
                        cells={e: {blinding.mapping[h]: c for h, c in row.items()} for e, row in cells.items()})


def assumptions(critical_confidence="M", tripped=False, action="STOP"):
    return [
        Assumption(id="A1", text="the pilot predicts demand", importance="H", confidence=critical_confidence,
                   owner="R", critical=True,
                   kill_indicator=KillIndicator(observable="pilot signups", threshold="< 12 by 15 Jan",
                                                cadence="weekly", action=action, tripped=tripped)),
        Assumption(id="A2", text="faculty can teach online", importance="M", confidence="M", owner="R"),
        Assumption(id="A3", text="the platform holds 40 people", importance="L", confidence="H", owner="R"),
    ]


def red_team(families=("anthropic", "openai"), cites=("E4",)):
    return [
        RedTeamReport(scorer_id=f"red-{f}", family=f, premortem="The pilot sold to friends only.",
                      strongest_argument="Hall talk is not demand.",
                      failure_scenarios=[FailureScenario(text="friends only", watch_signal="no strangers sign up"),
                                         FailureScenario(text="price shock", watch_signal="drop at checkout")],
                      mitigations=["sell to strangers first"],
                      biases=[BiasFlag(bias="availability", note="Potential bias risk: vivid hall talk.",
                                       cites=list(cites))])
        for f in families
    ]


def estimates(staff=78, reds=((72, "anthropic"), (75, "openai")), human=None):
    out = [ProbabilityEstimate(by="staff", family="qwen", role="staff", percent=staff)]
    out += [ProbabilityEstimate(by=f"red-{f}", family=f, role="red_team", percent=p) for p, f in reds]
    if human is not None:
        out.append(ProbabilityEstimate(by=OWNER, family="human", role="human", percent=human))
    return out


def case(reversibility="reversible", staff_cells=TRUE, red_cells=(TRUE, TRUE), ev=None, hyps=None,
         asm=None, rt=None, est=None, hand=None, seed=7):
    hs = hyps or hypotheses()
    b = core.make_blinding([h.id for h in hs], seed)
    mats = [blinded(staff_cells, b, "staff", "qwen", "staff")]
    mats += [blinded(c, b, f"red-{f}", f, "red_team") for c, f in zip(red_cells, ("anthropic", "openai"))]
    return CaseFile(charter=charter(reversibility), staff_family="qwen", hypotheses=hs,
                    evidence=ev or evidence(), blinding=b, matrices=mats,
                    assumptions=asm or assumptions(), red_team=rt or red_team(),
                    estimates=est or estimates(), human_hand=hand)


def prose():
    return BriefProse(least_disproved_reason="nothing in the register contradicts the pilot",
                      main_risks=["the pilot sells to friends only"],
                      recommendation_reason="a cheap reversible test",
                      next_action=NextAction(action="open pilot sign-up", owner="R", deadline=date(2026, 12, 15),
                                             success_metric="12 strangers pay", kill_criterion="< 12 by 15 Jan"),
                      ooda_trigger="reassess when sign-ups stall for two weeks")


# ----------------------------------------------------------------------------- the whole case


def test_standard_reversible_case_goes():
    diag, gate = core.run_gate(case())
    assert gate.failures == []
    assert diag.leader == "H2"
    assert gate.raw_confidence == 72            # min(staff 78, red median_low(72, 75))
    assert gate.confidence == 72 and gate.bar == 70
    assert gate.allowed == ["GO", "LEARN", "STOP"] and gate.recommendation == "GO"
    assert gate.label == "M"


def test_four_diagnostics():
    diag, _ = core.run_gate(case())
    assert diag.most_diagnostic == ["E1", "E3", "E2"]
    assert diag.most_contradicted == "H4"
    assert (diag.leader, diag.runner_up, diag.margin) == ("H2", "H3", 10)
    # without E1, H3 (no assumptions) ties H2 at zero and leads by Occam
    assert (diag.flip_item, diag.flip_reason) == ("E1", "removal")
    assert diag.disagreements == []


def test_occam_order_fewest_assumptions_first():
    assert [h.id for h in core.occam_order(hypotheses())] == ["H3", "H2", "H1", "H4"]


def test_least_disproved_leads_not_the_most_supported():
    cells = {e: {"H1": "C", "H2": "N", "H3": "I", "H4": "I"} for e in TRUE}
    cells["E5"] = {"H1": "I", "H2": "N", "H3": "I", "H4": "I"}     # one inconsistency sinks H1
    diag, _ = core.run_gate(case(staff_cells=cells, red_cells=(cells, cells)))
    assert diag.leader == "H2"                                      # five N beat four C and one I


def test_tie_goes_to_the_simpler_hypothesis():
    cells = with_cell(TRUE, "E1", "H3", "C")                       # H2 and H3 both at zero
    diag, gate = core.run_gate(case(staff_cells=cells, red_cells=(cells, cells)))
    assert (diag.leader, diag.tie_broken_by) == ("H3", "occam")
    assert "GO" in gate.allowed


def test_unresolved_tie_blocks_go():
    hs = hypotheses()
    hs[1] = hs[1].model_copy(update={"assumptions": []})            # H2 now as simple as H3
    cells = with_cell(TRUE, "E1", "H3", "C")
    diag, gate = core.run_gate(case(hyps=hs, staff_cells=cells, red_cells=(cells, cells)))
    assert diag.tie_broken_by == "unresolved"
    assert "GO" not in gate.allowed


def test_disagreement_becomes_unknown_and_the_flip_item_caps():
    dissent = with_cell(TRUE, "E1", "H2", "I")
    diag, gate = core.run_gate(case(red_cells=(TRUE, dissent)))
    assert diag.consensus["E1"]["H2"] == "?"
    assert [(d.evidence, d.hypothesis) for d in diag.disagreements] == [("E1", "H2")]
    assert (diag.flip_item, diag.flip_reason) == ("E1", "unknown_on_leader")
    assert gate.confidence == 69 and gate.recommendation == "LEARN"
    assert gate.next_observation == "E1"


# ----------------------------------------------------------------------------- the reversibility bar


def test_reversible_bar_is_70():
    _, at_bar = core.run_gate(case(est=estimates(70, ((70, "anthropic"), (71, "openai")))))
    _, under = core.run_gate(case(est=estimates(69, ((80, "anthropic"), (80, "openai")))))
    assert at_bar.recommendation == "GO"
    assert under.recommendation == "LEARN" and "GO" not in under.allowed


def test_irreversible_needs_80_and_the_owners_hand():
    high = estimates(85, ((82, "anthropic"), (84, "openai")))
    _, no_hand = core.run_gate(case("irreversible", est=high))
    assert no_hand.bar == 80 and "GO" not in no_hand.allowed
    assert any("human hand" in f for f in no_hand.failures)
    _, wrong_hand = core.run_gate(case("irreversible", est=high, hand=HumanHand(name="Someone", at=T0)))
    assert "GO" not in wrong_hand.allowed
    _, signed = core.run_gate(case("irreversible", est=high, hand=HumanHand(name=OWNER, at=T0)))
    assert signed.recommendation == "GO" and signed.confidence == 82
    _, under = core.run_gate(case("severe", est=estimates(79, ((90, "anthropic"), (90, "openai"))),
                                  hand=HumanHand(name=OWNER, at=T0)))
    assert under.recommendation == "LEARN"


def test_irreversible_needs_a_channel_designed_to_disconfirm_the_leader():
    high = estimates(90, ((90, "anthropic"), (90, "openai")))
    hand = HumanHand(name=OWNER, at=T0)
    _, gate = core.run_gate(case("irreversible", est=high, hand=hand, ev=evidence(e5_targets="H1")))
    assert any("disconfirm the leader H2" in f for f in gate.failures)
    _, few = core.run_gate(case("irreversible", est=high, hand=hand,
                                ev=evidence(groups=["a", "a", "b", "b", "b"])))
    assert any("independent channel" in f for f in few.failures)


def test_tripped_kill_indicator_takes_away_go():
    _, stop = core.run_gate(case(asm=assumptions(tripped=True, action="STOP")))
    assert stop.recommendation == "STOP" and stop.tripped == ["A1"]
    _, learn = core.run_gate(case(asm=assumptions(tripped=True, action="LEARN")))
    assert learn.recommendation == "LEARN" and "GO" not in learn.allowed


def test_caps():
    _, few = core.run_gate(case(ev=evidence(groups=["a", "a", "b", "b", "b"])))
    assert few.confidence == 69 and few.recommendation == "LEARN"
    _, low = core.run_gate(case(asm=assumptions(critical_confidence="L"),
                                est=estimates(90, ((90, "anthropic"), (90, "openai")))))
    assert low.confidence == 79 and low.recommendation == "GO"
    assert any("low confidence" in c for c in low.caps_applied)


# ----------------------------------------------------------------------------- red team and estimates


def test_the_staffs_family_is_not_red_team():
    _, gate = core.run_gate(case(rt=red_team(families=("qwen", "anthropic"))))
    assert any(f.startswith("red team: reports from 1") for f in gate.failures)


def test_no_evidence_no_bias_flag():
    _, gate = core.run_gate(case(rt=red_team(cites=("E99",))))
    assert any("no evidence, no flag" in f for f in gate.failures)


def test_one_family_is_one_red_team_voice():
    _, gate = core.run_gate(case(est=estimates(80, ((80, "anthropic"), (80, "anthropic")))))
    assert any("1 independent red-team estimate" in f for f in gate.failures)


def test_human_estimate_can_only_lower_confidence():
    _, gate = core.run_gate(case(est=estimates(human=60)))
    assert gate.confidence == 60


# ----------------------------------------------------------------------------- blinding


def test_blinding_is_deterministic_and_hides_identity():
    a = core.make_blinding(["H1", "H2", "H3", "H4"], 7)
    b = core.make_blinding(["H4", "H3", "H2", "H1"], 7)
    assert a == b and sorted(a.mapping.values()) == ["X1", "X2", "X3", "X4"]
    view = core.blinded_hypotheses(hypotheses(), a)
    assert all(set(v) == {"label", "title", "assumptions"} for v in view)
    assert all("designed_to_disconfirm" not in v for v in core.blinded_evidence(evidence()))


def test_unblind_refuses_incomplete_or_unblinded_matrices():
    b = core.make_blinding(["H1", "H2", "H3", "H4"], 7)
    eids = [e.id for e in evidence()]
    partial = {e: row for e, row in TRUE.items() if e != "E5"}
    with pytest.raises(ValueError):
        core.unblind(blinded(partial, b, "s", "qwen", "staff"), b, eids)
    missing_label = blinded(TRUE, b, "s", "qwen", "staff")
    missing_label.cells["E1"].pop(b.mapping["H1"])
    with pytest.raises(ValueError):
        core.unblind(missing_label, b, eids)
    plain = ScorerMatrix(scorer_id="s", family="qwen", role="staff", blinded=False, cells=TRUE)
    with pytest.raises(ValueError):
        core.unblind(plain, b, eids)
    _, gate = core.run_gate(case(staff_cells=partial))
    assert "GO" not in gate.allowed


# ----------------------------------------------------------------------------- schemas fail closed


def test_schemas_fail_closed():
    with pytest.raises(ValidationError):
        Statement(text="it works", label="fact")                                   # fact without source
    with pytest.raises(ValidationError):
        Intent(why="w", what="w", by_when="w", constraints=["c"], statement="One. Two. Three.")
    with pytest.raises(ValidationError):
        charter(set_by="a model")                                                  # D sets reversibility
    with pytest.raises(ValidationError):
        charter("irreversible", mode="quick")
    with pytest.raises(ValidationError):
        KillTest(text="t", variable="price and channel", reject_if="x")           # one variable per probe
    with pytest.raises(ValidationError):
        Evidence(id="E1", statement="s", cutoff=date(2026, 1, 1), reliability="H", channel="c",
                 independence_group="g")                                           # no source, no evidence
    with pytest.raises(ValidationError):
        Evidence(id="E1", statement="s", source_ref="/Users/x/file.pdf", cutoff=date(2026, 1, 1),
                 reliability="H", channel="c", independence_group="g")              # no local paths
    with pytest.raises(ValidationError):
        Assumption(id="A1", text="t", importance="H", confidence="M", owner="o", critical=True)
    with pytest.raises(ValidationError):
        Statement(text="t", label="assumption", stray="field")                      # unknown fields refused
    assert Intent(why="w", what="w", by_when="w", constraints=["c"], statement="Commit 70.5 % now.")


def test_hypothesis_set_rules():
    hs = hypotheses()
    assert core.check_hypotheses(hs, "standard") == []
    assert any("minimal-change" in f for f in core.check_hypotheses([hs[0], hs[1], hs[3]], "standard"))
    assert any("3–6" in f for f in core.check_hypotheses(hs[:2], "standard"))


# ----------------------------------------------------------------------------- decision, brief, receipt


def test_only_the_owner_decides_and_overrides_are_recorded():
    _, gate = core.run_gate(case())
    with pytest.raises(ValueError):
        core.decide(gate, charter(), "GO", by="a model", at=T0, reason="r")
    same = core.decide(gate, charter(), "GO", by=OWNER, at=T0, reason="agree")
    other = core.decide(gate, charter(), "LEARN", by=OWNER, at=T0, reason="one more test")
    assert (same.override, other.override) == (False, True)


def test_brief_numbers_come_from_code():
    c = case()
    diag, gate = core.run_gate(c)
    brief = core.assemble_brief(c, diag, gate, prose(), aar_date=date(2027, 2, 15))
    assert (brief.least_disproved, brief.confidence, brief.recommendation) == ("H2", 72, "GO")
    assert brief.assessment.startswith("Based on the available evidence, VIS assesses H2")
    assert [h.status for h in brief.hypotheses if h.id == "H2"] == ["leader"]
    assert "11. **OODA trigger:**" in core.render_brief(brief)
    with pytest.raises(ValidationError):
        BriefProse(main_risks=["r"], recommendation_reason="The AI decided to go.",
                   next_action=prose().next_action, ooda_trigger="t")


def test_receipt_hash_is_stable_and_detects_tampering():
    c = case()
    diag, gate = core.run_gate(c)
    brief = core.assemble_brief(c, diag, gate, prose(), aar_date=date(2027, 2, 15))
    decision = core.decide(gate, c.charter, "GO", by=OWNER, at=T0, reason="agree")
    r1 = core.issue_receipt(c, diag, gate, brief, created_at=T0, decision=decision)
    r2 = core.issue_receipt(c, diag, gate, brief, created_at=T0, decision=decision)
    assert r1.receipt_hash == r2.receipt_hash and len(r1.receipt_hash) == 64
    assert core.verify_receipt(r1)
    assert not core.verify_receipt(r1.model_copy(update={"recommendation": "STOP"}))
    ref = core.to_reference(r1, schema_id="external-artifact-reference", schema_version="1")
    assert ref["source_system"] == "vis" and ref["receipt_hash"] == r1.receipt_hash
    with pytest.raises(ValueError):
        core.to_reference(r1, schema_id="", schema_version="1")


# ----------------------------------------------------------------------------- quick mode


def test_quick_vis_reversible_only_and_without_matrix():
    hs = hypotheses()[1:]                                    # H2, H3, H4
    c = CaseFile(charter=charter(mode="quick"), staff_family="qwen", hypotheses=hs,
                 assumptions=assumptions()[:1], estimates=estimates(75, ()))
    diag, gate = core.run_gate(c)
    assert diag is None and gate.recommendation == "GO" and gate.label == "L"
    brief = core.assemble_brief(c, diag, gate, prose(), aar_date=date(2026, 11, 20))
    assert brief.least_disproved is None


# ----------------------------------------------------------------------------- after action, lessons pack


def test_after_action_and_the_lessons_pack():
    c = case()
    diag, gate = core.run_gate(c)
    brief = core.assemble_brief(c, diag, gate, prose(), aar_date=date(2027, 2, 15))
    receipt = core.issue_receipt(c, diag, gate, brief, created_at=T0)

    lesson = Lesson(id="L1", source_receipt_hash=receipt.receipt_hash, stage="evidence",
                    applies_to=["courses"], kind="check", text="Count strangers, not alumni, as demand.")
    proposal = Lesson(id="L2", source_receipt_hash=receipt.receipt_hash, stage="gate",
                      applies_to=["*"], kind="threshold_proposal", text="Raise the bar for courses to 75 %.")
    aar = AfterAction(receipt_hash=receipt.receipt_hash, held_on=date(2027, 2, 15), planned="12 strangers",
                      happened="7 strangers", learned="alumni inflate demand", next_step="widen the channel",
                      leader_held=False, assumption_failures=["A1"], lessons=[lesson, proposal])
    assert core.check_after_action(aar, receipt, c.assumptions) == []
    assert core.forecast_score(72, True) == 0.0784 and core.forecast_score(72, False) == 0.5184

    with pytest.raises(ValidationError):
        core.add_to_pack(LessonsPack(), [lesson])                    # PROPOSED lessons stay out
    pack = core.add_to_pack(LessonsPack(), [core.accept_lesson(lesson, OWNER, T0),
                                             core.accept_lesson(proposal, OWNER, T0)])
    assert pack.version == 1
    assert [l.id for l in core.lessons_for(pack, ["courses"])] == ["L1", "L2"]
    assert [l.id for l in core.lessons_for(pack, ["hiring"])] == ["L2"]

    loaded, lessons = core.load_lessons(c, pack)
    assert loaded.lessons_pack_hash == content_hash(pack) and loaded.lessons_loaded == ["L1", "L2"]
    _, gate_with = core.run_gate(loaded)
    assert gate_with == gate and core.BAR["reversible"] == 70        # a lesson never moves the bar

    stray = lesson.model_copy(update={"source_receipt_hash": "0" * 64})
    bad = aar.model_copy(update={"lessons": [stray]})
    assert any("cites another battle" in f for f in core.check_after_action(bad, receipt, c.assumptions))
