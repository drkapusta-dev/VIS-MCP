"""VIS-MCP core — the pure-code part of the VIS pipeline (v1, step 1).

Nothing here calls a model or the network. Models draft and score; this file
decides every form and every number:

- the Occam order (hypotheses are born simplest-first),
- the blinding of hypothesis labels before scoring, and the unblinding after,
- the ACH aggregation and its four diagnostics (ACH eliminates: the
  least-disproved hypothesis leads, never the most supported one),
- the gates of each stage and the reversibility bar (decisions are taken in the
  70–100 % confidence range: 70 is the minimum, more is better),
- which of GO / LEARN / STOP the evidence allows,
- the Decision Brief's computed fields, the receipt hash, the lessons pack.

VIS assesses; the decision owner decides. Output is decision-support evidence,
never a verdict, a settlement or a certification.
"""
from __future__ import annotations

import random
import statistics
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple, get_args

from vis_schemas import (
    AfterAction,
    Assumption,
    Blinding,
    BriefProse,
    CaseFile,
    Cell,
    CellRef,
    Charter,
    Decision,
    DecisionBrief,
    DecisionReceipt,
    Diagnostics,
    Evidence,
    GateResult,
    Hypothesis,
    HypothesisScore,
    HypothesisStatus,
    Lesson,
    LessonsPack,
    Level,
    Mode,
    ProbabilityEstimate,
    Recommendation,
    RedTeamReport,
    ScorerMatrix,
    Stage,
    content_hash,
)

# --- The owner's ruling (11 Nov 2026): decisions are taken in the 70–100 % confidence range;
#     70 is the minimum, more is better. The confidence number is the estimate itself — code
#     never bends it; a weak spot either blocks GO or is shown as a warning.
FLOOR = 70
# --- Ratified (VIS constitution v0, amendment 1, 2026-08-20). Changed only by the owner. ---
BAR: Dict[str, int] = {"reversible": FLOOR, "irreversible": 80, "severe": 80}
HUMAN_HAND_REQUIRED = frozenset({"irreversible", "severe"})
assert all(bar >= FLOOR for bar in BAR.values())

# --- From the method's text (§4, §7, paper §8.3). ---
HYPOTHESES_MIN, HYPOTHESES_MAX = 3, 6
ASSUMPTIONS_MIN, ASSUMPTIONS_MAX = 3, 7          # Standard VIS; Quick needs only the biggest one
CRITICAL_MIN, CRITICAL_MAX = 1, 2
MIN_INDEPENDENT_CHANNELS = 3
RED_TEAM_MIN_FAMILIES = 2

# --- Set 11 Nov 2026: one high-reliability item weighs as much as three low ones. ---
RELIABILITY_WEIGHT: Dict[str, int] = {"H": 3, "M": 2, "L": 1}

STAGE_ORDER: List[str] = list(get_args(Stage))
_LABEL_WORD = {"H": "HIGH", "M": "MEDIUM", "L": "LOW"}
_RELIABILITY_RANK = {"L": 0, "M": 1, "H": 2}


def _id_key(value: str) -> Tuple[str, int]:
    """H2 before H10: sort by prefix, then by number."""
    prefix = value.rstrip("0123456789")
    number = value[len(prefix):]
    return (prefix, int(number) if number else -1)


# =============================================================================
# Stage 2 — Occam generates
# =============================================================================
def occam_order(hypotheses: Sequence[Hypothesis]) -> List[Hypothesis]:
    """Fewest explicit assumptions first; equal counts keep the order they were proposed in."""
    indexed = list(enumerate(hypotheses))
    indexed.sort(key=lambda pair: (len(pair[1].assumptions), pair[0]))
    return [h for _, h in indexed]


def check_hypotheses(hypotheses: Sequence[Hypothesis], mode: Mode) -> List[str]:
    failures: List[str] = []
    n = len(hypotheses)
    if not HYPOTHESES_MIN <= n <= HYPOTHESES_MAX:
        failures.append(f"hypotheses: {n}; the method asks for {HYPOTHESES_MIN}–{HYPOTHESES_MAX} (§4)")
    ids = [h.id for h in hypotheses]
    if len(set(ids)) != len(ids):
        failures.append("hypotheses: ids are not unique")
    kinds = {h.kind for h in hypotheses}
    if "minimal_change" not in kinds:
        failures.append("hypotheses: no minimal-change / do-nothing / delay option (§4, H4)")
    if mode == "standard" and "preferred" not in kinds:
        failures.append("hypotheses: no current / preferred option (§4, H1)")
    return failures


def fastest_kill_test(hypotheses: Sequence[Hypothesis]) -> Optional[str]:
    """The preferred option's first kill test, else the simplest hypothesis's."""
    if not hypotheses:
        return None
    preferred = [h for h in hypotheses if h.kind == "preferred"]
    h = preferred[0] if preferred else occam_order(hypotheses)[0]
    return f"{h.id}: {h.kill_tests[0].text} (reject if {h.kill_tests[0].reject_if})"


# =============================================================================
# Stage 4 — blinding
# =============================================================================
def make_blinding(hypothesis_ids: Sequence[str], seed: int) -> Blinding:
    """A deterministic, seeded, recorded permutation of hypothesis labels.

    The input order does not matter: ids are sorted first, so the mapping depends
    on the set of ids and the seed alone.
    """
    ids = sorted(set(hypothesis_ids), key=_id_key)
    labels = [f"X{i + 1}" for i in range(len(ids))]
    random.Random(seed).shuffle(labels)
    return Blinding(seed=seed, mapping=dict(zip(ids, labels)))


def blinded_hypotheses(hypotheses: Sequence[Hypothesis], blinding: Blinding) -> List[dict]:
    """What a scorer sees: label, title, assumptions. Kind and proposer are withheld."""
    view = [
        {"label": blinding.mapping[h.id], "title": h.title, "assumptions": list(h.assumptions)}
        for h in hypotheses
    ]
    return sorted(view, key=lambda item: _id_key(item["label"]))


def blinded_evidence(evidence: Sequence[Evidence]) -> List[dict]:
    """What a scorer sees of the register. Which hypothesis a channel targets is withheld."""
    return [
        {"id": e.id, "statement": e.statement, "reliability": e.reliability,
         "channel": e.channel, "cutoff": e.cutoff.isoformat()}
        for e in evidence
    ]


def unblind(matrix: ScorerMatrix, blinding: Blinding, evidence_ids: Sequence[str]) -> ScorerMatrix:
    """Re-attach true hypothesis ids in code. A matrix that is incomplete or unknown fails."""
    if not matrix.blinded:
        raise ValueError(f"{matrix.scorer_id}: the matrix did not arrive blinded")
    reverse = {label: hid for hid, label in blinding.mapping.items()}
    if set(matrix.cells) != set(evidence_ids):
        raise ValueError(
            f"{matrix.scorer_id}: scored items {sorted(matrix.cells, key=_id_key)} "
            f"are not the register {sorted(evidence_ids, key=_id_key)}"
        )
    cells: Dict[str, Dict[str, Cell]] = {}
    for eid, row in matrix.cells.items():
        if set(row) != set(reverse):
            raise ValueError(f"{matrix.scorer_id}: {eid} must score every hypothesis label exactly once")
        cells[eid] = {reverse[label]: cell for label, cell in row.items()}
    return ScorerMatrix(scorer_id=matrix.scorer_id, family=matrix.family, role=matrix.role,
                        blinded=False, cells=cells)


# =============================================================================
# Stage 4 — ACH eliminates
# =============================================================================
def consensus(
    matrices: Sequence[ScorerMatrix], evidence_ids: Sequence[str], hypothesis_ids: Sequence[str]
) -> Tuple[Dict[str, Dict[str, Cell]], List[CellRef]]:
    """A cell stands when every scorer agrees; any disagreement makes it '?' (to be observed)."""
    cells: Dict[str, Dict[str, Cell]] = {}
    disagreements: List[CellRef] = []
    for eid in evidence_ids:
        cells[eid] = {}
        for hid in hypothesis_ids:
            votes = {m.cells[eid][hid] for m in matrices}
            if len(votes) == 1:
                cells[eid][hid] = votes.pop()
            else:
                cells[eid][hid] = "?"
                disagreements.append(CellRef(evidence=eid, hypothesis=hid))
    return cells, disagreements


def _inconsistency(
    cells: Dict[str, Dict[str, Cell]],
    evidence: Sequence[Evidence],
    hypothesis_ids: Sequence[str],
    skip: Optional[str] = None,
    force_inconsistent: Optional[Tuple[str, str]] = None,
) -> Dict[str, int]:
    """Weighted sum of I cells per hypothesis. C and N never rescue a hypothesis (§5)."""
    totals = {hid: 0 for hid in hypothesis_ids}
    for e in evidence:
        if e.id == skip:
            continue
        weight = RELIABILITY_WEIGHT[e.reliability]
        for hid in hypothesis_ids:
            cell = "I" if force_inconsistent == (e.id, hid) else cells[e.id][hid]
            if cell == "I":
                totals[hid] += weight
    return totals


def _leader(totals: Dict[str, int], ordered: Sequence[Hypothesis]) -> Tuple[str, Optional[str]]:
    """The least-disproved hypothesis; a tie goes to the simpler one (fewer assumptions)."""
    best = min(totals[h.id] for h in ordered)
    tied = [h for h in ordered if totals[h.id] == best]          # still in Occam order
    if len(tied) == 1:
        return tied[0].id, None
    if len(tied[0].assumptions) < len(tied[1].assumptions):
        return tied[0].id, "occam"
    return tied[0].id, "unresolved"


def _diagnosticity(row: Dict[str, Cell], weight: int) -> int:
    """An item is diagnostic when it splits the hypotheses: some I, some not."""
    known = [c for c in row.values() if c != "?"]
    n_inconsistent = sum(1 for c in known if c == "I")
    return weight * min(n_inconsistent, len(known) - n_inconsistent)


def _flip_item(
    cells: Dict[str, Dict[str, Cell]], evidence: Sequence[Evidence],
    ordered: Sequence[Hypothesis], leader: str,
) -> Tuple[Optional[str], Optional[str]]:
    """The one item most likely to change the leader (paper §8.5).

    First choice: an item still '?' on the leader that would change the leader if it
    turned out inconsistent (the most reliable such item first) — that is what to look
    for next. Otherwise: an item whose removal changes the leader (the least reliable
    first — the likeliest to be wrong).
    """
    hids = [h.id for h in ordered]
    candidates = []
    for e in evidence:
        rank = _RELIABILITY_RANK[e.reliability]
        if cells[e.id][leader] == "?":
            totals = _inconsistency(cells, evidence, hids, force_inconsistent=(e.id, leader))
            if _leader(totals, ordered)[0] != leader:
                candidates.append((0, -rank, _id_key(e.id), e.id, "unknown_on_leader"))
                continue
        totals = _inconsistency(cells, evidence, hids, skip=e.id)
        if _leader(totals, ordered)[0] != leader:
            candidates.append((1, rank, _id_key(e.id), e.id, "removal"))
    if not candidates:
        return None, None
    best = min(candidates)
    return best[3], best[4]


def score_ach(
    hypotheses: Sequence[Hypothesis], evidence: Sequence[Evidence], matrices: Sequence[ScorerMatrix]
) -> Diagnostics:
    """Aggregate unblinded matrices into the four diagnostics."""
    if not matrices or not hypotheses or not evidence:
        raise ValueError("scoring needs hypotheses, evidence and at least one unblinded matrix")
    if any(m.blinded for m in matrices):
        raise ValueError("unblind every matrix before scoring")
    ordered = occam_order(hypotheses)
    hids = [h.id for h in ordered]
    eids = [e.id for e in evidence]
    cells, disagreements = consensus(matrices, eids, hids)
    totals = _inconsistency(cells, evidence, hids)
    leader, tie = _leader(totals, ordered)
    by_rank = sorted(ordered, key=lambda h: (totals[h.id], hids.index(h.id)))
    others = [h.id for h in by_rank if h.id != leader]
    runner_up = others[0] if others else None
    most_contradicted = sorted(ordered, key=lambda h: (-totals[h.id], hids.index(h.id)))[0].id
    diag_values = {e.id: _diagnosticity(cells[e.id], RELIABILITY_WEIGHT[e.reliability]) for e in evidence}
    most_diagnostic = [
        eid for eid, value in sorted(diag_values.items(), key=lambda kv: (-kv[1], _id_key(kv[0])))
        if value > 0
    ][:3]
    flip, flip_reason = _flip_item(cells, evidence, ordered, leader)
    scores = [
        HypothesisScore(
            id=h.id, inconsistency=totals[h.id],
            unknown=sum(1 for eid in eids if cells[eid][h.id] == "?"),
            assumptions=len(h.assumptions), occam_rank=i,
        )
        for i, h in enumerate(ordered)
    ]
    return Diagnostics(
        scorers=[m.scorer_id for m in matrices], consensus=cells, disagreements=disagreements,
        scores=scores, most_diagnostic=most_diagnostic, most_contradicted=most_contradicted,
        leader=leader, tie_broken_by=tie, runner_up=runner_up,
        margin=(totals[runner_up] - totals[leader]) if runner_up else 0,
        flip_item=flip, flip_reason=flip_reason,
    )


def check_scorers(matrices: Sequence[ScorerMatrix], staff_family: str) -> List[str]:
    failures: List[str] = []
    if any(not m.blinded for m in matrices):
        failures.append("ach: a matrix arrived unblinded; scorers see blinded labels only")
    if not any(m.role == "staff" for m in matrices):
        failures.append("ach: no staff matrix")
    families = {m.family for m in matrices if m.role == "red_team" and m.family != staff_family}
    if len(families) < RED_TEAM_MIN_FAMILIES:
        failures.append(
            f"ach: red-team matrices from {len(families)} independent model families; "
            f"≥ {RED_TEAM_MIN_FAMILIES} needed, and the staff's family does not count"
        )
    return failures


# =============================================================================
# Stages 3, 5, 6 — the other gates
# =============================================================================
def check_evidence(
    evidence: Sequence[Evidence], hypotheses: Sequence[Hypothesis], reversibility: str,
    leader: Optional[str],
) -> Tuple[List[str], List[str]]:
    """Returns (failures, warnings). The ≥ 3-channel standard binds high stakes (paper §8.3);
    on a reversible decision a thinner register is shown, not blocked (§12: do not wait)."""
    failures: List[str] = []
    warnings: List[str] = []
    if not evidence:
        failures.append("evidence: the register is empty")
        return failures, warnings
    ids = [e.id for e in evidence]
    if len(set(ids)) != len(ids):
        failures.append("evidence: ids are not unique")
    known = {h.id for h in hypotheses}
    for e in evidence:
        if e.designed_to_disconfirm and e.designed_to_disconfirm not in known:
            failures.append(f"evidence: {e.id} targets an unknown hypothesis {e.designed_to_disconfirm}")
    groups = {e.independence_group for e in evidence}
    high_stakes = reversibility in HUMAN_HAND_REQUIRED
    if len(groups) < MIN_INDEPENDENT_CHANNELS:
        if high_stakes:
            failures.append(
                f"evidence: {len(groups)} independent channel(s); an irreversible or severe decision "
                f"needs ≥ {MIN_INDEPENDENT_CHANNELS}"
            )
        else:
            warnings.append(f"evidence: {len(groups)} independent channel(s), fewer than {MIN_INDEPENDENT_CHANNELS}")
    if high_stakes and leader and not any(e.designed_to_disconfirm == leader for e in evidence):
        failures.append(f"evidence: no channel was designed to disconfirm the leader {leader}")
    return failures, warnings


def check_assumptions(assumptions: Sequence[Assumption], mode: Mode) -> List[str]:
    failures: List[str] = []
    low = ASSUMPTIONS_MIN if mode == "standard" else 1
    if not low <= len(assumptions) <= ASSUMPTIONS_MAX:
        failures.append(f"assumptions: {len(assumptions)}; {low}–{ASSUMPTIONS_MAX} needed (§7)")
    ids = [a.id for a in assumptions]
    if len(set(ids)) != len(ids):
        failures.append("assumptions: ids are not unique")
    critical = [a for a in assumptions if a.critical]
    if not CRITICAL_MIN <= len(critical) <= CRITICAL_MAX:
        failures.append("assumptions: mark the 1–2 that would damage the recommendation most if wrong (§7)")
    return failures


def check_red_team(reports: Sequence[RedTeamReport], staff_family: str, known_ids: set) -> List[str]:
    failures: List[str] = []
    families = {r.family for r in reports if r.family != staff_family}
    if len(families) < RED_TEAM_MIN_FAMILIES:
        failures.append(
            f"red team: reports from {len(families)} independent model families; "
            f"≥ {RED_TEAM_MIN_FAMILIES} needed, and the staff's family does not count"
        )
    for r in reports:
        for flag in r.biases:
            unknown = [c for c in flag.cites if c not in known_ids]
            if unknown:
                failures.append(
                    f"red team {r.scorer_id}: the {flag.bias} flag cites unknown ids {unknown} — "
                    "no evidence, no flag (§8)"
                )
    return failures


# =============================================================================
# Stage 7 — confidence and the gate
# =============================================================================
def combine_confidence(
    estimates: Sequence[ProbabilityEstimate], mode: Mode, staff_family: str
) -> Tuple[Optional[int], List[str]]:
    """Standard: the lowest of the staff's estimate, the red-team median and any human estimate.
    Quick: the lowest of the staff's and the human estimates.

    One model family is one red-team voice: its lowest estimate stands for it, and the
    staff's own family never counts as red team.
    """
    staff = [e.percent for e in estimates if e.role == "staff"]
    human = [e.percent for e in estimates if e.role == "human"]
    per_family: Dict[str, int] = {}
    for e in estimates:
        if e.role == "red_team" and e.family != staff_family:
            per_family[e.family] = min(e.percent, per_family.get(e.family, 100))
    red = sorted(per_family.values())
    if mode == "standard":
        failures = []
        if not staff:
            failures.append("confidence: no staff estimate")
        if len(red) < RED_TEAM_MIN_FAMILIES:
            failures.append(f"confidence: {len(red)} independent red-team estimate(s); ≥ {RED_TEAM_MIN_FAMILIES} needed")
        if failures:
            return None, failures
        parts = [min(staff), statistics.median_low(red)] + ([min(human)] if human else [])
        return min(parts), []
    pool = staff + human
    if not pool:
        return None, ["confidence: no estimate"]
    return min(pool), []


def confidence_label(
    diagnostics: Optional[Diagnostics], hypotheses: Sequence[Hypothesis], evidence: Sequence[Evidence]
) -> Tuple[Level, str]:
    """§18 from the matrix's structure — never from the probability."""
    if diagnostics is None:
        return "L", "Quick VIS: no matrix was run, so the confidence rests on judgment, not on diagnostic evidence"
    leader = diagnostics.leader
    cells = diagnostics.consensus
    lead_score = next(s.inconsistency for s in diagnostics.scores if s.id == leader)
    against_each = []
    for h in hypotheses:
        if h.id == leader:
            continue
        groups = {
            e.independence_group for e in evidence
            if cells[e.id][h.id] == "I" and cells[e.id][leader] in ("C", "N")
        }
        against_each.append(len(groups))
    if lead_score == 0 and against_each and min(against_each) >= 2:
        return "H", "every alternative is contradicted by diagnostic items from ≥ 2 independent channels, and nothing contradicts the leader"
    weak = sum(1 for e in evidence if cells[e.id][leader] == "?" or _diagnosticity(cells[e.id], 1) == 0)
    if weak * 2 > len(evidence):
        return "L", f"{weak} of {len(evidence)} items say nothing diagnostic about the leader"
    return "M", "the evidence favours the leader, but important uncertainty remains"


def run_gate(case: CaseFile) -> Tuple[Optional[Diagnostics], GateResult]:
    """Every check of the case in one place. Any failure, a tripped kill indicator or a
    confidence under the bar removes GO from the allowed set."""
    charter = case.charter
    mode, reversibility = charter.mode, charter.reversibility
    high_stakes = reversibility in HUMAN_HAND_REQUIRED
    failures: List[str] = list(check_hypotheses(case.hypotheses, mode))
    warnings: List[str] = []
    diagnostics: Optional[Diagnostics] = None

    if mode == "standard":
        failures += check_scorers(case.matrices, case.staff_family)
        if case.blinding is None:
            failures.append("ach: no recorded blinding")
        elif set(case.blinding.mapping) != {h.id for h in case.hypotheses}:
            failures.append("ach: the blinding does not cover exactly the hypotheses")
        else:
            try:
                unblinded = [unblind(m, case.blinding, [e.id for e in case.evidence]) for m in case.matrices]
                diagnostics = score_ach(case.hypotheses, case.evidence, unblinded)
            except ValueError as exc:
                failures.append(f"ach: {exc}")
        if diagnostics is None:
            failures.append("ach: no matrix could be scored")
        elif diagnostics.tie_broken_by == "unresolved":
            failures.append("ach: the leader is tied with an equally simple hypothesis; gather the item that separates them")
        evidence_failures, evidence_warnings = check_evidence(
            case.evidence, case.hypotheses, reversibility, diagnostics.leader if diagnostics else None
        )
        failures += evidence_failures
        warnings += evidence_warnings
        known_ids = {e.id for e in case.evidence} | {a.id for a in case.assumptions}
        failures += check_red_team(case.red_team, case.staff_family, known_ids)
        if diagnostics and diagnostics.flip_reason == "unknown_on_leader":
            spot = f"the flip item {diagnostics.flip_item} is still '?' on the leader"
            if high_stakes:
                failures.append(f"evidence: {spot} — resolve it before an irreversible commitment")
            else:
                warnings.append(f"evidence: {spot} — observe it first after GO")

    failures += check_assumptions(case.assumptions, mode)
    if any(a.critical and a.confidence == "L" for a in case.assumptions):
        spot = "a critical assumption is held at low confidence"
        if high_stakes:
            failures.append(f"assumptions: {spot} — raise it before an irreversible commitment")
        else:
            warnings.append(f"assumptions: {spot} — its kill indicator is the first thing to watch")

    estimate, confidence_failures = combine_confidence(case.estimates, mode, case.staff_family)
    failures += confidence_failures
    confidence = estimate if estimate is not None else 0

    hand = case.human_hand
    if high_stakes:
        if hand is None:
            failures.append("gate: an irreversible or severe decision needs the decision owner's human hand")
        elif hand.name != charter.roles.decision_owner:
            failures.append(f"gate: the human hand must be the decision owner's ({charter.roles.decision_owner}), not {hand.name}'s")

    bar = BAR[reversibility]
    tripped = [a for a in case.assumptions if a.kill_indicator and a.kill_indicator.tripped]
    label, label_reason = confidence_label(diagnostics, case.hypotheses, case.evidence)

    go_allowed = not failures and not tripped and confidence >= bar
    allowed: List[Recommendation] = ["GO", "LEARN", "STOP"] if go_allowed else ["LEARN", "STOP"]
    if go_allowed:
        recommendation: Recommendation = "GO"
        reason = (f"confidence {confidence} %, {confidence - bar} above the {reversibility} bar of {bar} % "
                  f"(range {FLOOR}–100, more is better), and every gate passed")
    else:
        parts = []
        if failures:
            parts.append(f"{len(failures)} gate failure(s)")
        if tripped:
            parts.append("kill indicator tripped on " + ", ".join(a.id for a in tripped))
        if confidence < bar:
            parts.append(f"confidence {confidence} % < the {reversibility} bar of {bar} %")
        reason = "; ".join(parts)
        stop = any(a.kill_indicator.action == "STOP" for a in tripped)
        recommendation = "STOP" if stop else "LEARN"

    return diagnostics, GateResult(
        mode=mode, reversibility=reversibility, bar=bar, confidence=confidence,
        headroom=confidence - bar, label=label, label_reason=label_reason,
        failures=failures, warnings=warnings, tripped=[a.id for a in tripped], human_hand=hand, allowed=allowed,
        recommendation=recommendation, reason=reason,
        next_observation=(diagnostics.flip_item if diagnostics and diagnostics.flip_item
                          else fastest_kill_test(case.hypotheses)),
    )


def decide(gate: GateResult, charter: Charter, choice: Recommendation, by: str, at: datetime,
           reason: str) -> Decision:
    """Only the decision owner decides. A choice other than the recommendation is recorded as an override."""
    if by != charter.roles.decision_owner:
        raise ValueError("only the decision owner decides; VIS assesses")
    return Decision(choice=choice, by=by, at=at, reason=reason, recommended=gate.recommendation,
                    override=choice != gate.recommendation)


# =============================================================================
# Stage 8 — brief and receipt
# =============================================================================
def assemble_brief(case: CaseFile, diagnostics: Optional[Diagnostics], gate: GateResult,
                   prose: BriefProse, aar_date: date) -> DecisionBrief:
    """Code fills every computed field; the staff's prose fills the rest."""
    charter = case.charter
    titles = {h.id: h.title for h in case.hypotheses}
    if diagnostics is not None:
        lead_score = next(s.inconsistency for s in diagnostics.scores if s.id == diagnostics.leader)
        statuses = [
            HypothesisStatus(
                id=s.id, title=titles[s.id], inconsistency=s.inconsistency,
                status="leader" if s.id == diagnostics.leader
                else ("contradicted" if s.inconsistency > lead_score else "open"),
            )
            for s in diagnostics.scores
        ]
        assessment = (
            f"Based on the available evidence, VIS assesses {diagnostics.leader} "
            f"({titles[diagnostics.leader]}) as the least-disproved hypothesis, at {gate.confidence} % "
            f"({gate.headroom:+d} against the {gate.bar} % bar; confidence {_LABEL_WORD[gate.label]}); "
            f"recommendation: {gate.recommendation}."
        )
    else:
        statuses = [HypothesisStatus(id=h.id, title=h.title, status="open") for h in occam_order(case.hypotheses)]
        assessment = (
            f"Based on the available evidence, VIS assesses {gate.recommendation} at {gate.confidence} % "
            f"({gate.headroom:+d} against the {gate.bar} % bar)."
        )
    if charter.mode == "standard" and not prose.least_disproved_reason.strip():
        raise ValueError("a Standard VIS brief says why the leader is the least disproved (§19.5)")
    return DecisionBrief(
        case_id=charter.case_id, mode=charter.mode, intent=charter.intent.statement,
        question=charter.question.text(),
        key_facts=[s for s in charter.statements if s.label == "fact"],
        hypotheses=statuses, least_disproved=diagnostics.leader if diagnostics else None,
        least_disproved_reason=prose.least_disproved_reason, confidence=gate.confidence, bar=gate.bar,
        confidence_label=gate.label, confidence_reason=gate.label_reason,
        critical_assumptions=[a for a in case.assumptions if a.critical],
        main_risks=prose.main_risks, warnings=list(gate.warnings), recommendation=gate.recommendation,
        recommendation_reason=prose.recommendation_reason, next_action=prose.next_action,
        ooda_trigger=prose.ooda_trigger, aar_date=aar_date, assessment=assessment,
    )


def render_brief(brief: DecisionBrief) -> str:
    """The one-page brief as Markdown (§19 order)."""
    lines = [
        f"# VIS Decision Brief — {brief.case_id} ({brief.mode})", "",
        f"_{brief.assessment}_", "",
        f"1. **Commander's intent:** {brief.intent}",
        f"2. **Decision question:** {brief.question}",
        "3. **Key facts:** " + ("; ".join(f"{s.text} [{s.source_ref}]" for s in brief.key_facts) or "—"),
        "4. **Hypotheses:** " + "; ".join(
            f"{h.id} {h.title} — {h.status}" + (f" ({h.inconsistency})" if h.inconsistency is not None else "")
            for h in brief.hypotheses),
        f"5. **Least disproved:** {brief.least_disproved or '— (Quick VIS)'}"
        + (f" — {brief.least_disproved_reason}" if brief.least_disproved_reason else ""),
        f"6. **Confidence:** {brief.confidence} % ({brief.confidence - brief.bar:+d} against the {brief.bar} % bar; "
        f"range {FLOOR}–100, more is better) · {_LABEL_WORD[brief.confidence_label]} — {brief.confidence_reason}",
        "7. **Critical assumptions:** " + ("; ".join(
            f"{a.id} {a.text} — kill: {a.kill_indicator.observable} {a.kill_indicator.threshold} → {a.kill_indicator.action}"
            for a in brief.critical_assumptions if a.kill_indicator) or "—"),
        "8. **Main risks:** " + "; ".join(brief.main_risks)
        + ("" if not brief.warnings else " · **Warnings:** " + "; ".join(brief.warnings)),
        f"9. **Recommendation:** {brief.recommendation} — {brief.recommendation_reason}",
        f"10. **Next action:** {brief.next_action.action} · owner {brief.next_action.owner} · by "
        f"{brief.next_action.deadline.isoformat()} · success: {brief.next_action.success_metric} · "
        f"kill: {brief.next_action.kill_criterion}",
        f"11. **OODA trigger:** {brief.ooda_trigger}",
        "", f"**After-action review:** {brief.aar_date.isoformat()}",
    ]
    return "\n".join(lines) + "\n"


def _receipt_hash(receipt: DecisionReceipt) -> str:
    return content_hash(receipt.model_dump(mode="json", exclude={"receipt_hash"}))


def issue_receipt(case: CaseFile, diagnostics: Optional[Diagnostics], gate: GateResult,
                  brief: DecisionBrief, created_at: datetime,
                  decision: Optional[Decision] = None) -> DecisionReceipt:
    if decision is not None and decision.recommended != gate.recommendation:
        raise ValueError("the decision was taken on a different gate result")
    hashes = {
        "charter": content_hash(case.charter),
        "hypotheses": content_hash(case.hypotheses),
        "evidence": content_hash(case.evidence),
        "blinding": content_hash(case.blinding),
        "matrices": content_hash(case.matrices),
        "assumptions": content_hash(case.assumptions),
        "red_team": content_hash(case.red_team),
        "estimates": content_hash(case.estimates),
        "human_hand": content_hash(case.human_hand),
        "diagnostics": content_hash(diagnostics),
        "gate": content_hash(gate),
    }
    receipt = DecisionReceipt(
        case_id=case.charter.case_id, created_at=created_at, mode=case.charter.mode,
        reversibility=case.charter.reversibility, object_hashes=hashes,
        lessons_pack_hash=case.lessons_pack_hash, lessons_loaded=list(case.lessons_loaded),
        brief_hash=content_hash(brief), recommendation=gate.recommendation, decision=decision,
    )
    return receipt.model_copy(update={"receipt_hash": _receipt_hash(receipt)})


def verify_receipt(receipt: DecisionReceipt) -> bool:
    return receipt.receipt_hash == _receipt_hash(receipt)


def to_reference(receipt: DecisionReceipt, schema_id: str, schema_version: str) -> dict:
    """The reference object VIS hands across the data boundary (docs/EVIDENCE_SEAM.md).
    Carried, not attested: a reference is not a verification."""
    if not schema_id or not schema_version:
        raise ValueError("schema_id and schema_version are required")
    if not verify_receipt(receipt):
        raise ValueError("the receipt hash does not verify")
    return {
        "schema_id": schema_id,
        "schema_version": schema_version,
        "source_system": "vis",
        "receipt_hash": receipt.receipt_hash,
        "metadata": receipt.model_dump(mode="json"),
    }


# =============================================================================
# Stages 9–10 — after action and the lessons pack
# =============================================================================
def forecast_score(confidence_percent: int, leader_held: bool) -> float:
    """Brier score of the stored confidence against what happened (0 = perfect, 1 = worst)."""
    p = confidence_percent / 100
    return round((p - (1.0 if leader_held else 0.0)) ** 2, 4)


def check_after_action(aar: AfterAction, receipt: DecisionReceipt,
                       assumptions: Sequence[Assumption]) -> List[str]:
    failures: List[str] = []
    if not verify_receipt(receipt):
        failures.append("after action: the receipt does not verify")
    if aar.receipt_hash != receipt.receipt_hash:
        failures.append("after action: written for another receipt")
    for lesson in aar.lessons:
        if lesson.source_receipt_hash != receipt.receipt_hash:
            failures.append(f"lesson {lesson.id}: cites another battle")
        if lesson.status != "PROPOSED":
            failures.append(f"lesson {lesson.id}: a lesson starts PROPOSED; the decision owner accepts it")
    known = {a.id for a in assumptions}
    unknown = [a for a in aar.assumption_failures if a not in known]
    if unknown:
        failures.append(f"after action: unknown assumption ids {unknown}")
    return failures


def accept_lesson(lesson: Lesson, by: str, at: datetime) -> Lesson:
    data = lesson.model_dump()
    data.update(status="ACCEPTED", accepted_by=by, accepted_at=at)
    return Lesson(**data)


def add_to_pack(pack: LessonsPack, lessons: Sequence[Lesson]) -> LessonsPack:
    """A new pack version; only accepted lessons get in (the schema refuses the rest)."""
    return LessonsPack(version=pack.version + 1, lessons=list(pack.lessons) + list(lessons))


def lessons_for(pack: LessonsPack, tags: Sequence[str]) -> List[Lesson]:
    wanted = set(tags)
    chosen = [l for l in pack.lessons if "*" in l.applies_to or wanted & set(l.applies_to)]
    return sorted(chosen, key=lambda l: (STAGE_ORDER.index(l.stage), l.id))


def load_lessons(case: CaseFile, pack: LessonsPack) -> Tuple[CaseFile, List[Lesson]]:
    """Stage 0: attach the pack's version hash and the ids of the lessons that apply.
    Lessons warn and add checks; they never move a bar — that takes the owner's amendment."""
    lessons = lessons_for(pack, case.charter.tags)
    loaded = case.model_copy(update={"lessons_pack_hash": content_hash(pack),
                                     "lessons_loaded": [l.id for l in lessons]})
    return loaded, lessons
