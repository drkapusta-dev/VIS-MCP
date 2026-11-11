"""VIS-MCP schemas — single source of truth.

v0: what the multi-model fan-out needs (ModelConfig, AdvisoryResponse, AskAllResult)
plus the canonical-hashing seam.

v1: the typed objects of the decision pipeline. Each stage of the VIS method
produces one of these; the gates in ``vis_core.py`` read them. A malformed
object never gets built: validation fails closed, and unknown fields are
rejected (``extra="forbid"``), so a model answer with stray fields is refused
rather than silently trimmed.

Canonical JSON is byte-identical to the evidence backend's rule (sorted keys,
compact separators, ensure_ascii=False, null NOT dropped).
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Provider = Literal["openrouter", "xai", "anthropic"]


# =============================================================================
# v0 — multi-model fan-out
# =============================================================================
class ModelConfig(BaseModel):
    id: str                       # short handle, e.g. "opus"
    name: str                     # human name
    provider: Provider
    model_id: str                 # provider catalog slug (CONFIRM against live catalog)
    temperature: Optional[float] = 0.3
    max_tokens: int = 4096
    timeout: int = 120
    # "native" | "tool_use" | "prompt_parse"
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


# =============================================================================
# canonical hashing seam (matches the evidence backend byte-for-byte)
# =============================================================================
def _to_jsonable(data: Any) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    if isinstance(data, (list, tuple)):
        return [_to_jsonable(v) for v in data]
    if isinstance(data, dict):
        return {k: _to_jsonable(v) for k, v in data.items()}
    return data


def canonical_json(data: Any) -> str:
    return json.dumps(
        _to_jsonable(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def content_hash(data: Any) -> str:
    """sha256 of the canonical JSON of any schema object (or list / dict of them)."""
    return sha256_hex(canonical_json(data))


# =============================================================================
# v1 — vocabulary
# =============================================================================
Level = Literal["H", "M", "L"]
Cell = Literal["C", "I", "N", "?"]          # Consistent · Inconsistent · Not diagnostic · Unknown
Reversibility = Literal["reversible", "irreversible", "severe"]
Recommendation = Literal["GO", "LEARN", "STOP"]
StatementLabel = Literal["fact", "assumption", "interpretation"]
HypothesisKind = Literal["preferred", "alternative", "minimal_change", "aggressive", "wildcard"]
Role = Literal["staff", "red_team", "human"]
Mode = Literal["quick", "standard"]
Stage = Literal[
    "load", "frame", "occam", "evidence", "ach", "assumptions",
    "red_team", "gate", "brief", "after_action", "lessons",
]
Bias = Literal[
    "confirmation", "availability", "overconfidence", "anchoring",
    "mirror_imaging", "sunk_cost", "status_quo",
]

# Words a single probe variable may not contain: a probe tests ONE variable.
# A guard against the obvious ("price and channel"), not a proof.
_MULTI_VARIABLE_MARKERS = (",", ";", "+", "&", " and ", " or ", "/")

# Wording the method forbids in anything VIS writes: "Never imply 'AI decided'."
_FORBIDDEN_WORDING = ("ai decided", "vis decided", "the model decided", "the ai has decided")


def sentence_count(text: str) -> int:
    """Count sentences by their terminators ('.', '!', '?') followed by a space or the end.

    A full stop between two digits (``70.5``) is not a terminator.
    """
    count = 0
    stripped = text.strip()
    for i, ch in enumerate(stripped):
        if ch not in ".!?":
            continue
        nxt = stripped[i + 1] if i + 1 < len(stripped) else ""
        prv = stripped[i - 1] if i > 0 else ""
        if ch == "." and prv.isdigit() and nxt.isdigit():
            continue
        if nxt == "" or nxt.isspace():
            count += 1
    if stripped and stripped[-1] not in ".!?":
        count += 1                                  # a last sentence without a terminator
    return count


def _numbered_id(value: str, prefix: str) -> str:
    rest = value[len(prefix):]
    if not value.startswith(prefix) or not rest.isdigit():
        raise ValueError(f"id {value!r} must be {prefix}<number>, e.g. {prefix}1")
    return value


def _single_variable(value: str) -> str:
    low = f" {value.lower()} "
    for marker in _MULTI_VARIABLE_MARKERS:
        if marker in low:
            raise ValueError(f"one variable per probe: {value!r} looks like more than one")
    return value


def _not_local_path(value: Optional[str]) -> Optional[str]:
    """Evidence seam rule: a ``*_ref`` field is never an absolute local path."""
    if value is None:
        return value
    v = value.strip()
    windows_drive = len(v) > 2 and v[0].isalpha() and v[1] == ":" and v[2] in "\\/"
    if v.startswith(("/", "~", "file:", "\\")) or windows_drive:
        raise ValueError(f"a *_ref must not be an absolute local path: {value!r}")
    return value


def _no_forbidden_wording(value: str) -> str:
    low = value.lower()
    for phrase in _FORBIDDEN_WORDING:
        if phrase in low:
            raise ValueError(f"never imply that AI decided: {phrase!r} in {value!r}")
    return value


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Stage 1 — Frame (§2 intent, §3 decision question, §14 / paper §7.2 roles)
# =============================================================================
class Statement(Strict):
    text: str = Field(min_length=1)
    label: StatementLabel
    source_ref: Optional[str] = None

    @field_validator("source_ref")
    @classmethod
    def _ref(cls, v: Optional[str]) -> Optional[str]:
        return _not_local_path(v)

    @model_validator(mode="after")
    def _fact_needs_source(self) -> "Statement":
        if self.label == "fact" and not (self.source_ref or "").strip():
            raise ValueError("a fact needs a source_ref — never present an assumption as a fact (§3)")
        return self


class Intent(Strict):
    """Commander's intent (§2): why, what, by when, constraints; purpose + end state in ≤ 2 sentences."""
    why: str = Field(min_length=1)
    what: str = Field(min_length=1)
    by_when: str = Field(min_length=1)
    constraints: List[str] = Field(min_length=1)
    statement: str = Field(min_length=1)

    @field_validator("statement")
    @classmethod
    def _two_sentences(cls, v: str) -> str:
        if sentence_count(v) > 2:
            raise ValueError("commander's intent is at most 1–2 sentences (§2)")
        return v


class DecisionQuestion(Strict):
    """§3: Should [DECISION MAKER] do [ACTION] by [DATE / GATE] given [CONSTRAINTS] to achieve [OUTCOME]?"""
    decision_maker: str = Field(min_length=1)
    action: str = Field(min_length=1)
    by_when: str = Field(min_length=1)
    constraints: List[str] = Field(min_length=1)
    outcome: str = Field(min_length=1)

    def text(self) -> str:
        return (
            f"Should {self.decision_maker} {self.action} by {self.by_when} "
            f"given {', '.join(self.constraints)} in order to achieve {self.outcome}?"
        )


class Roles(Strict):
    """D owns the outcome; R owns the integrity of the process; C give evidence; I are told."""
    decision_owner: str = Field(min_length=1)
    recommender: str = Field(min_length=1)
    consulted: List[str] = Field(default_factory=list)
    informed: List[str] = Field(default_factory=list)


class Charter(Strict):
    case_id: str = Field(min_length=1)
    mode: Mode
    intent: Intent
    question: DecisionQuestion
    roles: Roles
    deadline: date
    reversibility: Reversibility
    reversibility_set_by: str = Field(min_length=1)
    statements: List[Statement] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)       # domain tags; the lessons pack is filtered by them

    @model_validator(mode="after")
    def _rules(self) -> "Charter":
        if self.reversibility_set_by != self.roles.decision_owner:
            raise ValueError("the reversibility class is set by the decision owner, never by a model")
        if self.mode == "quick" and self.reversibility != "reversible":
            raise ValueError("Quick VIS is for reversible decisions; irreversible or severe ones need Standard VIS")
        return self


# =============================================================================
# Stage 2 — Occam (§4, §10, amendment 2: simplest first, one variable per probe)
# =============================================================================
class KillTest(Strict):
    text: str = Field(min_length=1)        # the fastest inexpensive test that could falsify the hypothesis
    variable: str = Field(min_length=1)    # the one variable it probes
    reject_if: str = Field(min_length=1)   # the threshold that rejects

    @field_validator("variable")
    @classmethod
    def _one(cls, v: str) -> str:
        return _single_variable(v)


class Hypothesis(Strict):
    id: str
    title: str = Field(min_length=1)
    kind: HypothesisKind
    assumptions: List[str] = Field(default_factory=list)   # explicit; the Occam order counts them
    probe_variable: str = Field(min_length=1)
    kill_tests: List[KillTest] = Field(min_length=1)
    proposed_by: str = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _numbered_id(v, "H")

    @field_validator("probe_variable")
    @classmethod
    def _one(cls, v: str) -> str:
        return _single_variable(v)


# =============================================================================
# Stage 3 — Evidence (§6; paper §7.3 provenance rule, §8.3 independence)
# =============================================================================
class Evidence(Strict):
    id: str
    statement: str = Field(min_length=1)
    source_ref: Optional[str] = None          # traceable source record
    observer: Optional[str] = None            # tacit evidence: who observed …
    observed_on: Optional[date] = None        # … when …
    context: Optional[str] = None             # … in what context
    cutoff: date                              # freshness
    reliability: Level
    channel: str = Field(min_length=1)
    independence_group: str = Field(min_length=1)
    designed_to_disconfirm: Optional[str] = None   # hypothesis id this channel was set up to disprove

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _numbered_id(v, "E")

    @field_validator("source_ref")
    @classmethod
    def _ref(cls, v: Optional[str]) -> Optional[str]:
        return _not_local_path(v)

    @model_validator(mode="after")
    def _provenance(self) -> "Evidence":
        has_source = bool((self.source_ref or "").strip())
        has_attribution = bool(self.observer and self.observed_on and self.context)
        if not (has_source or has_attribution):
            raise ValueError(
                "no source, no evidence: give a source_ref, or for tacit evidence "
                "observer + observed_on + context (provenance rule)"
            )
        return self


# =============================================================================
# Stage 4 — ACH, blinded (§5; amendment 2: ACH eliminates)
# =============================================================================
class Blinding(Strict):
    seed: int
    mapping: Dict[str, str]                   # true hypothesis id -> blinded label

    @model_validator(mode="after")
    def _bijective(self) -> "Blinding":
        if len(set(self.mapping.values())) != len(self.mapping):
            raise ValueError("blinded labels must be unique")
        return self


class ScorerMatrix(Strict):
    scorer_id: str = Field(min_length=1)
    family: str = Field(min_length=1)         # model family; one family = one voice
    role: Role
    blinded: bool = True                      # as received from the scorer: keyed by blinded labels
    cells: Dict[str, Dict[str, Cell]]         # evidence id -> hypothesis label -> cell


class CellRef(Strict):
    evidence: str
    hypothesis: str


class HypothesisScore(Strict):
    id: str
    inconsistency: int                        # weighted sum of I cells (C and N never rescue)
    unknown: int                              # number of '?' cells
    assumptions: int
    occam_rank: int                           # 0 = simplest


class Diagnostics(Strict):
    scorers: List[str]
    consensus: Dict[str, Dict[str, Cell]]     # by true ids, after unblinding
    disagreements: List[CellRef]
    scores: List[HypothesisScore]             # in Occam order
    most_diagnostic: List[str]
    most_contradicted: str
    leader: str                               # the least-disproved hypothesis
    tie_broken_by: Optional[Literal["occam", "unresolved"]] = None
    runner_up: Optional[str] = None
    margin: int = 0
    flip_item: Optional[str] = None           # the one evidence item most likely to change the leader
    flip_reason: Optional[Literal["unknown_on_leader", "removal"]] = None


# =============================================================================
# Stage 5 — Assumptions and kill indicators (§7; paper §9.1–9.2)
# =============================================================================
class KillIndicator(Strict):
    observable: str = Field(min_length=1)
    threshold: str = Field(min_length=1)
    cadence: str = Field(min_length=1)
    action: Recommendation                    # the pre-set action when it trips
    tripped: bool = False


class Assumption(Strict):
    id: str
    text: str = Field(min_length=1)
    importance: Level
    confidence: Level
    owner: str = Field(min_length=1)
    critical: bool = False                    # one of the 1–2 that would damage the recommendation most
    kill_indicator: Optional[KillIndicator] = None

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _numbered_id(v, "A")

    @model_validator(mode="after")
    def _critical_needs_kill(self) -> "Assumption":
        if self.critical and self.kill_indicator is None:
            raise ValueError("every critical assumption needs a kill indicator (§7)")
        return self


# =============================================================================
# Stage 6 — Red team (§8 bias check, §9 premortem)
# =============================================================================
class FailureScenario(Strict):
    text: str = Field(min_length=1)
    watch_signal: str = Field(min_length=1)   # the evidence we would see if it were developing


class BiasFlag(Strict):
    bias: Bias
    note: str = Field(min_length=1)           # "Potential bias risk: …"
    cites: List[str] = Field(min_length=1)    # evidence / assumption ids — no evidence, no flag (§8)


class RedTeamReport(Strict):
    scorer_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    premortem: str = Field(min_length=1)
    strongest_argument: str = Field(min_length=1)
    failure_scenarios: List[FailureScenario] = Field(min_length=2)     # §9: at least 2; more is welcome
    mitigations: List[str] = Field(min_length=1)
    biases: List[BiasFlag] = Field(default_factory=list)


# =============================================================================
# Stage 7 — Gate (§12, §18, amendment 1)
# =============================================================================
class ProbabilityEstimate(Strict):
    by: str = Field(min_length=1)
    family: str = Field(min_length=1)
    role: Role
    percent: int = Field(ge=0, le=100)        # that the leading hypothesis holds


class HumanHand(Strict):
    name: str = Field(min_length=1)
    at: datetime


class GateResult(Strict):
    mode: Mode
    reversibility: Reversibility
    bar: int
    confidence: int                           # the estimate itself, never bent by code (range 70–100 to decide)
    headroom: int                             # confidence − bar; more is better
    label: Level
    label_reason: str
    failures: List[str] = Field(default_factory=list)    # each one takes GO away
    warnings: List[str] = Field(default_factory=list)    # shown in the brief; GO stays
    tripped: List[str] = Field(default_factory=list)
    human_hand: Optional[HumanHand] = None
    allowed: List[Recommendation]
    recommendation: Recommendation
    reason: str
    next_observation: Optional[str] = None


class Decision(Strict):
    choice: Recommendation
    by: str = Field(min_length=1)
    at: datetime
    reason: str = Field(min_length=1)
    recommended: Recommendation
    override: bool


# =============================================================================
# Stage 8 — Brief (§19) and receipt
# =============================================================================
class NextAction(Strict):
    action: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    deadline: date
    success_metric: str = Field(min_length=1)
    kill_criterion: str = Field(min_length=1)


class BriefProse(Strict):
    """What the staff writes. Never a number, never the leader, never the recommendation."""
    least_disproved_reason: str = ""
    main_risks: List[str] = Field(min_length=1)
    recommendation_reason: str = Field(min_length=1)
    next_action: NextAction
    ooda_trigger: str = Field(min_length=1)   # "reassess when: …"

    @model_validator(mode="after")
    def _wording(self) -> "BriefProse":
        for text in [self.least_disproved_reason, self.recommendation_reason, self.ooda_trigger,
                     self.next_action.action, *self.main_risks]:
            _no_forbidden_wording(text)
        return self


class HypothesisStatus(Strict):
    id: str
    title: str
    inconsistency: Optional[int] = None
    status: Literal["leader", "contradicted", "open"]


class DecisionBrief(Strict):
    case_id: str
    mode: Mode
    intent: str                               # (1)
    question: str                             # (2)
    key_facts: List[Statement]                # (3)
    hypotheses: List[HypothesisStatus]        # (4)
    least_disproved: Optional[str] = None     # (5) — Standard VIS
    least_disproved_reason: str = ""
    confidence: int                           # (6)
    bar: int
    confidence_label: Level
    confidence_reason: str
    critical_assumptions: List[Assumption]    # (7)
    main_risks: List[str]                     # (8)
    warnings: List[str] = Field(default_factory=list)
    recommendation: Recommendation            # (9)
    recommendation_reason: str
    next_action: NextAction                   # (10)
    ooda_trigger: str                         # (11)
    aar_date: date                            # no brief without its after-action date
    assessment: str

    @model_validator(mode="after")
    def _standard_has_leader(self) -> "DecisionBrief":
        if self.mode == "standard" and not self.least_disproved:
            raise ValueError("a Standard VIS brief names the least-disproved hypothesis (§19.5)")
        if self.mode == "standard" and not self.critical_assumptions:
            raise ValueError("a Standard VIS brief lists the critical assumptions (§19.7)")
        return self


class DecisionReceipt(Strict):
    schema_version: Literal["vis.decision_receipt.v1"] = "vis.decision_receipt.v1"
    case_id: str
    created_at: datetime
    mode: Mode
    reversibility: Reversibility
    object_hashes: Dict[str, str]
    lessons_pack_hash: Optional[str] = None
    lessons_loaded: List[str] = Field(default_factory=list)
    brief_hash: str
    recommendation: Recommendation
    decision: Optional[Decision] = None
    receipt_hash: str = ""                    # filled by code over everything above


# =============================================================================
# Stages 9–10 — After action and the lessons pack (amendment 4)
# =============================================================================
class Lesson(Strict):
    id: str = Field(min_length=1)
    source_receipt_hash: str = Field(min_length=64, max_length=64)
    stage: Stage
    applies_to: List[str] = Field(min_length=1)       # domain tags; "*" = every battle
    kind: Literal["check", "warning", "threshold_proposal"]
    text: str = Field(min_length=1)
    status: Literal["PROPOSED", "ACCEPTED"] = "PROPOSED"
    accepted_by: Optional[str] = None
    accepted_at: Optional[datetime] = None

    @model_validator(mode="after")
    def _acceptance(self) -> "Lesson":
        if self.status == "ACCEPTED" and not (self.accepted_by and self.accepted_at):
            raise ValueError("an accepted lesson names who accepted it and when")
        return self


class LessonsPack(Strict):
    version: int = Field(default=0, ge=0)
    lessons: List[Lesson] = Field(default_factory=list)

    @model_validator(mode="after")
    def _only_accepted(self) -> "LessonsPack":
        ids = [lesson.id for lesson in self.lessons]
        if len(set(ids)) != len(ids):
            raise ValueError("lesson ids in a pack are unique")
        if any(lesson.status != "ACCEPTED" for lesson in self.lessons):
            raise ValueError("only accepted lessons enter the pack")
        return self


class AfterAction(Strict):
    receipt_hash: str = Field(min_length=64, max_length=64)
    held_on: date
    planned: str = Field(min_length=1)
    happened: str = Field(min_length=1)
    learned: str = Field(min_length=1)
    next_step: str = Field(min_length=1)
    leader_held: bool                          # did the least-disproved hypothesis hold?
    assumption_failures: List[str] = Field(default_factory=list)
    decision_velocity_days: Optional[int] = Field(default=None, ge=0)
    execution_friction: str = ""
    kpi_deltas: Dict[str, float] = Field(default_factory=dict)
    failed_stage: Optional[Stage] = None
    lessons: List[Lesson] = Field(default_factory=list)


# =============================================================================
# The case file — everything one battle produces, in one typed object
# =============================================================================
class CaseFile(Strict):
    charter: Charter
    staff_family: str = Field(min_length=1)
    lessons_pack_hash: Optional[str] = None
    lessons_loaded: List[str] = Field(default_factory=list)
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    blinding: Optional[Blinding] = None
    matrices: List[ScorerMatrix] = Field(default_factory=list)   # as received: blinded
    assumptions: List[Assumption] = Field(default_factory=list)
    red_team: List[RedTeamReport] = Field(default_factory=list)
    estimates: List[ProbabilityEstimate] = Field(default_factory=list)
    human_hand: Optional[HumanHand] = None
