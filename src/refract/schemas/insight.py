"""The personalised insight — the agent's main output.

It answers six questions: what happened, why it matters to this subject, what may follow (risks,
opportunities, scenarios), what the institution thinks, what can be done, and what each conclusion rests on.
Every statement carries its origin and sources; grounding rules are enforced when the object is created.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from ..textutil import trim_sentence
from .common import (
    SCHEMA_VERSION,
    Confidence,
    Contract,
    Direction,
    Horizon,
    ImpactLevel,
    Origin,
    ReaderRole,
    SourceRef,
    SourceType,
)


class Claim(Contract):
    """One statement with its origin, sources and the subject's features it relies on."""

    claim_id: str
    text: str
    origin: Origin
    source_ids: list[str] = Field(default_factory=list)
    features: list[str] = Field(
        default_factory=list,
        description="Feature keys of the subject the statement relies on, e.g. products.loans",
    )
    confidence: Confidence
    caveat: str | None = Field(
        default=None, description="Caveat: an assumption, missing data or disagreement with the house view"
    )


class SubjectClaim(Claim):
    """A “why it matters” statement: it must name the subject's features it relies on."""

    features: list[str] = Field(
        description="Required: one or more keys from the subject's Features list, e.g. products.loans, tax_regime"
    )

    @model_validator(mode="before")
    @classmethod
    def _tolerate_missing_features(cls, data: object) -> object:
        # The field stays required in the JSON schema; a small model that omits it is repaired later.
        if isinstance(data, dict) and data.get("features") is None:
            return {**data, "features": []}
        return data


class QuantEstimate(Contract):
    low: float | None = None
    high: float | None = None
    unit: str = Field(description="%, p.p., RUB, USD m…")
    basis: str = Field(description="How the estimate was obtained — shown to the reader")


class Effect(Claim):
    """A risk or an opportunity for the subject."""

    kind: Literal["risk", "opportunity"]
    impact: ImpactLevel
    horizon: Horizon
    estimate: QuantEstimate | None = None


class Scenario(Contract):
    """A way events may unfold, without a precise probability."""

    name: str
    description: str
    plausibility: Literal["low", "medium", "high"]
    triggers: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class ActionKind(str, Enum):
    ANALYSIS = "analysis"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    CONTACT_MANAGER = "contact_manager"
    REQUEST_CALLBACK = "request_callback"
    APPLY_PRODUCT = "apply_product"


class Action(Contract):
    action_id: str
    text: str
    kind: ActionKind
    rationale: str
    product: str | None = Field(default=None, description="A product — only when an action-policy rule fired")
    claim_ids: list[str] = Field(default_factory=list, description="Statements the action relies on")


_CLAIM_BLOCKS = ("what_happened", "why_it_matters", "risks", "opportunities", "house_view")


class InsightContent(Contract):
    """The part of the insight the model generates (structured output)."""

    headline: str = Field(max_length=160, description="What the event means for this subject, one sentence")
    what_happened: list[Claim] = Field(min_length=1)
    why_it_matters: list[SubjectClaim] = Field(min_length=1)
    risks: list[Effect] = Field(default_factory=list)
    opportunities: list[Effect] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(default_factory=list)
    house_view: list[Claim] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    overall_impact: ImpactLevel
    overall_direction: Direction

    @model_validator(mode="before")
    @classmethod
    def _trim_headline(cls, data: object) -> object:
        # Models often overshoot the length; a headline cut at a clause boundary beats a rejected draft.
        if isinstance(data, dict) and isinstance(data.get("headline"), str) and len(data["headline"]) > 160:
            return {**data, "headline": trim_sentence(data["headline"], 160)}
        return data

    @model_validator(mode="before")
    @classmethod
    def _unique_ids(cls, data: object) -> object:
        """Small models repeat an id (two statements “c1”); the later one gets a suffix (“c1_2”) instead of the
        whole insight failing. An action that cites the repeated id keeps pointing at the first statement."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for blocks, key in ((_CLAIM_BLOCKS, "claim_id"), (("actions",), "action_id")):
            seen: set[str] = set()
            for block in blocks:
                items = out.get(block)
                if not isinstance(items, list):
                    continue
                renamed = []
                for item in items:
                    ident = item.get(key) if isinstance(item, dict) else getattr(item, key, None)
                    ident = ident.strip() if isinstance(ident, str) else ident
                    if isinstance(ident, str) and ident in seen:
                        n = 2
                        while f"{ident}_{n}" in seen:
                            n += 1
                        new = f"{ident}_{n}"
                        item = (
                            {**item, key: new}
                            if isinstance(item, dict)
                            else item.model_copy(update={key: new})
                        )
                        ident = new
                    if isinstance(ident, str):
                        seen.add(ident)
                    renamed.append(item)
                out[block] = renamed
        return out

    def all_claims(self) -> list[Claim]:
        return [*self.what_happened, *self.why_it_matters, *self.risks, *self.opportunities, *self.house_view]


class Insight(InsightContent):
    schema_version: str = SCHEMA_VERSION
    insight_id: str
    profile_id: str
    event_id: str
    role: ReaderRole | None = None
    language: str = "ru"
    created_at: datetime
    sources: list[SourceRef] = Field(min_length=1)
    trace_id: str | None = None
    models: list[str] = Field(default_factory=list)
    disclaimer: str = ""
    validated: bool = Field(default=True, description="False only in ablation runs without rule checks")

    @model_validator(mode="after")
    def _check_grounding(self) -> Insight:
        errors = violations(self)
        if errors:
            raise ValueError("; ".join(i.message for i in errors))
        return self


# ---------------------------------------------------------------------------------------------------- checks


class IssueCode(str, Enum):
    UNKNOWN_SOURCE = "unknown_source"
    MISSING_SOURCE = "missing_source"
    NUMBER_NOT_IN_SOURCES = "number_not_in_sources"
    HOUSE_VIEW_WITHOUT_RESEARCH = "house_view_without_research"
    NO_SUBJECT_LINK = "no_subject_link"
    CONTRADICTION = "contradiction"
    RETELLING = "retelling"
    SHALLOW = "shallow"
    INSUFFICIENT_DATA = "insufficient_data"
    DUPLICATE_ID = "duplicate_id"
    WRONG_LANGUAGE = "wrong_language"
    SCHEMA_ERROR = "schema_error"
    OFF_TOPIC = "off_topic"  # a house view about something else than the event


class VerificationIssue(Contract):
    code: IssueCode
    severity: Literal["error", "warning"]
    claim_id: str | None = None
    message: str
    fix_hint: str | None = None


_MESSAGES: dict[str, dict[str, str]] = {
    "dup_sources": {"ru": "повторяются source_id в sources", "en": "duplicate source_id in sources"},
    "dup_claims": {"ru": "повторяются claim_id", "en": "duplicate claim_id"},
    "unknown": {"ru": "{cid}: нет источников {ids}", "en": "{cid}: unknown sources {ids}"},
    "fact": {"ru": "{cid}: факт без источника", "en": "{cid}: fact without a source"},
    "house": {
        "ru": "{cid}: позиция института без ссылки на его аналитику",
        "en": "{cid}: house view without a research source",
    },
    "hyp": {"ru": "{cid}: гипотеза без подтверждений", "en": "{cid}: hypothesis without any support"},
    "features": {
        "ru": "{cid}: «почему важно» не опирается на признаки субъекта",
        "en": "{cid}: “why it matters” does not rely on any subject feature",
    },
    "block": {
        "ru": "{cid}: в блоке позиции института утверждение с происхождением {origin}",
        "en": "{cid}: house_view block contains a statement of origin {origin}",
    },
    "scenario": {
        "ru": "сценарий «{name}»: нет источников {ids}",
        "en": "scenario “{name}”: unknown sources {ids}",
    },
    "dup_actions": {"ru": "повторяются action_id", "en": "duplicate action_id"},
    "action": {
        "ru": "{aid}: ссылается на несуществующие выводы {ids}",
        "en": "{aid}: refers to missing statements {ids}",
    },
}


def violations(ins: Insight) -> list[VerificationIssue]:
    """Hard grounding rules. An empty list means the insight is well-formed."""
    lang = "en" if (ins.language or "").startswith("en") else "ru"

    def issue(code: IssueCode, key: str, claim_id: str | None = None, **kw: object) -> VerificationIssue:
        return VerificationIssue(
            code=code, severity="error", claim_id=claim_id, message=_MESSAGES[key][lang].format(**kw)
        )

    out: list[VerificationIssue] = []
    by_id = {s.source_id: s for s in ins.sources}
    if len(by_id) != len(ins.sources):
        out.append(issue(IssueCode.DUPLICATE_ID, "dup_sources"))

    claims = ins.all_claims()
    claim_ids = [c.claim_id for c in claims]
    if len(set(claim_ids)) != len(claim_ids):
        out.append(issue(IssueCode.DUPLICATE_ID, "dup_claims"))

    for c in claims:
        unknown = [s for s in c.source_ids if s not in by_id]
        if unknown:
            out.append(issue(IssueCode.UNKNOWN_SOURCE, "unknown", c.claim_id, cid=c.claim_id, ids=unknown))
        if c.origin is Origin.FACT and not c.source_ids:
            out.append(issue(IssueCode.MISSING_SOURCE, "fact", c.claim_id, cid=c.claim_id))
        if c.origin is Origin.HOUSE_VIEW and not any(
            by_id[s].type is SourceType.RESEARCH for s in c.source_ids if s in by_id
        ):
            out.append(issue(IssueCode.HOUSE_VIEW_WITHOUT_RESEARCH, "house", c.claim_id, cid=c.claim_id))
        if c.origin is Origin.HYPOTHESIS and not (c.source_ids or c.features):
            out.append(issue(IssueCode.MISSING_SOURCE, "hyp", c.claim_id, cid=c.claim_id))

    for c in ins.why_it_matters:
        if not c.features:
            out.append(issue(IssueCode.NO_SUBJECT_LINK, "features", c.claim_id, cid=c.claim_id))
    for c in ins.house_view:
        if c.origin is not Origin.HOUSE_VIEW:
            out.append(
                issue(
                    IssueCode.HOUSE_VIEW_WITHOUT_RESEARCH,
                    "block",
                    c.claim_id,
                    cid=c.claim_id,
                    origin=c.origin.value,
                )
            )

    for sc in ins.scenarios:
        unknown = [s for s in sc.source_ids if s not in by_id]
        if unknown:
            out.append(issue(IssueCode.UNKNOWN_SOURCE, "scenario", name=sc.name, ids=unknown))

    action_ids = [a.action_id for a in ins.actions]
    if len(set(action_ids)) != len(action_ids):
        out.append(issue(IssueCode.DUPLICATE_ID, "dup_actions"))
    known = set(claim_ids)
    for a in ins.actions:
        missing = [cid for cid in a.claim_ids if cid not in known]
        if missing:
            out.append(issue(IssueCode.UNKNOWN_SOURCE, "action", a.action_id, aid=a.action_id, ids=missing))
    return out
