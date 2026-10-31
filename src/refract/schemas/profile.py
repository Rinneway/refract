"""Profile of the reader's subject: a company, a sole proprietor, a self-employed person or a private individual.

The agent never sees raw data — only this profile. Any data source (a CRM, a bank's client base, a form the user
fills in) is converted into it once, so the agent stays the same whatever the source.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field

from .common import SCHEMA_VERSION, Confidence, Contract, Direction, Origin


class SubjectKind(str, Enum):
    COMPANY = "company"
    SOLE_PROPRIETOR = "sole_proprietor"  # e.g. Russian IP (ИП), UK sole trader
    SELF_EMPLOYED = "self_employed"  # freelancer, Russian NPD (самозанятый)
    INDIVIDUAL = "individual"  # private person: saver, borrower, investor, would-be entrepreneur


class Industry(Contract):
    """Activity code. NACE Rev. 2 and OKVED 2 share sections (A–U) and two-digit divisions."""

    code: str = Field(description="Activity code, e.g. 46.73")
    name: str
    section: str | None = Field(default=None, description="Section letter A–U")
    scheme: Literal["nace", "okved", "other"] = "nace"


class Scale(Contract):
    revenue: float | None = Field(default=None, ge=0, description="Annual revenue in `currency`")
    group_revenue: float | None = Field(default=None, ge=0, description="Revenue of the group of companies")
    currency: str = Field(default="RUB", description="ISO 4217 currency of the amounts")
    segment: Literal["large", "medium", "small", "micro", "household", "unknown"] = "unknown"
    employees: str | None = Field(default=None, description="Headcount bucket, e.g. 51-250")
    age_years: str | None = Field(default=None, description="Age bucket of the business")


class IndustryPosition(Contract):
    peers_in_sample: int = Field(ge=0, description="Subjects with the same activity code in the sample")
    revenue_percentile: float | None = Field(default=None, ge=0, le=100)
    method: str = Field(description="How it was computed — shown to the reader")


class Products(Contract):
    """Financial products the subject holds with the institution. Unknown elsewhere."""

    current_account: bool = False
    loans: bool = False
    mortgage: bool = False
    leasing: bool = False
    deposits: bool = False
    payroll: bool = False
    factoring: bool = False
    guarantees: bool = False
    acquiring: bool = False
    fx_account: bool = False
    investments: bool = False


class ForeignTrade(Contract):
    active: bool = False
    countries: list[str] = Field(default_factory=list, description="ISO 3166-1 alpha-2 codes or names")
    direction: Literal["import", "export", "both", "unknown"] = "unknown"


class Counterparty(Contract):
    """A payment link. Transfers to oneself are excluded before profiling."""

    counterparty_id: str
    role: Literal["supplier", "customer"] = Field(
        description="supplier — the subject pays them; customer — they pay the subject"
    )
    amount: float = Field(ge=0)
    payments_count: int = Field(default=0, ge=0)
    share_of_flow: float = Field(ge=0, le=1, description="Share of incoming or outgoing payment flow")
    industry: Industry | None = Field(
        default=None, description="None — the counterparty's activity is unknown"
    )
    region: str | None = None
    same_group: bool = False


class Affiliation(Contract):
    """A related party: confirmed (same group) or a hypothesis (shared person)."""

    profile_id: str
    kind: Literal["group", "shared_person"]
    origin: Origin
    confidence: Confidence
    evidence: str


class Exposure(Contract):
    """Sensitivity of the subject to an external factor."""

    factor: str = Field(description="Factor from the vocabulary: policy_rate, fx:CNY, sector:46, country:CN…")
    direction: Direction
    strength: Confidence
    via: Literal["own", "counterparty", "group"] = "own"
    reason: str
    evidence_fields: list[str] = Field(default_factory=list, description="Features it is derived from")


class DataQuality(Contract):
    missing: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Profile(Contract):
    schema_version: str = SCHEMA_VERSION
    profile_id: str
    kind: SubjectKind = SubjectKind.COMPANY
    name: str | None = Field(default=None, description="Display name")
    country: str | None = Field(default=None, description="ISO 3166-1 alpha-2 country of residence")
    region: str | None = None
    city: str | None = None
    industry: Industry | None = Field(default=None, description="Main activity; may be empty for individuals")
    tax_regime: str | None = Field(
        default=None, description="Tax regime, e.g. simplified 6%, patent, VAT payer"
    )
    scale: Scale = Field(default_factory=Scale)
    industry_position: IndustryPosition | None = None
    products: Products = Field(default_factory=Products)
    foreign_trade: ForeignTrade = Field(default_factory=ForeignTrade)
    group_id: str | None = None
    group_members: list[str] = Field(default_factory=list)
    counterparties: list[Counterparty] = Field(default_factory=list)
    affiliations: list[Affiliation] = Field(default_factory=list)
    exposures: list[Exposure] = Field(default_factory=list)
    user_provided: dict[str, str] = Field(
        default_factory=dict, description="What the reader said about themselves; treated as untrusted text"
    )
    data_quality: DataQuality = Field(default_factory=DataQuality)
    summary: str = Field(description="Short description for prompts, 2–6 sentences")
