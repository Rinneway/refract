"""Shared types for all contracts."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"

Confidence = Annotated[float, Field(ge=0.0, le=1.0, description="Confidence from 0 to 1")]


class Contract(BaseModel):
    """Base of every contract: unknown fields are rejected, strings are stripped."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, use_enum_values=False)


class Origin(str, Enum):
    """Where a statement comes from. The reader always sees it next to the statement."""

    FACT = "fact"  # stated in a source: the news, the subject's data
    HOUSE_VIEW = "house_view"  # the institution's published position (research notes, forecasts)
    AI = "ai"  # the model's inference from the sources
    HYPOTHESIS = "hypothesis"  # an assumption the data cannot confirm


class SourceType(str, Enum):
    NEWS = "news"
    RESEARCH = "research"  # the institution's research: theses and document excerpts
    PROFILE = "profile"  # data about the subject
    PAYMENTS = "payments"  # payment links with counterparties
    PUBLIC_STATS = "public_stats"
    USER_INPUT = "user_input"


class ImpactLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


IMPACT_LEVEL_DEFINITIONS: dict[str, dict[ImpactLevel, str]] = {
    "ru": {
        ImpactLevel.LOW: "Информационно: планы менять не нужно, полезно учитывать в фоне.",
        ImpactLevel.MEDIUM: (
            "Может потребовать корректировки отдельных решений — цен, закупок, финансирования, бюджета — "
            "в горизонте до квартала."
        ),
        ImpactLevel.HIGH: (
            "Существенно влияет на выручку, затраты, ликвидность или операции; нужно решение в ближайшие недели."
        ),
    },
    "en": {
        ImpactLevel.LOW: "Informational: no plans need to change; worth keeping in mind.",
        ImpactLevel.MEDIUM: (
            "May require adjusting specific decisions — prices, purchasing, financing, budget — within a quarter."
        ),
        ImpactLevel.HIGH: (
            "Materially affects revenue, costs, liquidity or operations; a decision is needed within weeks."
        ),
    },
}


class Horizon(str, Enum):
    IMMEDIATE = "immediate"  # up to 1 month
    SHORT = "short"  # 1–3 months
    MEDIUM = "medium"  # 3–12 months
    LONG = "long"  # more than a year


class Direction(str, Enum):
    NEGATIVE = "negative"
    POSITIVE = "positive"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class ReaderRole(str, Enum):
    """Who reads the insight. The role sets depth, tone and which blocks are expanded."""

    OWNER = "owner"
    CEO = "ceo"
    CFO = "cfo"
    CHIEF_ACCOUNTANT = "chief_accountant"
    TREASURER = "treasurer"
    ENTREPRENEUR = "entrepreneur"  # sole proprietor or self-employed person running their own business
    INDIVIDUAL = "individual"  # a private person: saver, borrower, investor, future entrepreneur
    OTHER = "other"


class SourceRef(Contract):
    """A source that statements cite: a news article, a research page, the subject's data."""

    source_id: str
    type: SourceType
    title: str
    url: str | None = None
    publisher: str | None = None
    published_at: datetime | None = None
    page: int | None = Field(default=None, ge=1, description="Page of a research document")
    excerpt: str | None = Field(default=None, max_length=1000, description="Quote that supports a statement")
