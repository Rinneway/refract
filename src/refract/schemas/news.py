"""News events: one or more articles about the same thing."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from .common import SCHEMA_VERSION, Confidence, Contract


class EventType(str, Enum):
    MONETARY_POLICY = "monetary_policy"
    INFLATION = "inflation"
    FX = "fx"
    COMMODITIES = "commodities"
    MACRO_STATS = "macro_stats"
    FINANCIAL_MARKETS = "financial_markets"
    REGULATION = "regulation"
    TAX = "tax"
    SANCTIONS = "sanctions"
    TRADE = "trade"
    INDUSTRY = "industry"
    COMPANY = "company"
    LABOR = "labor"
    LOGISTICS = "logistics"
    OTHER = "other"


class NewsArticle(Contract):
    article_id: str
    url: str
    publisher: str
    title: str
    published_at: datetime
    text: str | None = Field(default=None, description="Full text, if you are allowed to store it")
    language: str | None = Field(default=None, description="ru, en…")


class KeyFact(Contract):
    fact_id: str
    text: str
    quote: str | None = Field(default=None, description="Fragment of the source text")
    article_id: str


class NewsEvent(Contract):
    """An event is one or more articles about the same thing (duplicates merged)."""

    schema_version: str = SCHEMA_VERSION
    event_id: str
    title: str
    summary: str
    event_type: EventType = EventType.OTHER
    published_at: datetime = Field(description="Time of the first publication")
    articles: list[NewsArticle] = Field(min_length=1)
    key_facts: list[KeyFact] = Field(default_factory=list)
    industry_codes: list[str] = Field(
        default_factory=list,
        description="Two-digit NACE/OKVED divisions the event names directly; in trade (45–47) the three-digit "
        "group when the goods are known (46.2), since the division alone does not say what is traded",
    )
    industry_sections: list[str] = Field(default_factory=list, description="Affected sections A–U")
    regions: list[str] = Field(default_factory=list)
    countries: list[str] = Field(
        default_factory=list,
        description="ISO codes or names of the countries the news is about, the domestic one included. National "
        "factors (the policy rate, taxes, demand…) link the event only to subjects of these countries; empty — "
        "unknown, no restriction",
    )
    commodities: list[str] = Field(default_factory=list)
    macro_factors: list[str] = Field(
        default_factory=list, description="Up to three factors from the vocabulary, the main one first"
    )
    entities: list[str] = Field(default_factory=list)
    importance: Confidence = Field(default=0.5, description="Strength of the event regardless of the subject")
    language: str = Field(default="ru", description="Language of the articles")
