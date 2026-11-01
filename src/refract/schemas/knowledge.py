"""The institution's research: verified theses and searchable document excerpts.

The “house view” is whatever the operator publishes as its own position — a bank's macro research, a
consultancy's sector notes, an accountant's memo on a new tax rule. The agent may attribute a statement to the
institution only when it cites one of these objects.
"""

from __future__ import annotations

from datetime import date

from pydantic import Field

from .common import SCHEMA_VERSION, Contract


class Figure(Contract):
    name: str
    value: float | str
    unit: str | None = None
    period: str | None = None


def applies_to(countries: list[str], country: str | None) -> bool:
    """A thesis or excerpt tied to other economies does not apply to the subject (a Russian policy-rate forecast
    says nothing about a UK company's loan); one tied to no country applies everywhere."""
    return not countries or not country or country.upper() in {c.upper() for c in countries}


class Thesis(Contract):
    """A position of the institution on a topic, with numbers and the exact quote it rests on."""

    schema_version: str = SCHEMA_VERSION
    thesis_id: str
    topic: str = Field(description="Topic from the vocabulary: policy_rate, inflation, fx, taxes…")
    statement: str = Field(description="The position in one or two sentences")
    figures: list[Figure] = Field(default_factory=list)
    horizon: str | None = Field(default=None, description="E.g. “by the end of 2027”")
    factors: list[str] = Field(default_factory=list, description="Vocabulary factors the thesis touches")
    doc_id: str
    doc_title: str
    doc_date: date | None = None
    page: int | None = Field(default=None, ge=1)
    quote: str = Field(description="Exact quote from the document")
    language: str = "ru"
    countries: list[str] = Field(
        default_factory=list,
        description="ISO 3166-1 alpha-2 codes of the economies the thesis is about (a central bank's rate, a tax "
        "rule, UK exporters → GB); empty — not tied to a country",
    )
    is_stale: bool = Field(default=False, description="The document is too old for current conclusions")


class KnowledgeChunk(Contract):
    """A fragment of a research document for retrieval."""

    chunk_id: str
    doc_id: str
    doc_title: str
    doc_date: date | None = None
    page: int | None = Field(default=None, ge=1)
    text: str
    language: str = "ru"
    countries: list[str] = Field(
        default_factory=list, description="ISO codes of the economies the document is about; empty — any"
    )
    tags: list[str] = Field(default_factory=list)
