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
    tags: list[str] = Field(default_factory=list)
