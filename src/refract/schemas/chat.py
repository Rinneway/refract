"""Follow-up chat over a finished insight."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field

from .common import SCHEMA_VERSION, Contract, SourceRef
from .insight import Claim


class ChatIntent(str, Enum):
    CLARIFY = "clarify"  # clarify a statement of the insight
    SLICE = "slice"  # another angle: suppliers, group, another factor or research topic
    SOURCE = "source"  # show the primary source
    OUT_OF_SCOPE = "out_of_scope"  # not about the subject, the event or the economy


class ChatTurn(Contract):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=4000)


class ChatRequest(Contract):
    schema_version: str = SCHEMA_VERSION
    insight_id: str
    question: str = Field(min_length=1, max_length=1000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class ChatAnswer(Contract):
    """The part of the answer the model generates (structured output)."""

    intent: ChatIntent
    answer: str = Field(description="Direct answer, 2–5 sentences")
    claims: list[Claim] = Field(default_factory=list, max_length=6)
    follow_ups: list[str] = Field(default_factory=list, max_length=3)


class ChatResponse(Contract):
    schema_version: str = SCHEMA_VERSION
    insight_id: str
    question: str
    intent: ChatIntent
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    trace_id: str | None = None
    language: str = "ru"
    created_at: datetime
    disclaimer: str = ""
