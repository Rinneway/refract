"""A personal news feed: stored events ranked for one subject by code (`agent/feed.py`)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from .common import SCHEMA_VERSION, Contract, ImpactLevel

Relevance = Literal["direct", "background", "none"]


class FeedItem(Contract):
    event_id: str
    title: str
    published_at: datetime
    language: str
    relevance: Relevance = Field(
        description="direct — at least a moderate link to the subject; background — only weak links; none"
    )
    strength: int = Field(
        ge=0, le=3, description="Strength of the strongest link: 3 strong, 2 moderate, 1 weak"
    )
    impact_ceiling: ImpactLevel = Field(description="The highest impact level the agent may give this event")
    reason: str = Field(description="Why: the strongest link, as the agent sees it")
    links: list[str] = Field(default_factory=list, description="Every link between the event and the subject")
    importance: float = Field(ge=0, le=1, description="Strength of the event regardless of the subject")
    tagged: bool = Field(
        description="False: the event has no factors, so only activity and country links count"
    )


class Feed(Contract):
    schema_version: str = SCHEMA_VERSION
    profile_id: str
    language: str
    created_at: datetime
    items: list[FeedItem] = Field(
        description="Strongest links first; within a level by ceiling, links, recency"
    )
    considered: int = Field(description="Events that passed the filters, before `top`")
    untagged: list[str] = Field(default_factory=list, description="Events the model could not tag")


class BriefItem(Contract):
    event_id: str
    title: str
    relevance: Relevance
    insight_id: str | None = Field(
        default=None, description="The full insight; None if the agent produced none"
    )
    headline: str | None = None
    overall_impact: ImpactLevel | None = None
    overall_direction: str | None = None
    why: str | None = Field(default=None, description="The first checked statement of “why it matters”")
    action: str | None = Field(default=None, description="The first action")
    error: str | None = Field(default=None, description="Why there is no insight")


class Brief(Contract):
    """The strongest news of the subject's feed, each explained by the agent and cut down to one paragraph by code:
    no extra model call, so the brief says nothing the insights did not."""

    schema_version: str = SCHEMA_VERSION
    profile_id: str
    language: str
    created_at: datetime
    items: list[BriefItem] = Field(description="Highest impact first")
    considered: int = Field(description="Direct news in the feed before `top`")
