"""Agent state, plan, tool calls and the verification report."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from .common import Contract, ReaderRole, SourceRef
from .insight import InsightContent, VerificationIssue
from .knowledge import KnowledgeChunk, Thesis
from .news import NewsEvent
from .profile import Profile
from .trace import Trace


class ToolName(str, Enum):
    """Allow-list of agent tools. All of them are read-only and bound to one subject."""

    GET_PROFILE = "get_profile"
    GET_COUNTERPARTIES = "get_counterparties"
    GET_GROUP = "get_group"
    GET_HOUSE_VIEW = "get_house_view"
    SEARCH_RESEARCH = "search_research"
    SEARCH_NEWS = "search_news"


class PlanStep(Contract):
    step_id: str
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    purpose: str = Field(description="Why this step is needed — goes to the trace")
    done: bool = False


class Plan(Contract):
    rationale: str
    steps: list[PlanStep] = Field(min_length=1)


class ToolCall(Contract):
    call_id: str
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(Contract):
    call_id: str
    tool: ToolName
    ok: bool
    data: Any = None
    error: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)


class VerificationReport(Contract):
    passed: bool
    issues: list[VerificationIssue] = Field(default_factory=list)
    need_more_data: list[str] = Field(default_factory=list, description="What else to search for")


class AgentContext(Contract):
    """Everything the agent gathered about the subject and the event — one context for all steps."""

    profile: Profile | None = None
    event: NewsEvent | None = None
    theses: list[Thesis] = Field(default_factory=list)
    research: list[KnowledgeChunk] = Field(default_factory=list)
    related_events: list[NewsEvent] = Field(default_factory=list)
    sources: dict[str, SourceRef] = Field(default_factory=dict)


class AgentState(Contract):
    run_id: str
    profile_id: str
    event_id: str
    role: ReaderRole | None = None
    language: str = "ru"
    plan: Plan | None = None
    context: AgentContext = Field(default_factory=AgentContext)
    draft: InsightContent | None = None
    verification: VerificationReport | None = None
    iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=1, ge=0)
    trace: Trace
