"""Trace of an agent run: every tool call and model call with timing, tokens, cost and verdict."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field

from .common import SCHEMA_VERSION, Contract, ReaderRole


class StepKind(str, Enum):
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    DRAFT = "draft"
    VERIFY = "verify"
    REVISE = "revise"
    FINALIZE = "finalize"
    ERROR = "error"


class TraceStep(Contract):
    index: int = Field(ge=0)
    kind: StepKind
    name: str = Field(description="Tool or step name, e.g. search_research")
    started_at: datetime
    latency_ms: int = Field(default=0, ge=0)
    model: str | None = None
    tool: str | None = None
    input_summary: str = ""
    output_summary: str = ""
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0, description="Cost in the trace currency")
    cached: bool = False
    source_ids: list[str] = Field(default_factory=list)
    verdict: Literal["ok", "issues", "failed", "skipped"] | None = None
    issues: list[str] = Field(default_factory=list)


class TraceTotals(Contract):
    llm_calls: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    iterations: int = 0


class Trace(Contract):
    schema_version: str = SCHEMA_VERSION
    trace_id: str
    profile_id: str
    event_id: str
    role: ReaderRole | None = None
    language: str = "ru"
    insight_id: str | None = None
    config: dict = Field(default_factory=dict, description="Agent configuration of the run (for ablations)")
    currency: str = "USD"
    started_at: datetime
    finished_at: datetime | None = None
    status: Literal["running", "done", "failed"] = "running"
    steps: list[TraceStep] = Field(default_factory=list)
    totals: TraceTotals = Field(default_factory=TraceTotals)

    def recompute_totals(self, iterations: int | None = None) -> TraceTotals:
        self.totals = TraceTotals(
            llm_calls=sum(s.model is not None for s in self.steps),
            tool_calls=sum(s.kind is StepKind.TOOL_CALL for s in self.steps),
            tokens_in=sum(s.tokens_in for s in self.steps),
            tokens_out=sum(s.tokens_out for s in self.steps),
            cost=round(sum(s.cost for s in self.steps), 6),
            latency_ms=sum(s.latency_ms for s in self.steps),
            iterations=iterations if iterations is not None else self.totals.iterations,
        )
        return self.totals
