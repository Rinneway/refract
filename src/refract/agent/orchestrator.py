"""The orchestrator: tag → plan → tools → draft → self-check → revise → finalize.

Plain code without an agent framework, so every step is visible in the trace: model, tokens, cost, sources,
verdict. Self-check = rule checks in code (`checks.py`) + a semantic review by a second model. If data is missing
the agent searches again and revises the draft; errors that survive `max_iterations` are cut out so the reader
never sees an unsupported conclusion.

`AgentConfig` switches components off one by one — that is how the ablations in `refract.eval` are run.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from ..config import Settings
from ..i18n import norm_lang, t
from ..knowledge.store import DataStore
from ..llm import JsonResult, LLMError
from ..llm.client import LLM
from ..prompts import classify as classify_prompt
from ..prompts import headline as headline_prompt
from ..prompts import insight as insight_prompt
from ..prompts import plan as plan_prompt
from ..prompts import revise as revise_prompt
from ..prompts import verify as verify_prompt
from ..schemas import (
    AgentContext,
    AgentState,
    Insight,
    InsightContent,
    IssueCode,
    NewsEvent,
    Plan,
    PlanStep,
    ReaderRole,
    StepKind,
    ToolCall,
    ToolName,
    Trace,
    TraceStep,
    VerificationIssue,
    VerificationReport,
)
from ..textutil import invented_numbers
from . import checks
from .actions import enforce_actions
from .context import InsightContext
from .impact import DEFAULT_IMPACT, ImpactPolicy
from .policy import DEFAULT_POLICY, Policy
from .tools import Toolbox, summarize

log = logging.getLogger(__name__)

PROMPT_VERSIONS = {
    "classify": classify_prompt.VERSION,
    "plan": plan_prompt.VERSION,
    "insight": insight_prompt.VERSION,
    "verify": verify_prompt.VERSION,
    "revise": revise_prompt.VERSION,
    "headline": headline_prompt.VERSION,
}


@dataclass(frozen=True)
class AgentConfig:
    """Which components run. The defaults are the full agent."""

    language: str = "ru"
    max_iterations: int = 1  # revise rounds; a second round rarely helps and costs ~30 s
    planner: bool = True  # False — a fixed default plan instead of the model's plan
    llm_verifier: bool = True  # False — no second-model review
    rule_checks: bool = True  # False — no code checks (and no grounding validation of the result)
    impact_cap: bool = True  # False — the model sets the impact level freely
    action_policy: bool = True  # False — no policy rules: actions are whatever the model writes
    prune: bool = True  # False — statements with unresolved errors stay in the result
    max_plan_steps: int = 6

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


ABLATIONS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_plan": {"planner": False},
    "no_verifier": {"llm_verifier": False},
    "no_rules": {"rule_checks": False},
    "no_revise": {"max_iterations": 0},
    "no_self_check": {"llm_verifier": False, "rule_checks": False, "max_iterations": 0, "prune": False},
    "no_cap": {"impact_cap": False},
    "no_policy": {"action_policy": False},
}


def ablation(name: str, base: AgentConfig | None = None) -> AgentConfig:
    return replace(base or AgentConfig(), **ABLATIONS[name])


ProgressFn = Callable[[dict[str, Any]], None]


@dataclass
class AgentRun:
    insight: Insight | None
    trace: Trace
    state: AgentState
    issues: list[VerificationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.insight is not None


class Agent:
    def __init__(
        self,
        llm: LLM,
        store: DataStore,
        *,
        config: AgentConfig | None = None,
        settings: Settings | None = None,
        policy: Policy | None = None,
        impact_policy: ImpactPolicy | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        on_progress: ProgressFn | None = None,
    ) -> None:
        """`on_progress` receives {"status": "started"|"done"|"finished", "name", "label", "step"} events — the UI
        shows what the agent is doing right now (e.g. over SSE)."""
        self.llm, self.store = llm, store
        self.settings = settings or getattr(llm, "settings", None) or Settings()
        self.config = config or AgentConfig(language=self.settings.language)
        self.policy = policy or DEFAULT_POLICY
        self.impact_policy = impact_policy or DEFAULT_IMPACT
        self.now, self.on_progress = now, on_progress
        self._local = threading.local()  # per-thread run state: one Agent may serve several threads

    # ------------------------------------------------------------------ per-run state

    @property
    def _models(self) -> set[str]:
        return self._local.__dict__.setdefault("models", set())

    @_models.setter
    def _models(self, value: set[str]) -> None:
        self._local.models = value

    @property
    def _need_more(self) -> list[str]:
        return self._local.__dict__.setdefault("need_more", [])

    @_need_more.setter
    def _need_more(self, value: list[str]) -> None:
        self._local.need_more = value

    @property
    def lang(self) -> str:
        return self._local.__dict__.get("lang", norm_lang(self.config.language))

    # ------------------------------------------------------------------ run

    def run(
        self, profile_id: str, event: NewsEvent | str, role: str = "cfo", language: str | None = None
    ) -> AgentRun:
        """An insight for the subject about the event. Unknown subject or event — KeyError. The UI always gets a
        `finished` event (ok=False and an error on failure), so it never waits forever."""
        self._local.lang = norm_lang(language or self.config.language)
        try:
            return self._run(profile_id, event, role)
        except Exception as e:
            self._emit("finished", "finalize", extra={"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
            raise

    def _run(self, profile_id: str, event: NewsEvent | str, role: str) -> AgentRun:
        lang, cfg, inst = self.lang, self.config, self.settings.institution
        event = self.store.events[event] if isinstance(event, str) else event
        self.store.add_event(event)
        run_id = uuid.uuid4().hex[:10]
        reader = ReaderRole(role) if role in ReaderRole._value2member_map_ else ReaderRole.OTHER
        trace = Trace(
            trace_id=f"tr_{run_id}",
            profile_id=profile_id,
            event_id=event.event_id,
            role=reader,
            language=lang,
            config={**cfg.as_dict(), "language": lang, "prompts": PROMPT_VERSIONS},
            currency=self.settings.currency,
            started_at=self.now(),
        )
        state = AgentState(
            run_id=run_id,
            profile_id=profile_id,
            event_id=event.event_id,
            role=reader,
            language=lang,
            context=AgentContext(),
            max_iterations=cfg.max_iterations,
            trace=trace,
        )
        tools = Toolbox(self.store, profile_id, current_event_id=event.event_id, lang=lang, institution=inst)
        self._models, self._need_more = set(), []

        # 1. The profile always comes first, without the model.
        profile = self._tool(trace, tools, ToolCall(call_id="c0", tool=ToolName.GET_PROFILE)).data
        state.context.profile = profile

        # 2. Tag a fresh news item if the event has no factors.
        if not event.macro_factors:
            res = self._llm(
                trace,
                StepKind.LLM_CALL,
                "classify_news",
                classify_prompt.build_messages(event, lang),
                classify_prompt.NewsTags,
            )
            if res.value:
                event = classify_prompt.apply(event, res.value)
                self.store.add_event(event)
        state.context.event = event
        ctx = InsightContext(
            profile=profile,
            event=event,
            role=reader.value,
            lang=lang,
            institution=inst,
            policy=self.policy,
            impact_policy=self.impact_policy,
            use_cap=cfg.impact_cap,
            use_actions=cfg.action_policy,
            offer_products=self.settings.offer_products,
        )

        # 3. Plan and gather data.
        plan = self._plan(trace, ctx)
        state.plan = plan
        for step in plan.steps[: cfg.max_plan_steps]:
            self._apply(
                ctx,
                self._tool(
                    trace, tools, ToolCall(call_id=step.step_id, tool=step.tool, args=step.args), step.purpose
                ),
            )
            step.done = True

        # 4. Draft → review → revise.
        draft_res = self._llm(
            trace, StepKind.DRAFT, "draft_insight", insight_prompt.build_messages(ctx), InsightContent
        )
        draft = draft_res.value
        issues: list[VerificationIssue] = [] if draft else [self._schema_issue(draft_res)]
        while True:
            if draft is not None:
                issues = self._verify(trace, ctx, draft)
            state.draft = draft
            state.verification = VerificationReport(
                passed=draft is not None and not checks.errors(issues),
                issues=issues,
                need_more_data=self._need_more,
            )
            if draft is not None and not checks.errors(issues):
                break
            if state.iteration >= cfg.max_iterations:
                break
            state.iteration += 1
            for q in (state.verification.need_more_data if state.verification else [])[:2]:
                call = ToolCall(
                    call_id=f"r{state.iteration}_{q[:12]}",
                    tool=ToolName.SEARCH_RESEARCH,
                    args={"query": q, "limit": 2},
                )
                purpose = (
                    "extra search requested by the review"
                    if lang == "en"
                    else "дополнительный поиск по замечанию проверки"
                )
                self._apply(ctx, self._tool(trace, tools, call, purpose))
            if draft is None:
                res = self._llm(
                    trace,
                    StepKind.REVISE,
                    "redraft",
                    insight_prompt.build_messages(ctx),
                    InsightContent,
                    route_name="draft_insight",
                )
            else:
                res = self._llm(
                    trace,
                    StepKind.REVISE,
                    "revise",
                    revise_prompt.build_messages(ctx, draft, checks.errors(issues)),
                    InsightContent,
                )
            draft = res.value or draft

        # 5. Finalize.
        insight = self._finalize(trace, ctx, state, draft, issues)
        trace.finished_at = self.now()
        trace.status = "done" if insight else "failed"
        trace.insight_id = insight.insight_id if insight else None
        trace.recompute_totals(iterations=state.iteration)
        state.context.theses, state.context.research, state.context.related_events = (
            ctx.theses,
            ctx.research,
            ctx.related,
        )
        state.context.sources = {s.source_id: s for s in ctx.sources()}
        self._emit(
            "finished",
            "finalize",
            extra={
                "ok": insight is not None,
                "insight_id": trace.insight_id,
                "totals": trace.totals.model_dump(),
            },
        )
        return AgentRun(insight=insight, trace=trace, state=state, issues=issues)

    # ------------------------------------------------------------------ steps

    def default_plan(self, event: NewsEvent) -> list[PlanStep]:
        en = self.lang == "en"
        return [
            PlanStep(
                step_id="s0",
                tool=ToolName.GET_HOUSE_VIEW,
                args={"factors": event.macro_factors, "query": event.title, "limit": 6},
                purpose="house view on the event's factors"
                if en
                else "позиция института по факторам события",
            ),
            PlanStep(
                step_id="s_cp",
                tool=ToolName.GET_COUNTERPARTIES,
                args={"limit": 5},
                purpose="industries of counterparties" if en else "отрасли контрагентов",
            ),
            PlanStep(
                step_id="s_rag",
                tool=ToolName.SEARCH_RESEARCH,
                args={"query": event.title, "limit": 3},
                purpose="research excerpts on the event" if en else "фрагменты аналитики по теме события",
            ),
        ]

    def _plan(self, trace: Trace, ctx: InsightContext) -> Plan:
        event = ctx.event
        if not self.config.planner:
            self._record(
                trace,
                TraceStep(
                    index=len(trace.steps),
                    kind=StepKind.PLAN,
                    name="plan",
                    started_at=self.now(),
                    verdict="skipped",
                    issues=[t("note.default_plan", self.lang)],
                ),
            )
            return Plan(rationale=t("note.default_plan", self.lang), steps=self.default_plan(event))
        msgs = plan_prompt.build_messages(ctx.profile, event, self.lang, self.settings.institution)
        res = self._llm(trace, StepKind.PLAN, "plan", msgs, Plan)
        steps = [s for s in (res.value.steps if res.value else []) if s.tool is not ToolName.GET_PROFILE]
        defaults = self.default_plan(event)
        if not any(s.tool is ToolName.GET_HOUSE_VIEW for s in steps):  # the house view is mandatory
            steps.insert(0, defaults[0])
        if res.value is None:  # the model did not answer in time — default plan
            steps += defaults[1:]
        rationale = res.value.rationale if res.value else t("note.default_plan", self.lang)
        return Plan(rationale=rationale, steps=steps)

    def _verify(self, trace: Trace, ctx: InsightContext, draft: InsightContent) -> list[VerificationIssue]:
        rule: list[VerificationIssue] = []
        if self.config.rule_checks:
            started = time.perf_counter()
            self._emit("started", "rule_check")
            rule = checks.rule_check(draft, ctx)
            self._record(
                trace,
                TraceStep(
                    index=len(trace.steps),
                    kind=StepKind.VERIFY,
                    name="rule_check",
                    started_at=self.now(),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    verdict="issues" if rule else "ok",
                    issues=[f"{i.severity}: {i.message}" for i in rule][:20],
                ),
            )
        if not self.config.llm_verifier:
            self._need_more = []
            return rule
        res = self._llm(
            trace,
            StepKind.VERIFY,
            "verify",
            verify_prompt.build_messages(ctx, draft),
            VerificationReport,
            report=False,
        )
        llm_issues = res.value.issues if res.value else []
        confirmed: list[str] = []
        if self.config.rule_checks:
            llm_issues, confirmed = checks.confirm_by_sources(llm_issues, draft, ctx)
        merged = checks.merge(rule, llm_issues)
        self._need_more = res.value.need_more_data if res.value else []
        trace.steps[-1].verdict = "ok" if not checks.errors(merged) else "issues"
        trace.steps[-1].issues = (
            confirmed + [f"{i.severity} {i.claim_id or ''}: {i.code.value} — {i.message}" for i in llm_issues]
        )[:20]
        self._emit("done", "verify", trace.steps[-1])
        return merged

    def _finalize(
        self,
        trace: Trace,
        ctx: InsightContext,
        state: AgentState,
        draft: InsightContent | None,
        issues: list[VerificationIssue],
    ) -> Insight | None:
        lang, cfg = self.lang, self.config
        started = time.perf_counter()
        self._emit("started", "finalize")
        verdict, notes = "ok", []
        insight = None
        if draft is not None:
            errs = {i.claim_id for i in checks.errors(issues) if i.claim_id}
            bad = errs - {checks.HEADLINE}
            removed_house = [c for c in draft.house_view if c.claim_id in bad]
            if bad and cfg.prune:
                hard = (
                    {i.claim_id for i in checks.errors(checks.rule_check(draft, ctx)) if i.claim_id}
                    if cfg.rule_checks
                    else set()
                )
                pruned, restored = checks.keep_required(checks.drop_claims(draft, bad), draft, hard, lang)
                draft = pruned
                verdict, notes = "issues", [t("note.pruned", lang, ids=sorted(bad)), *restored]
            fallback = (
                checks.house_view_from_theses(ctx, removed_house)
                if cfg.prune and not draft.house_view and removed_house
                else []
            )
            if fallback:
                draft = draft.model_copy(update={"house_view": fallback})
                notes.append(t("note.house_view_restored", lang))
            if checks.HEADLINE in errs and cfg.prune:
                reasons = [i.message for i in checks.errors(issues) if i.claim_id == checks.HEADLINE]
                headline, note = self._rewrite_headline(trace, ctx, draft, reasons)
                draft = draft.model_copy(update={"headline": headline})
                verdict = "issues"
                notes.append(note)
            if cfg.impact_cap:
                cap = ctx.cap()
                draft, capped = checks.clamp_impact(draft, cap.level)
                if capped:
                    notes.append(t("note.capped", lang, level=cap.level.value, reason=cap.reason, ids=capped))
            if cfg.action_policy:
                draft, action_notes = enforce_actions(
                    draft,
                    ctx.candidates(),
                    ctx.event,
                    policy=self.policy,
                    lang=lang,
                    offer_products=self.settings.offer_products,
                )
                notes += action_notes
            known = {s.source_id for s in ctx.sources()}
            draft = draft.model_copy(
                update={
                    "scenarios": [
                        sc.model_copy(update={"source_ids": [s for s in sc.source_ids if s in known]})
                        for sc in draft.scenarios
                    ]
                }
            )
            used = {s for c in draft.all_claims() for s in c.source_ids} | {
                s for sc in draft.scenarios for s in sc.source_ids
            }
            fields = {
                **draft.model_dump(),
                "insight_id": f"ins_{state.run_id}",
                "profile_id": state.profile_id,
                "event_id": state.event_id,
                "role": state.role,
                "language": lang,
                "created_at": self.now(),
                "sources": [s for s in ctx.sources() if s.source_id in used] or ctx.sources()[:1],
                "trace_id": trace.trace_id,
                "models": sorted(self._models),
                "disclaimer": t("disclaimer", lang, **ctx.inst()),
            }
            try:
                insight = Insight(**fields)
            except ValidationError as e:
                if cfg.rule_checks:
                    verdict, notes = "failed", [t("note.invalid", lang, msg=e.errors()[0]["msg"])]
                else:  # ablation without rule checks: keep the raw result for evaluation
                    insight = Insight.model_construct(**{**fields, "validated": False})
                    notes.append(t("note.unvalidated", lang))
        else:
            verdict, notes = "failed", [t("note.no_draft", lang)]
        self._record(
            trace,
            TraceStep(
                index=len(trace.steps),
                kind=StepKind.FINALIZE,
                name="finalize",
                started_at=self.now(),
                latency_ms=int((time.perf_counter() - started) * 1000),
                verdict=verdict,
                issues=notes,
                source_ids=[s.source_id for s in insight.sources] if insight else [],
            ),
        )
        return insight

    # ------------------------------------------------------------------ helpers

    def label(self, name: str) -> str:
        key = f"step.{name}"
        try:
            inst = self.settings.institution
            return t(
                key, self.lang, institution=inst.display(self.lang), institution_gen=inst.genitive(self.lang)
            )
        except KeyError:
            return name

    def _record(self, trace: Trace, step: TraceStep, report: bool = True) -> None:
        trace.steps.append(step)
        if report:
            self._emit("done", step.name, step)

    def _emit(self, status: str, name: str, step: TraceStep | None = None, extra: dict | None = None) -> None:
        if not self.on_progress:
            return
        event = {
            "status": status,
            "name": name,
            "label": self.label(name),
            "step": step.model_dump(mode="json") if step else None,
            **(extra or {}),
        }
        try:
            self.on_progress(event)
        except Exception as e:  # noqa: BLE001 — a UI error must not break the agent
            log.warning("on_progress: %r", e)

    def _rewrite_headline(
        self, trace: Trace, ctx: InsightContext, draft: InsightContent, reasons: list[str]
    ) -> tuple[str, str]:
        """The headline failed the review: a short headline from verified statements, else the first statement."""
        msgs = headline_prompt.build_messages(draft, reasons, self.lang, ctx.inst())
        res = self._llm(trace, StepKind.REVISE, "headline", msgs, headline_prompt.HeadlineOut)
        new = res.value.headline.strip() if res.value else ""
        if new and len(new) <= 160 and not invented_numbers([new], ctx.render()):
            return new, t("note.headline_rewritten", self.lang)
        return checks.short_headline(draft), t("note.headline_fallback", self.lang)

    def _apply(self, ctx: InsightContext, result) -> None:
        if not result.ok or not result.data:
            return
        if result.tool is ToolName.GET_HOUSE_VIEW:
            have = {th.thesis_id for th in ctx.theses}
            ctx.theses += [th for th in result.data if th.thesis_id not in have]
        elif result.tool is ToolName.SEARCH_RESEARCH:
            have = {c.chunk_id for c in ctx.research}
            ctx.research += [c for c in result.data if c.chunk_id not in have]
        elif result.tool is ToolName.SEARCH_NEWS:
            have = {e.event_id for e in ctx.related}
            ctx.related += [e for e in result.data if e.event_id not in have]
        # profile, counterparties and group are already in the context through the profile

    def _tool(self, trace: Trace, tools: Toolbox, call: ToolCall, purpose: str = "") -> Any:
        started = time.perf_counter()
        self._emit("started", call.tool.value)
        result = tools.call(call)
        self._record(
            trace,
            TraceStep(
                index=len(trace.steps),
                kind=StepKind.TOOL_CALL,
                name=call.tool.value,
                tool=call.tool.value,
                started_at=self.now(),
                latency_ms=int((time.perf_counter() - started) * 1000),
                input_summary=(
                    (f"{purpose}; " if purpose else "") + ", ".join(f"{k}={v}" for k, v in call.args.items())
                )[:300],
                output_summary=summarize(result, self.lang),
                source_ids=[s.source_id for s in result.sources],
                verdict="ok" if result.ok else "failed",
            ),
        )
        return result

    def _llm(
        self,
        trace: Trace,
        kind: StepKind,
        name: str,
        messages,
        schema,
        route_name: str | None = None,
        report: bool = True,
    ) -> JsonResult:
        """report=False — the caller sends the `done` event after it fills in the verdict."""
        r = self.settings.route(route_name or name)
        self._emit("started", name)
        res = self._call(r.model, messages, schema, r, name)
        if res.value is None and r.fallback:  # fallback model
            res = self._call(r.fallback, messages, schema, r, f"{name}:fallback")
        model = res.calls[-1].model if res.calls else self.settings.model_id(r.model)
        self._models.add(model)
        self._record(
            trace,
            TraceStep(
                index=len(trace.steps),
                kind=kind,
                name=name,
                started_at=self.now(),
                latency_ms=res.latency_ms,
                model=model,
                tokens_in=res.tokens_in,
                tokens_out=res.tokens_out,
                cost=round(res.cost, 6),
                cached=bool(res.calls) and all(c.cached for c in res.calls),
                verdict="ok" if res.ok else "failed",
                issues=[res.error] if res.error else [],
            ),
            report=report,
        )
        return res

    def _call(self, model: str, messages, schema, r, step: str) -> JsonResult:
        try:
            return self.llm.chat_json(
                model,
                messages,
                schema,
                max_tokens=r.max_tokens,
                temperature=r.temperature,
                step=step,
                timeout_s=r.timeout_s,
            )
        except LLMError as e:  # timeout or unreachable endpoint — the step fails, the agent moves on
            return JsonResult(value=None, error=str(e)[:300], calls=[])

    @staticmethod
    def _schema_issue(res: JsonResult) -> VerificationIssue:
        return VerificationIssue(
            code=IssueCode.SCHEMA_ERROR, severity="error", message=res.error or "no schema-valid answer"
        )
