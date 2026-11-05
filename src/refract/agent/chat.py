"""Follow-up chat over a finished insight.

A chat turn is a small agent: the model decides whether data is enough, calls up to three tools if needed (another
angle: suppliers, group, another research topic), answers by a schema with an origin for every statement, and code
checks citations and numbers. Statements that fail the checks are dropped. The context is the insight's
(`AgentState`), so the answer stays consistent with it; every turn writes its own trace.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from ..i18n import norm_lang, t
from ..knowledge.store import DataStore
from ..llm.client import LLM
from ..prompts import chat as chat_prompt
from ..schemas import (
    AgentState,
    ChatAnswer,
    ChatIntent,
    ChatResponse,
    ChatTurn,
    Claim,
    Insight,
    Origin,
    SourceType,
    StepKind,
    ToolCall,
    ToolName,
    Trace,
    TraceStep,
)
from ..textutil import invented_numbers
from .context import InsightContext
from .orchestrator import Agent, ProgressFn
from .tools import Toolbox


@dataclass
class ChatRun:
    response: ChatResponse
    trace: Trace


def context_from_state(state: AgentState, agent: Agent | None = None) -> InsightContext:
    c = state.context
    if c.profile is None or c.event is None:
        raise ValueError("the run state has no profile or event: it ended before the agent loaded them")
    kwargs = {}
    if agent is not None:
        kwargs = {
            "institution": agent.settings.institution,
            "policy": agent.policy,
            "impact_policy": agent.impact_policy,
            "use_cap": agent.config.impact_cap,
            "use_actions": agent.config.action_policy,
            "offer_products": agent.settings.offer_products,
        }
    return InsightContext(
        profile=c.profile,
        event=c.event,
        theses=list(c.theses),
        role=state.role.value if state.role else "cfo",
        research=list(c.research),
        related=list(c.related_events),
        lang=state.language,
        **kwargs,
    )


def render_extra(result, lang: str = "ru") -> str:
    """Tool data that is not in the shared context (counterparties, group) as text for the model."""
    en = norm_lang(lang) == "en"
    if result.tool is ToolName.GET_COUNTERPARTIES:
        lines = []
        for cp in result.data:
            who = (
                ("customer" if cp.role == "customer" else "supplier")
                if en
                else ("покупатель" if cp.role == "customer" else "поставщик")
            )
            activity = (
                f"{cp.industry.code} {cp.industry.name}"
                if cp.industry
                else ("industry unknown" if en else "отрасль неизвестна")
            )
            region = cp.region or ("region unknown" if en else "регион неизвестен")
            flow = "of the flow" if en else "потока"
            lines.append(
                f"- [src_payments] {who} {cp.counterparty_id}: {activity}, {region}, {cp.share_of_flow:.0%} {flow}"
            )
        return ("Counterparties by payments:\n" if en else "Контрагенты по платежам:\n") + "\n".join(lines)
    if result.tool is ToolName.GET_GROUP:
        d = result.data
        aff = "; ".join(
            f"{a.profile_id} ({a.kind}, {a.confidence:.0%}: {a.evidence})" for a in d["affiliations"]
        )
        none = "none" if en else "нет"
        if en:
            return f"[src_profile] Group {d['group_id']}: {len(d['members'])} other entities. Links: {aff or none}"
        return f"[src_profile] Группа {d['group_id']}: {len(d['members'])} других юрлиц. Связи: {aff or none}"
    return ""


class ChatAgent:
    def __init__(
        self,
        llm: LLM,
        store: DataStore,
        *,
        agent: Agent | None = None,
        now=lambda: datetime.now(timezone.utc),
        on_progress: ProgressFn | None = None,
    ) -> None:
        self.store, self.now = store, now
        # shared steps with trace and progress events: model calls and tool calls
        self._steps = agent or Agent(llm, store, now=now, on_progress=on_progress)
        if on_progress is not None:
            self._steps.on_progress = on_progress

    def answer(
        self, state: AgentState, insight: Insight, question: str, history: list[ChatTurn] | None = None
    ) -> ChatRun:
        self._steps._local.lang = norm_lang(state.language)
        try:
            return self._answer(state, insight, question, history or [])
        except Exception as e:
            self._steps._emit(
                "finished", "chat", extra={"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
            )
            raise

    def _answer(self, state: AgentState, insight: Insight, question: str, history: list[ChatTurn]) -> ChatRun:
        steps, lang = self._steps, norm_lang(state.language)
        steps._models, steps._need_more = set(), []
        trace = Trace(
            trace_id=f"tr_chat_{uuid.uuid4().hex[:8]}",
            profile_id=state.profile_id,
            event_id=state.event_id,
            role=state.role,
            language=lang,
            insight_id=insight.insight_id,
            currency=steps.settings.currency,
            started_at=self.now(),
        )
        ctx = context_from_state(state, steps)
        tools = Toolbox(
            self.store,
            state.profile_id,
            current_event_id=state.event_id,
            lang=lang,
            institution=ctx.institution,
        )

        # 1. Is new data needed?
        plan = steps._llm(
            trace,
            StepKind.PLAN,
            "chat_plan",
            chat_prompt.plan_messages(insight, ctx, question, history),
            chat_prompt.ChatPlan,
        )
        intent = plan.value.intent if plan.value else ChatIntent.CLARIFY
        extra: list[str] = []
        if plan.value and intent is not ChatIntent.OUT_OF_SCOPE:
            for step in plan.value.steps[:3]:
                if step.tool is ToolName.GET_PROFILE:
                    continue
                res = steps._tool(
                    trace, tools, ToolCall(call_id=step.step_id, tool=step.tool, args=step.args), step.purpose
                )
                steps._apply(ctx, res)
                if res.ok and res.data:
                    extra.append(render_extra(res, lang))

        # 2. Answer.
        new_data = "\n\n".join(x for x in extra if x)
        res = steps._llm(
            trace,
            StepKind.LLM_CALL,
            "chat",
            chat_prompt.answer_messages(insight, ctx, question, history, new_data),
            ChatAnswer,
        )
        answer = res.value or ChatAnswer(intent=intent, answer=t("chat.no_data", lang))

        # 3. Code checks.
        steps._emit("started", "chat_check")
        answer, notes = self._check(answer, ctx, insight, new_data)
        steps._record(
            trace,
            TraceStep(
                index=len(trace.steps),
                kind=StepKind.VERIFY,
                name="chat_check",
                started_at=self.now(),
                verdict="issues" if notes else "ok",
                issues=notes,
            ),
        )

        by_id = {s.source_id: s for s in ctx.sources()} | {s.source_id: s for s in insight.sources}
        used = [s for c in answer.claims for s in c.source_ids]
        if answer.intent is ChatIntent.SOURCE and not used:
            used = [s for c in insight.all_claims() for s in c.source_ids]
        sources = [by_id[s] for s in dict.fromkeys(used) if s in by_id]

        trace.finished_at, trace.status = self.now(), "done"
        trace.recompute_totals(iterations=0)
        response = ChatResponse(
            insight_id=insight.insight_id,
            question=question,
            intent=answer.intent,
            answer=answer.answer,
            claims=answer.claims,
            sources=sources,
            follow_ups=answer.follow_ups,
            trace_id=trace.trace_id,
            language=lang,
            created_at=self.now(),
            disclaimer=insight.disclaimer,
        )
        steps._emit(
            "finished",
            "chat",
            extra={"ok": True, "trace_id": trace.trace_id, "totals": trace.totals.model_dump()},
        )
        return ChatRun(response=response, trace=trace)

    def _check(
        self, answer: ChatAnswer, ctx: InsightContext, insight: Insight, new_data: str
    ) -> tuple[ChatAnswer, list[str]]:
        en = ctx.lang == "en"
        sources = {s.source_id: s for s in ctx.sources()} | {s.source_id: s for s in insight.sources}
        context = "\n".join(
            [ctx.render(), chat_prompt.insight_block(insight, ctx.lang, ctx.inst()), new_data]
        )
        kept: list[Claim] = []
        notes: list[str] = []
        for c in answer.claims:
            unknown = [s for s in c.source_ids if s not in sources]
            problem = None
            if unknown:
                problem = f"unknown sources {unknown}" if en else f"нет источников {unknown}"
            elif c.origin is Origin.FACT and not c.source_ids:
                problem = "fact without a source" if en else "факт без источника"
            elif c.origin is Origin.HOUSE_VIEW and not any(
                sources[s].type is SourceType.RESEARCH for s in c.source_ids
            ):
                problem = "house view without research" if en else "позиция института без его аналитики"
            elif nums := invented_numbers([c.text], context):
                problem = (
                    f"numbers outside the context {sorted(nums)}"
                    if en
                    else f"числа вне контекста {sorted(nums)}"
                )
            if problem:
                notes.append(f"{c.claim_id}: {problem} — {'removed' if en else 'убрано'}")
            else:
                kept.append(c)
        text = answer.answer
        if not text.strip():
            notes.append(
                "the model gave no answer text — built from verified statements"
                if en
                else "модель не дала текста ответа — он собран из проверенных утверждений"
            )
            text = " ".join(c.text for c in kept) or t("chat.no_data", ctx.lang)
        elif invented_numbers([text], context):
            notes.append(
                "numbers outside the context in the answer — rebuilt from verified statements"
                if en
                else "в тексте ответа числа вне контекста — ответ собран из проверенных утверждений"
            )
            text = " ".join(c.text for c in kept) or t("chat.no_data", ctx.lang)
        return answer.model_copy(update={"answer": text, "claims": kept}), notes
