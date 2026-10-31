"""Follow-up chat prompts: decide whether new data is needed, then answer with an origin for every statement."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..agent.tools import tool_specs
from ..i18n import norm_lang, t
from ..schemas import ChatIntent, ChatTurn, Insight, PlanStep
from ..schemas.vocab import factors_prompt, topics_prompt
from ..textutil import untrusted
from . import output_language, rule

VERSION = "chat-v4"
HISTORY_TURNS = 6


class ChatPlan(BaseModel):
    intent: ChatIntent
    steps: list[PlanStep] = Field(default_factory=list, max_length=3)


PLAN_SYSTEM = {
    "ru": """Читатель задаёт уточняющий вопрос по готовому инсайту. Определи тип вопроса и реши, нужны ли данные \
сверх того, что уже есть в инсайте и контексте.

intent: clarify — уточнить вывод инсайта; slice — посмотреть в другом разрезе (поставщики, покупатели, группа, \
другой фактор, другая тема аналитики); source — показать первоисточник; out_of_scope — вопрос не о субъекте, \
событии или экономике.

steps — от 0 до 3 вызовов инструментов, только если ответа нет в контексте. Для clarify и source обычно хватает \
контекста — тогда steps пустой. Для out_of_scope steps всегда пустой.

Инструменты:
{tools}

Темы тезисов: {topics}
Факторы:
{factors}

{untrusted}""",
    "en": """The reader asks a follow-up question about a finished insight. Decide what kind of question it is and \
whether data beyond the insight and its context is needed.

intent: clarify — clarify a conclusion of the insight; slice — look from another angle (suppliers, customers, \
group, another factor, another research topic); source — show the primary source; out_of_scope — not about the \
subject, the event or the economy.

steps — 0 to 3 tool calls, only if the answer is not in the context. For clarify and source the context is usually \
enough — then steps is empty. For out_of_scope steps is always empty.

Tools:
{tools}

Thesis topics: {topics}
Factors:
{factors}

{untrusted}""",
}

ANSWER_SYSTEM = {
    "ru": """Ты — аналитик {institution_gen}. Читатель задаёт уточняющий вопрос по инсайту, который ты подготовил. \
Ответь по существу, опираясь только на контекст: инсайт, данные о субъекте, источники и новые данные.

Правила:
1. answer — прямой ответ на вопрос, 2–5 предложений, без пересказа всего инсайта.
2. claims — ключевые утверждения ответа (до 5), у каждого origin: fact — из источника или данных о субъекте; \
house_view — только позиция института из его аналитики; ai — твой вывод; hypothesis — допущение с оговоркой в caveat. \
Указывай source_ids только из списка источников. claim_id: q1, q2…
3. Числа — только из контекста. Если нужных данных нет, прямо скажи, каких именно, и что можно сделать: уточнить у \
менеджера, дать сведения о себе, посмотреть первоисточник.
4. Если просят первоисточник — назови документ, страницу или издание и приведи короткую цитату из источника.
5. Вопрос не о субъекте, событии или экономике — intent out_of_scope: вежливо скажи, что помогаешь с анализом \
событий для бизнеса и личных финансов, claims пустой.
6. Не давай индивидуальных инвестиционных, налоговых или юридических рекомендаций и не обещай результатов.
7. follow_ups — 2–3 коротких вопроса, которые логично задать дальше.
{language}

{untrusted}""",
    "en": """You are an analyst at {institution}. The reader asks a follow-up question about the insight you \
prepared. Answer to the point, relying only on the context: the insight, the subject data, the sources and new data.

Rules:
1. answer — a direct answer, 2–5 sentences, without retelling the whole insight.
2. claims — the key statements of the answer (up to 5), each with an origin: fact — from a source or the subject \
data; house_view — only the institution's position from its research; ai — your inference; hypothesis — an \
assumption with a caveat. source_ids only from the list of sources. claim_id: q1, q2…
3. Numbers only from the context. If the data needed is missing, say exactly which data, and what can be done: ask \
the manager, provide details about yourself, look at the primary source.
4. If asked for the primary source, name the document, page or publisher and give a short quote from it.
5. Not about the subject, the event or the economy — intent out_of_scope: politely say that you help analyse events \
for business and personal finances; claims empty.
6. Do not give individual investment, tax or legal advice and do not promise results.
7. follow_ups — 2–3 short questions that make sense to ask next.
{language}

{untrusted}""",
}
LABELS = {
    "ru": {
        "headline": "Заголовок", "impact": "Влияние", "direction": "направление", "sources": "источники",
        "action": "Действие", "insight": "Инсайт", "features": "Признаки", "question": "Вопрос",
        "new_data": "Новые данные по вопросу", "task": "Ответь строго по JSON-схеме.",
        "blocks": ("Что произошло", "Почему важно", "Риски", "Возможности", "Позиция института"),
    },
    "en": {
        "headline": "Headline", "impact": "Impact", "direction": "direction", "sources": "sources",
        "action": "Action", "insight": "Insight", "features": "Features", "question": "Question",
        "new_data": "New data for the question", "task": "Answer strictly by the JSON schema.",
        "blocks": ("What happened", "Why it matters", "Risks", "Opportunities", "House view"),
    },
}  # fmt: skip


def insight_block(ins: Insight, lang: str = "ru", inst: dict | None = None) -> str:
    lang = norm_lang(lang)
    inst = inst or {"institution": "Demo Bank", "institution_gen": "Демо-банка"}
    L = LABELS[lang]
    lines = [
        f"{L['headline']}: {ins.headline}",
        f"{L['impact']}: {ins.overall_impact.value}, {L['direction']}: {ins.overall_direction.value}",
    ]
    blocks = (ins.what_happened, ins.why_it_matters, ins.risks, ins.opportunities, ins.house_view)
    for title, block in zip(L["blocks"], blocks, strict=True):
        for c in block:
            label = t(f"origin.{c.origin.value}", lang, **inst)
            lines.append(
                f"- {title} [{c.claim_id}, {label}]: {c.text} ({L['sources']}: {', '.join(c.source_ids)})"
            )
    for a in ins.actions:
        lines.append(f"- {L['action']} [{a.action_id}]: {a.text}")
    return "\n".join(lines)


def history_messages(history: list[ChatTurn]) -> list[dict[str, str]]:
    return [{"role": h.role, "content": h.text} for h in history[-HISTORY_TURNS:]]


def plan_messages(ins: Insight, ctx, question: str, history: list[ChatTurn]) -> list[dict[str, str]]:
    lang = norm_lang(ctx.lang)
    L = LABELS[lang]
    system = PLAN_SYSTEM[lang].format(
        tools=tool_specs(lang, ctx.institution),
        topics=topics_prompt(lang),
        factors=factors_prompt(lang),
        untrusted=rule(lang),
    )
    feats = "\n".join(f"- {k} — {v}" for k, v in ctx.features())
    user = f"{L['insight']}:\n{insight_block(ins, lang, ctx.inst())}\n\n{L['features']}:\n{feats}\n\n{L['question']}: {question}"
    return [
        {"role": "system", "content": system},
        *history_messages(history),
        {"role": "user", "content": user},
    ]


def answer_messages(
    ins: Insight, ctx, question: str, history: list[ChatTurn], new_data: str
) -> list[dict[str, str]]:
    lang = norm_lang(ctx.lang)
    L = LABELS[lang]
    system = ANSWER_SYSTEM[lang].format(language=output_language(lang), untrusted=rule(lang), **ctx.inst())
    user = (
        f"## {L['insight']}\n{insight_block(ins, lang, ctx.inst())}\n\n{ctx.render()}\n\n"
        + (f"## {L['new_data']}\n{untrusted(new_data)}\n\n" if new_data else "")
        + f"{L['question']}: {question}\n\n{L['task']}"
    )
    return [
        {"role": "system", "content": system},
        *history_messages(history),
        {"role": "user", "content": user},
    ]
