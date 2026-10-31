"""LLM-as-a-judge over the six questions every insight must answer, and two baselines to compare with.

The judge sees the same context the evaluated system saw and the answer in one neutral text form, without knowing
which system wrote it. Scores 0–2 per question, overall 1–5, a list of unsupported statements and two flags.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..i18n import norm_lang, t
from ..schemas import InsightContent

VERSION = "judge-v2"

CRITERIA = ("what_happened", "why_it_matters", "consequences", "actions", "house_view", "grounding")


class Criterion(BaseModel):
    note: str = Field(description="One or two sentences: what is good and what is missing")
    score: int = Field(ge=0, le=2)


class JudgeScore(BaseModel):
    unsupported: list[str] = Field(
        default_factory=list, max_length=6, description="Statements absent from or contradicting the context"
    )
    what_happened: Criterion
    why_it_matters: Criterion
    consequences: Criterion
    actions: Criterion
    house_view: Criterion
    grounding: Criterion
    retelling: bool = Field(
        description="The answer merely retells the news without interpreting it for the subject"
    )
    impact_calibrated: bool = Field(
        description="The stated impact level matches how much the event really concerns the subject"
    )
    overall: int = Field(ge=1, le=5)
    comment: str = Field(description="The main weakness in one sentence")

    def total(self) -> int:
        return sum(getattr(self, c).score for c in CRITERIA)


JUDGE_SYSTEM = {
    "ru": """Ты — строгий эксперт и оцениваешь сервис новостной аналитики. Сервис получает новость и данные о \
субъекте (компания, ИП, самозанятый или частное лицо) и пишет интерпретацию для читателя. Ты видишь КОНТЕКСТ — всё, \
что было доступно сервису (данные о субъекте, новость, аналитика {institution_gen}), — и ОТВЕТ сервиса.

Сначала выпиши в unsupported утверждения ответа, которых нет в контексте или которые ему противоречат: выдуманные \
факты, числа, даты, участники, позиция института без опоры на его материалы. Логичные выводы из контекста, явно \
помеченные как вывод AI или гипотеза, выдумкой не считаются. Затем оцени ответ по шести вопросам, 0 — нет или \
плохо, 1 — частично, 2 — хорошо:

1. what_happened — «Что произошло?»: событие описано верно, есть ключевые факты, участники, сроки и причины.
2. why_it_matters — «Почему это важно именно этому субъекту?»: связь события с конкретным делом или личными \
финансами через признаки субъекта (род деятельности, продукты, контрагенты, регион, ВЭД, налоговый режим, масштаб). \
Общие слова, верные для любого, — 0.
3. consequences — «Какие последствия возможны?»: риски, возможности и факторы развития; учтено больше одного \
очевидного следствия.
4. actions — «Что можно сделать?»: конкретные шаги, вопросы для анализа или сценарии, вытекающие из ситуации \
субъекта. Шаблонные советы («следите за ситуацией») — 0.
5. house_view — «Как это оценивает {institution}?»: позиция есть, опирается на материалы из контекста, относится к \
событию, и видно, что — позиция института, а что — вывод модели. Нет позиции или она выдумана — 0; если в контексте \
нет материалов по теме и ответ честно об этом говорит — 1.
6. grounding — «На чём основан вывод?»: выводы связаны с источниками и подтверждаются ими; нет выдуманных фактов и \
чисел. Ссылок на источники нет — не больше 1.

retelling — true, если ответ по сути пересказывает новость и не объясняет, что она значит для этого субъекта.
impact_calibrated — true, если заявленный уровень влияния (low/medium/high) соответствует тому, насколько событие \
действительно касается субъекта.
overall — общая полезность для читателя, 1–5: 5 — можно принимать решения, 3 — полезно, но поверхностно или с \
ошибками, 1 — бесполезно или вредно.

Длина ответа сама по себе не достоинство. Будь строг и последователен. Пиши по-русски.""",
    "en": """You are a strict expert evaluating a news analytics service. The service receives a news item and \
data about a subject (a company, a sole proprietor, a self-employed person or a private individual) and writes an \
interpretation for the reader. You see the CONTEXT — everything available to the service (subject data, the news, \
research by {institution}) — and the service's ANSWER.

First list in unsupported the statements of the answer that are absent from the context or contradict it: invented \
facts, numbers, dates, participants, an institutional view not backed by its materials. Logical inferences from the \
context that are clearly marked as AI inference or hypothesis are not fabrications. Then score the answer on six \
questions, 0 — missing or poor, 1 — partly, 2 — good:

1. what_happened — "What happened?": the event is described correctly, with key facts, participants, dates and causes.
2. why_it_matters — "Why does it matter to this subject?": the link from the event to their business or personal \
finances through their features (activity, products, counterparties, region, foreign trade, tax regime, scale). \
Generic statements true for anyone — 0.
3. consequences — "What may follow?": risks, opportunities and drivers; more than one obvious consequence.
4. actions — "What can be done?": concrete steps, questions for analysis or scenarios that follow from the \
subject's situation. Boilerplate ("monitor the situation") — 0.
5. house_view — "What does {institution} think?": a position is present, backed by materials in the context, \
relevant to the event, and it is clear what is the institution's position and what is the model's inference. No \
position or a fabricated one — 0; if the context has no research on the topic and the answer says so honestly — 1.
6. grounding — "What is it based on?": conclusions are tied to sources and supported by them; no invented facts or \
numbers. No source citations — at most 1.

retelling — true if the answer essentially retells the news without explaining what it means for this subject.
impact_calibrated — true if the stated impact level (low/medium/high) matches how much the event really concerns \
the subject.
overall — usefulness for the reader, 1–5: 5 — good enough to act on, 3 — useful but shallow or with errors, \
1 — useless or harmful.

Length is not a merit in itself. Be strict and consistent. Write in English.""",
}

_ANSWER = {
    "ru": {
        "headline": "Заголовок", "impact": "Влияние на субъекта", "direction": "направление", "caveat": "оговорка",
        "sources": "источники", "impact_short": "влияние", "horizon": "горизонт", "scenarios": "Сценарии",
        "actions": "Что можно сделать", "rationale": "обоснование", "none": "нет",
        "blocks": [("what_happened", "Что произошло"), ("why_it_matters", "Почему это важно"), ("risks", "Риски"), ("opportunities", "Возможности")],
        "house_view": "Позиция института",
        "context": "# КОНТЕКСТ", "answer": "# ОТВЕТ СЕРВИСА", "task": "Оцени ответ строго по JSON-схеме.",
    },
    "en": {
        "headline": "Headline", "impact": "Impact on the subject", "direction": "direction", "caveat": "caveat",
        "sources": "sources", "impact_short": "impact", "horizon": "horizon", "scenarios": "Scenarios",
        "actions": "What can be done", "rationale": "rationale", "none": "none",
        "blocks": [("what_happened", "What happened"), ("why_it_matters", "Why it matters"), ("risks", "Risks"), ("opportunities", "Opportunities")],
        "house_view": "House view",
        "context": "# CONTEXT", "answer": "# SERVICE ANSWER", "task": "Score the answer strictly by the JSON schema.",
    },
}  # fmt: skip


def render_answer(content: InsightContent | dict, lang: str = "ru", inst: dict | None = None) -> str:
    """Any system's answer in one neutral text form for the judge."""
    lang = norm_lang(lang)
    L = _ANSWER[lang]
    inst = inst or {"institution": "Demo Bank", "institution_gen": "Демо-банка"}
    c = content if isinstance(content, dict) else content.model_dump(mode="json")
    lines = [
        f"{L['headline']}: {c.get('headline', '')}",
        f"{L['impact']}: {c.get('overall_impact')}, {L['direction']}: {c.get('overall_direction')}",
    ]

    def claim(x) -> str:
        if isinstance(x, str):
            return f"- {x}"
        label = t(f"origin.{x['origin']}", lang, **inst) if x.get("origin") else ""
        extra = []
        if x.get("impact"):
            extra.append(f"{L['impact_short']} {x['impact']}, {L['horizon']} {x.get('horizon')}")
        if x.get("caveat"):
            extra.append(f"{L['caveat']}: {x['caveat']}")
        if x.get("source_ids"):
            extra.append(f"{L['sources']}: " + ", ".join(x["source_ids"]))
        tail = f" ({'; '.join(extra)})" if extra else ""
        return f"- [{label}] {x['text']}{tail}" if label else f"- {x['text']}{tail}"

    for key, title in L["blocks"]:
        if c.get(key):
            lines += ["", f"{title}:"] + [claim(x) for x in c[key]]
    if c.get("scenarios"):
        lines += ["", f"{L['scenarios']}:"]
        lines += [
            f"- {s['name']}: {s['description']}" if isinstance(s, dict) else f"- {s}" for s in c["scenarios"]
        ]
    hv = c.get("house_view")
    lines += ["", f"{L['house_view']}:"] + ([claim(x) for x in hv] if hv else [f"- {L['none']}"])
    if c.get("actions"):
        lines += ["", f"{L['actions']}:"]
        lines += [
            f"- {a['text']} ({L['rationale']}: {a['rationale']})" if isinstance(a, dict) else f"- {a}"
            for a in c["actions"]
        ]
    return "\n".join(lines)


def judge_messages(
    context: str, answer: str, lang: str = "ru", inst: dict | None = None
) -> list[dict[str, str]]:
    lang = norm_lang(lang)
    L = _ANSWER[lang]
    inst = inst or {"institution": "Demo Bank", "institution_gen": "Демо-банка"}
    user = f"{L['context']}\n{context}\n\n{L['answer']}\n{answer}\n\n{L['task']}"
    return [
        {"role": "system", "content": JUDGE_SYSTEM[lang].format(**inst)},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------- baseline “retelling by one model call”


class BaselineAnswer(BaseModel):
    headline: str
    what_happened: list[str]
    why_it_matters: list[str]
    risks: list[str]
    opportunities: list[str]
    scenarios: list[str]
    house_view: list[str]
    actions: list[str]
    overall_impact: str = Field(description="low, medium or high")
    overall_direction: str = Field(description="negative, positive, mixed or unclear")


BASELINE_SYSTEM = {
    "ru": """Ты — аналитик. Объясни читателю, что значит эта новость для него. Заполни разделы: заголовок, что \
произошло, почему это важно, риски, возможности, сценарии, позиция института, что можно сделать, уровень влияния \
(low/medium/high) и направление (negative/positive/mixed/unclear). Пиши по-русски, кратко, по 1–3 пункта в разделе.""",
    "en": """You are an analyst. Explain to the reader what this news means for them. Fill in the sections: \
headline, what happened, why it matters, risks, opportunities, scenarios, the institution's view, what can be done, \
the impact level (low/medium/high) and the direction (negative/positive/mixed/unclear). Write in English, briefly, \
1–3 points per section.""",
}
_BASELINE_TASK = {"ru": "Ответь строго по JSON-схеме.", "en": "Answer strictly by the JSON schema."}


def baseline_messages(context: str, lang: str = "ru") -> list[dict[str, str]]:
    lang = norm_lang(lang)
    return [
        {"role": "system", "content": BASELINE_SYSTEM[lang]},
        {"role": "user", "content": f"{context}\n\n{_BASELINE_TASK[lang]}"},
    ]
