"""The plan step: which data to gather to explain the event for this subject."""

from __future__ import annotations

from ..agent.context import profile_features
from ..agent.tools import tool_specs
from ..config import Institution
from ..i18n import norm_lang
from ..schemas import NewsEvent, Profile
from ..schemas.vocab import factors_prompt, topics_prompt
from ..textutil import untrusted
from . import rule

VERSION = "plan-v3"

SYSTEM = {
    "ru": """Ты планируешь работу аналитика {institution_gen}, который объясняет читателю — руководителю компании, \
предпринимателю или частному лицу, — что значит для него событие. Составь план из 3–6 вызовов инструментов, чтобы \
собрать данные для обоснованного вывода.

Инструменты (только они):
{tools}

Правила:
- Профиль уже загружен — get_profile не вызывай.
- Позиция института обязательна: вызови get_house_view с факторами события и темами; если тезисов мало, добавь \
search_research с конкретным запросом (например, «прогноз ключевой ставки конец года»).
- Если событие касается отрасли, сырья, поставок или спроса и у субъекта есть контрагенты — вызови \
get_counterparties: важны отрасли поставщиков и покупателей.
- search_news — только если для понимания события нужен контекст других новостей.
- В purpose одной фразой объясни, зачем шаг. step_id: s1, s2…
- rationale — как событие может быть связано с субъектом, одно-два предложения.

Темы тезисов: {topics}
Словарь факторов:
{factors}

{untrusted}""",
    "en": """You plan the work of an analyst at {institution} who explains to a reader — a company executive, an \
entrepreneur or a private individual — what an event means for them. Make a plan of 3–6 tool calls that gathers \
the data for a well-grounded conclusion.

Tools (only these):
{tools}

Rules:
- The profile is already loaded — do not call get_profile.
- The house view is mandatory: call get_house_view with the event's factors and topics; if there are few theses, \
add search_research with a specific query (e.g. "policy rate forecast end of year").
- If the event concerns an industry, commodities, supply or demand and the subject has counterparties, call \
get_counterparties: the industries of suppliers and customers matter.
- search_news — only if other news is needed to understand the event.
- In purpose explain in one sentence why the step is needed. step_id: s1, s2…
- rationale — how the event may be linked to the subject, one or two sentences.

Thesis topics: {topics}
Factor vocabulary:
{factors}

{untrusted}""",
}
USER = {
    "ru": "Субъект: {summary}\nПризнаки:\n{features}\n\nСобытие ({date}):\n{news}\nТип: {type}; факторы: {factors}",
    "en": "Subject: {summary}\nFeatures:\n{features}\n\nEvent ({date}):\n{news}\nType: {type}; factors: {factors}",
}
NO_FACTORS = {"ru": "не размечены", "en": "not tagged"}


def build_messages(
    profile: Profile, event: NewsEvent, lang: str = "ru", institution: Institution | None = None
) -> list[dict[str, str]]:
    lang, inst = norm_lang(lang), institution or Institution()
    system = SYSTEM[lang].format(
        institution=inst.display(lang),
        institution_gen=inst.genitive(lang),
        tools=tool_specs(lang, inst),
        topics=topics_prompt(lang),
        factors=factors_prompt(lang),
        untrusted=rule(lang),
    )
    feats = "\n".join(f"- {k} — {v}" for k, v in profile_features(profile, lang, inst))
    user = USER[lang].format(
        summary=profile.summary,
        features=feats,
        date=f"{event.published_at:%Y-%m-%d}",
        news=untrusted(event.title + "\n" + event.summary[:800]),
        type=event.event_type.value,
        factors=", ".join(event.macro_factors) or NO_FACTORS[lang],
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
