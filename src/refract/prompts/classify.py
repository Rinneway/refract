"""Tagging a fresh news item when the event has no factors yet (e.g. a user pasted it)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..i18n import norm_lang
from ..schemas import EventType, NewsEvent
from ..schemas.vocab import factors_prompt, is_known_factor
from ..textutil import untrusted
from . import rule

VERSION = "classify-v5"
MAX_FACTORS = 3


class NewsTags(BaseModel):
    is_business_relevant: bool = Field(
        description="Whether the news matters for business or household decisions"
    )
    event_type: EventType
    macro_factors: list[str] = Field(
        description="Up to three factors from the vocabulary, the main one first: what changed or was decided"
    )
    industry_sections: list[str] = Field(description="Affected NACE/OKVED sections, letters A–U")
    industry_codes: list[str] = Field(
        default_factory=list,
        description="Activities the news names directly: two-digit NACE/OKVED divisions, e.g. 41 or 56; for trade "
        "the three-digit group when the goods are clear, e.g. 46.2",
    )
    regions: list[str]
    countries: list[str] = Field(
        description="ISO 3166-1 alpha-2 codes of the countries the news is about, the domestic one included"
    )
    commodities: list[str]
    importance: float = Field(ge=0, le=1, description="Strength of the event for the economy, 0–1")
    key_facts: list[str] = Field(max_length=3, description="Up to three key facts with numbers and dates")
    language: str = Field(default="ru", description="Language of the news: ru or en")


SYSTEM = {
    "ru": (
        "Ты размечаешь новости для системы бизнес-аналитики. Определи тип события, факторы влияния строго из "
        "словаря, затронутые разделы NACE/ОКВЭД, регионы, страны (коды ISO, например CN, AE, US), сырьё, силу "
        "события, язык новости и до трёх ключевых фактов по тексту. Факторов — не больше трёх, первым — главный "
        "(что изменилось или решено); то, что в новости лишь упомянуто как фон (например, центробанк говорит о "
        "рынке труда в решении по ставке), не добавляй. Виды деятельности (две цифры NACE/ОКВЭД) указывай, только "
        "если новость прямо называет отрасль: 46 — оптовая торговля, 47 — розница, 41 — строительство зданий, "
        "56 — общепит; для торговли, если понятно, какие товары, — группу из трёх цифр: 46.2 — оптовая торговля "
        "сельхозсырьём, 46.3 — продуктами, 46.7 — топливом, металлами, стройматериалами, 47.1 — неспециализированная "
        "розница; общую макроновость не привязывай ко всем отраслям. В странах укажи и ту, о чьей экономике "
        "новость, даже если она внутренняя (решение центробанка — его страна), если её можно понять из текста. "
        "Не додумывай того, чего нет в новости. "
    ),
    "en": (
        "You tag news for a business analytics system. Determine the event type, impact factors strictly from the "
        "vocabulary, affected NACE sections, regions, countries (ISO codes such as CN, AE, US), commodities, the "
        "strength of the event, the language of the news and up to three key facts from the text. At most three "
        "factors, the main one first (what changed or was decided); do not add what the news only mentions as "
        "background (e.g. a central bank talking about the labour market in a rate decision). Give activities "
        "(two-digit NACE divisions) only if the news names the industry directly: 46 — wholesale, 47 — retail, "
        "41 — construction of buildings, 56 — food service; for trade, when the goods are clear, the three-digit "
        "group: 46.2 — wholesale of agricultural raw materials, 46.3 — of food, 46.7 — of fuel, metals and "
        "building materials, 47.1 — non-specialised retail; do not attach a general macro story to every "
        "industry. In countries, include the country whose economy the news is about even when it is domestic news "
        "(a central bank's decision — its country), if the text makes it clear. Do not add anything that is not in "
        "the news. "
    ),
}
VOCAB = {"ru": "\n\nСловарь факторов:\n", "en": "\n\nFactor vocabulary:\n"}
PUBLISHER = {"ru": "Издание", "en": "Publisher"}


def build_messages(event: NewsEvent, lang: str = "ru") -> list[dict[str, str]]:
    lang = norm_lang(lang)
    system = SYSTEM[lang] + rule(lang) + VOCAB[lang] + factors_prompt(lang)
    a = event.articles[0]
    text = a.text or event.summary
    user = f"{PUBLISHER[lang]}: {a.publisher}\n{untrusted(event.title + chr(10) + text[:3000])}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_ACTIVITY = re.compile(r"\d{2}(?:\.\d{1,2})?")  # a division (46) or, in trade, a group (46.2, 46.21)


def apply(event: NewsEvent, tags: NewsTags) -> NewsEvent:
    return event.model_copy(
        update={
            "event_type": tags.event_type,
            "macro_factors": [f for f in tags.macro_factors if is_known_factor(f)][:MAX_FACTORS],
            "industry_sections": [s.strip().upper()[:1] for s in tags.industry_sections if s.strip()],
            "industry_codes": [d.strip() for d in tags.industry_codes if _ACTIVITY.fullmatch(d.strip())],
            "regions": tags.regions,
            "countries": tags.countries,
            "commodities": tags.commodities,
            "importance": tags.importance,
            "language": "en" if tags.language.lower().startswith("en") else "ru",
        }
    )
