"""Tagging a fresh news item when the event has no factors yet (e.g. a user pasted it)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..i18n import norm_lang
from ..schemas import EventType, NewsEvent
from ..schemas.vocab import factors_prompt, is_known_factor
from ..textutil import untrusted
from . import rule

VERSION = "classify-v3"
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
        description="Activities the news names directly: two-digit NACE/OKVED divisions, e.g. 46 or 47",
    )
    regions: list[str]
    countries: list[str] = Field(description="ISO 3166-1 alpha-2 codes of countries involved")
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
        "56 — общепит; общую макроновость не привязывай ко всем отраслям. Не додумывай того, чего нет в новости. "
    ),
    "en": (
        "You tag news for a business analytics system. Determine the event type, impact factors strictly from the "
        "vocabulary, affected NACE sections, regions, countries (ISO codes such as CN, AE, US), commodities, the "
        "strength of the event, the language of the news and up to three key facts from the text. At most three "
        "factors, the main one first (what changed or was decided); do not add what the news only mentions as "
        "background (e.g. a central bank talking about the labour market in a rate decision). Give activities "
        "(two-digit NACE divisions) only if the news names the industry directly: 46 — wholesale, 47 — retail, "
        "41 — construction of buildings, 56 — food service; do not attach a general macro story to every "
        "industry. Do not add anything that is not in the news. "
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


def apply(event: NewsEvent, tags: NewsTags) -> NewsEvent:
    return event.model_copy(
        update={
            "event_type": tags.event_type,
            "macro_factors": [f for f in tags.macro_factors if is_known_factor(f)][:MAX_FACTORS],
            "industry_sections": [s.strip().upper()[:1] for s in tags.industry_sections if s.strip()],
            "industry_codes": [d for d in tags.industry_codes if d.isdigit() and len(d) == 2],
            "regions": tags.regions,
            "countries": tags.countries,
            "commodities": tags.commodities,
            "importance": tags.importance,
            "language": "en" if tags.language.lower().startswith("en") else "ru",
        }
    )
