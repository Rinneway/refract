"""The headline step: rewrite the headline when the review rejected it.

The model sees only statements that already passed the review, plus the findings about the old headline. A short
careful headline beats a truncated conclusion with an ellipsis.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..i18n import norm_lang, t
from ..schemas import InsightContent

VERSION = "headline-v2"
MAX_CHARS = 120


class HeadlineOut(BaseModel):
    headline: str = Field(description=f"One sentence of up to {MAX_CHARS} characters")


SYSTEM = {
    "ru": f"""Ты редактор аналитики {{institution_gen}}. Напиши заголовок инсайта для читателя: одна фраза до \
{MAX_CHARS} символов — что событие значит для этого субъекта.

Правила:
- только то, что есть в утверждениях ниже; ничего не добавляй;
- прогноз подавай как прогноз («институт ожидает…», «может…»), вывод AI и гипотезу — без категоричности;
- числа — только из утверждений, и не больше одного-двух;
- учти замечания к прежнему заголовку и не повторяй их.
Пиши по-русски.""",
    "en": f"""You are an editor of {{institution}} research. Write the insight's headline for the reader: one \
sentence of up to {MAX_CHARS} characters — what the event means for this subject.

Rules:
- only what the statements below say; add nothing;
- present a forecast as a forecast ("expects…", "may…"), an AI inference or a hypothesis without certainty;
- numbers only from the statements, and no more than one or two;
- take the findings about the previous headline into account and do not repeat them.
Write in English.""",
}
BLOCKS = {
    "ru": ("Что произошло", "Почему важно", "Позиция института"),
    "en": ("What happened", "Why it matters", "House view"),
}
USER = {
    "ru": (
        "Утверждения инсайта (прошли проверку):",
        "Прежний заголовок",
        "Замечания к нему",
        "Верни новый заголовок строго по JSON-схеме.",
    ),
    "en": (
        "Statements of the insight (passed the review):",
        "Previous headline",
        "Findings about it",
        "Return the new headline strictly by the JSON schema.",
    ),
}


def build_messages(
    content: InsightContent, reasons: list[str], lang: str = "ru", inst: dict | None = None
) -> list[dict[str, str]]:
    lang = norm_lang(lang)
    inst = inst or {"institution": "Demo Bank", "institution_gen": "Демо-банка"}
    lines = []
    for title, block in zip(
        BLOCKS[lang], (content.what_happened, content.why_it_matters, content.house_view), strict=True
    ):
        for c in block[:3]:
            label = t(f"origin.{c.origin.value}", lang, **inst)
            lines.append(f"- {title} [{label}]: {c.text}")
    head, prev, findings, task = USER[lang]
    user = (
        f"{head}\n"
        + "\n".join(lines)
        + f"\n\n{prev}: {content.headline}\n{findings}:\n"
        + "\n".join(f"- {r}" for r in reasons[:4])
        + f"\n\n{task}"
    )
    return [{"role": "system", "content": SYSTEM[lang].format(**inst)}, {"role": "user", "content": user}]
