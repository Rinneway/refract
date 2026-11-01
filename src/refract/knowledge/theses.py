"""Extract the institution's theses from research documents.

For every page (or Markdown section) the model extracts positions — forecasts, assessments, expectations — with
numbers and an exact quote. Code keeps a thesis only if the quote really is on the page (≥ 90% match), drops
numbers absent from the page and marks theses from documents older than a year as stale. Answers are cached, so a
repeated run is free.
"""

from __future__ import annotations

import datetime as dt
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

from ..i18n import norm_lang
from ..llm.client import LLM
from ..schemas import Figure, Thesis
from ..schemas.vocab import THESIS_TOPICS, factors_prompt, is_known_factor
from ..textutil import numbers, quote_grounding
from .ingest import Document, read_documents

VERSION = "theses-v2"
MIN_GROUNDING = 0.9
STALE_AFTER_DAYS = 365


class ThesisDraft(BaseModel):
    topic: str = Field(description="Topic from the topic list")
    statement: str = Field(description="The position in your own words, one or two sentences")
    figures: list[Figure] = Field(default_factory=list)
    horizon: str | None = None
    factors: list[str] = Field(default_factory=list)
    quote: str = Field(description="An exact quote from the text, one or two sentences")


class ThesesOut(BaseModel):
    theses: list[ThesisDraft] = Field(max_length=8)


SYSTEM = {
    "ru": (
        "Ты извлекаешь из аналитического документа тезисы — позицию автора: прогнозы, оценки, ожидания и выводы. "
        "Статистику без оценки не включай. Для каждого тезиса: тема из списка, суть своими словами (не усиливай "
        "формулировку: «ожидает» не равно «так будет»), числа (значение, единица, период) только из текста, "
        "горизонт, факторы из словаря и ДОСЛОВНАЯ цитата из текста, которая подтверждает тезис. Не более 6 тезисов; "
        "если в тексте нет выводов — верни пустой список. Пиши по-русски.\n\nТемы:\n{topics}\n\nФакторы:\n{factors}"
    ),
    "en": (
        "You extract theses from a research document — the author's position: forecasts, assessments, expectations "
        "and conclusions. Leave out statistics without an assessment. For each thesis: a topic from the list, the "
        "gist in your own words (do not strengthen it: “expects” is not “will”), numbers (value, unit, period) only "
        "from the text, the horizon, factors from the vocabulary and an EXACT quote from the text that supports the "
        "thesis. At most 6 theses; if the text has no conclusions, return an empty list. Write in English."
        "\n\nTopics:\n{topics}\n\nFactors:\n{factors}"
    ),
}


def page_messages(text: str, page: int, title: str, lang: str) -> list[dict[str, str]]:
    lang = norm_lang(lang)
    topics = "\n".join(f"- {k} — {v[lang]}" for k, v in THESIS_TOPICS.items())
    system = SYSTEM[lang].format(topics=topics, factors=factors_prompt(lang))
    head = "Документ" if lang == "ru" else "Document"
    part = "Страница/раздел" if lang == "ru" else "Page/section"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{head}: {title}\n{part} {page}:\n{text}"},
    ]


def to_theses(
    doc: Document, page: int, text: str, drafts: list[ThesisDraft], today: dt.date | None = None
) -> list[Thesis]:
    page_nums = numbers(text)
    today = today or dt.date.today()
    stale = bool(doc.doc_date and doc.doc_date < today - dt.timedelta(days=STALE_AFTER_DAYS))
    out: list[Thesis] = []
    for i, d in enumerate(drafts, 1):
        if quote_grounding(d.quote, text) < MIN_GROUNDING:
            continue
        figures = [
            f
            for f in d.figures
            if not isinstance(f.value, (int, float)) or round(float(f.value), 3) in page_nums
        ]
        out.append(
            Thesis(
                thesis_id=f"{doc.doc_id}_p{page}_{i}",
                topic=d.topic if d.topic in THESIS_TOPICS else "industry",
                statement=d.statement,
                figures=figures,
                horizon=d.horizon,
                factors=[f for f in d.factors if is_known_factor(f)],
                doc_id=doc.doc_id,
                doc_title=doc.title,
                doc_date=doc.doc_date,
                page=page,
                quote=d.quote,
                language=doc.language,
                countries=doc.countries,
                is_stale=stale,
            )
        )
    return out


def extract_theses(llm: LLM, docs_dir: Path, *, workers: int = 4) -> tuple[list[Thesis], dict]:
    """All theses from a folder of documents, plus statistics: pages, drafts, kept (grounded), failed, cost."""
    docs = read_documents(docs_dir)
    r = llm.settings.route("extract_theses")
    jobs = [(d, page, text) for d in docs for page, text in d.pages if len(text.strip()) > 200]
    stats = {"documents": len(docs), "pages": len(jobs), "drafts": 0, "kept": 0, "failed": 0, "cost": 0.0}
    lock = threading.Lock()

    def one(job) -> list[Thesis]:
        doc, page, text = job
        try:
            res = llm.chat_json(
                r.model,
                page_messages(text, page, doc.title, doc.language),
                ThesesOut,
                max_tokens=r.max_tokens,
                step="extract_theses",
            )
        except Exception:  # noqa: BLE001 — the page is retried on the next run
            with lock:
                stats["failed"] += 1
            return []
        drafts = res.value.theses if res.value else []
        with lock:
            stats["cost"] += res.cost
            stats["drafts"] += len(drafts)
        return to_theses(doc, page, text, drafts)

    theses: list[Thesis] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for got in pool.map(one, jobs):
            theses.extend(got)
    stats["kept"] = len(theses)
    return theses, stats
