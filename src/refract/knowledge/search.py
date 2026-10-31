"""Retrieval over research documents and news: BM25 for Russian and English, optional dense retrieval.

BM25 needs nothing but the standard library. If an embedding model is configured, `HybridIndex` fuses BM25 and
cosine similarity by reciprocal rank fusion; embeddings are cached on disk so a corpus is embedded once.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

WORD = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)

STOP_RU = set(
    [
        "и",
        "в",
        "во",
        "не",
        "что",
        "он",
        "на",
        "я",
        "с",
        "со",
        "как",
        "а",
        "то",
        "все",
        "она",
        "так",
        "его",
        "но",
        "да",
        "ты",
        "к",
        "у",
        "же",
        "вы",
        "за",
        "бы",
        "по",
        "только",
        "ее",
        "мне",
        "было",
        "вот",
        "от",
        "меня",
        "еще",
        "нет",
        "о",
        "из",
        "ему",
        "теперь",
        "когда",
        "даже",
        "ну",
        "ли",
        "если",
        "уже",
        "или",
        "ни",
        "быть",
        "был",
        "него",
        "до",
        "вас",
        "нибудь",
        "опять",
        "уж",
        "вам",
        "ведь",
        "там",
        "потом",
        "себя",
        "ничего",
        "ей",
        "может",
        "они",
        "тут",
        "где",
        "есть",
        "надо",
        "ней",
        "для",
        "мы",
        "тебя",
        "их",
        "чем",
        "была",
        "сам",
        "чтобы",
        "без",
        "будто",
        "чего",
        "раз",
        "тоже",
        "себе",
        "под",
        "будет",
        "ж",
        "тогда",
        "кто",
        "этот",
        "того",
        "потому",
        "этого",
        "какой",
        "совсем",
        "ним",
        "здесь",
        "этом",
        "один",
        "почти",
        "мой",
        "тем",
        "нее",
        "сейчас",
        "были",
        "куда",
        "зачем",
        "всех",
        "никогда",
        "можно",
        "при",
        "наконец",
        "два",
        "об",
        "другой",
        "хоть",
        "после",
        "над",
        "больше",
        "тот",
        "через",
        "эти",
        "нас",
        "про",
        "всего",
        "них",
        "какая",
        "много",
        "разве",
        "три",
        "эту",
        "моя",
        "впрочем",
        "хорошо",
        "свою",
        "этой",
        "перед",
        "иногда",
        "лучше",
        "чуть",
        "том",
        "нельзя",
        "такой",
        "им",
        "более",
        "всегда",
        "конечно",
        "всю",
        "между",
        "это",
        "также",
        "г",
        "гг",
        "год",
        "года",
    ]
)
STOP_EN = set(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "his",
        "her",
        "their",
        "our",
        "your",
        "not",
        "no",
        "so",
        "than",
        "then",
        "there",
        "here",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "too",
        "very",
        "can",
        "will",
        "just",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "having",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "about",
        "against",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "also",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "per",
        "year",
        "years",
    ]
)
STOP = STOP_RU | STOP_EN


def tokens(text: str) -> list[str]:
    """Lower-cased words without stop words, crudely stemmed to the first six letters."""
    out = []
    for w in WORD.findall((text or "").lower().replace("ё", "е")):
        if w in STOP or len(w) < 2:
            continue
        out.append(w[:6] if len(w) > 6 and not w.isdigit() else w)
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens(d)) for d in docs]
        self.lens = [sum(c.values()) for c in self.docs]
        self.avg = sum(self.lens) / len(self.lens) if self.lens else 0.0
        df: Counter[str] = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for c, length in zip(self.docs, self.lens, strict=True):
            s = 0.0
            for t in q:
                tf = c.get(t, 0)
                if tf:
                    s += (
                        self.idf[t]
                        * tf
                        * (self.k1 + 1)
                        / (tf + self.k1 * (1 - self.b + self.b * length / (self.avg or 1)))
                    )
            out.append(s)
        return out

    def top(self, query: str, k: int = 5, min_score: float = 0.0) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > min_score]


Embedder = Callable[[list[str]], list[list[float]]]


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0


class HybridIndex:
    """BM25 + embeddings fused by reciprocal rank fusion (k = 60). Without an embedder it is plain BM25."""

    def __init__(
        self, docs: list[str], embed: Embedder | None = None, cache_dir: Path | None = None, rrf_k: int = 60
    ):
        self.docs, self.bm25, self.embed, self.rrf_k = docs, BM25(docs), embed, rrf_k
        self.cache_dir = cache_dir
        self.vectors: list[list[float]] | None = self._embed_docs() if embed else None

    def _embed_docs(self) -> list[list[float]]:
        key = hashlib.sha256("\n\x00".join(self.docs).encode("utf-8")).hexdigest()[:24]
        path = self.cache_dir / f"emb_{key}.json" if self.cache_dir else None
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        vectors: list[list[float]] = []
        for i in range(0, len(self.docs), 64):
            vectors += self.embed(self.docs[i : i + 64])  # type: ignore[misc]
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(vectors), encoding="utf-8")
        return vectors

    def top(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        lexical = self.bm25.top(query, k=max(k * 4, 20))
        if not self.vectors or not self.embed:
            return lexical[:k]
        qv = self.embed([query])[0]
        dense = sorted(((i, _cosine(qv, v)) for i, v in enumerate(self.vectors)), key=lambda x: -x[1])[
            : max(k * 4, 20)
        ]
        fused: dict[int, float] = {}
        for ranking in (lexical, dense):
            for rank, (i, _) in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]
