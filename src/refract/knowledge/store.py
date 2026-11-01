"""The agent's data store: profiles, news events, verified theses and research chunks.

`DataStore.from_dir()` reads a folder laid out like `examples/demo/`:

    profiles/*.json            Profile (one object or a list per file)
    events/*.json              NewsEvent
    knowledge/theses*.json     Thesis — the institution's verified positions
    knowledge/docs/            research documents (.md, .txt, .pdf) → chunks for search

Replace it with a database-backed class with the same methods when you need to; the agent only uses the methods.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from ..schemas import KnowledgeChunk, NewsEvent, Profile, Thesis
from ..schemas.knowledge import applies_to
from .ingest import read_documents, split_chunks
from .search import BM25, Embedder, HybridIndex

STALE_AFTER_DAYS = 365


def load_models(paths: Iterable[Path], model):
    out = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        out += [model.model_validate(x) for x in (data if isinstance(data, list) else [data])]
    return out


class DataStore:
    def __init__(
        self,
        profiles: list[Profile],
        events: list[NewsEvent],
        theses: list[Thesis],
        chunks: list[KnowledgeChunk] | None = None,
        *,
        embed: Embedder | None = None,
        cache_dir: Path | None = None,
        today: date | None = None,
    ) -> None:
        self.profiles = {p.profile_id: p for p in profiles}
        self.events = {e.event_id: e for e in events}
        self.theses = list(theses)
        self.chunks = chunks or []
        self.today = today
        self._chunk_index = (
            HybridIndex([f"{c.doc_title}\n{c.text}" for c in self.chunks], embed, cache_dir)
            if self.chunks
            else None
        )
        self._thesis_index = BM25([f"{t.statement} {t.quote}" for t in self.theses]) if self.theses else None

    @classmethod
    def from_dir(
        cls,
        root: str | Path,
        profile_ids: Iterable[str] | None = None,
        *,
        embed: Embedder | None = None,
        cache_dir: Path | None = None,
    ) -> DataStore:
        root = Path(root)
        profile_dir = root / "profiles"
        if profile_ids is None:
            profile_files = sorted(profile_dir.glob("*.json"))
        else:
            profile_files = [
                p for c in dict.fromkeys(profile_ids) if (p := profile_dir / f"{c}.json").exists()
            ]
        knowledge = root / "knowledge"
        docs_dir = knowledge / "docs"
        chunks = [c for d in read_documents(docs_dir) for c in split_chunks(d)] if docs_dir.exists() else []
        return cls(
            load_models(profile_files, Profile),
            load_models(sorted((root / "events").glob("*.json")), NewsEvent),
            load_models(sorted(knowledge.glob("theses*.json")), Thesis),
            chunks,
            embed=embed,
            cache_dir=cache_dir,
        )

    # ---------- profiles and events

    def add_profile(self, profile: Profile) -> None:
        self.profiles[profile.profile_id] = profile

    def add_event(self, event: NewsEvent) -> None:
        self.events[event.event_id] = event

    def search_events(self, query: str, k: int = 3, exclude: str | None = None) -> list[NewsEvent]:
        evs = [e for e in self.events.values() if e.event_id != exclude]
        if not evs:
            return []
        index = BM25([f"{e.title} {e.summary}" for e in evs])
        return [evs[i] for i, _ in index.top(query, k)]

    # ---------- research

    def search_chunks(self, query: str, k: int = 3, country: str | None = None) -> list[KnowledgeChunk]:
        if not self._chunk_index:
            return []
        fresh = [
            (i, s)
            for i, s in self._chunk_index.top(query, k * 6)
            if not self.is_stale(self.chunks[i].doc_date) and applies_to(self.chunks[i].countries, country)
        ]
        return [self.chunks[i] for i, _ in fresh[:k]]

    def find_theses(
        self, factors: list[str], topics: list[str], query: str = "", k: int = 6, country: str | None = None
    ) -> list[Thesis]:
        """Theses by matching factors and topics plus text search; fresh documents first, stale ones last.

        With `country`, theses tied to other economies are skipped."""
        text_scores = (
            self._thesis_index.scores(query) if (self._thesis_index and query) else [0.0] * len(self.theses)
        )
        ranked = []
        for t, ts in zip(self.theses, text_scores, strict=True):
            if not applies_to(t.countries, country):
                continue
            score = 2.0 * len(set(t.factors) & set(factors)) + 1.5 * (t.topic in topics) + 0.5 * ts
            if score > 0:
                stale = t.is_stale or self.is_stale(t.doc_date)
                ranked.append((stale, -score, -(t.doc_date.toordinal() if t.doc_date else 0), t))
        ranked.sort(key=lambda x: x[:3])
        return [t for *_, t in ranked[:k]]

    def is_stale(self, d: date | None) -> bool:
        return bool(d and ((self.today or date.today()) - d).days > STALE_AFTER_DAYS)
