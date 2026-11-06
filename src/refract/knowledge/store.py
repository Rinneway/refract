"""The agent's data store: profiles, news events, verified theses and research chunks.

`DataStore.from_dir()` reads a folder laid out like `examples/demo/`:

    profiles/*.json            Profile (one object or a list per file)
    events/*.json              NewsEvent
    knowledge/theses*.json     Thesis — the institution's verified positions
    knowledge/docs/            research documents (.md, .txt, .pdf) → chunks for search

Replace it with a database-backed class with the same methods when you need to; the agent only uses the methods.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from ..schemas import KnowledgeChunk, NewsEvent, Profile, Thesis
from ..schemas.knowledge import applies_to
from ..textutil import DataError, read_json
from .ingest import read_documents, split_chunks
from .search import BM25, Embedder, Embeddings, HybridIndex, cosine

STALE_AFTER_DAYS = 365


def load_models(paths: Iterable[Path], model, problems: list[str] | None = None) -> list:
    """Objects from JSON files (one object or a list per file). A file or an object that does not load is
    skipped with a note in `problems` naming the file — one typo must not take the whole folder down. Without
    `problems` the first error is raised."""
    out = []
    for p in paths:
        try:
            data = read_json(p)
        except DataError as e:
            if problems is None:
                raise
            problems.append(str(e))
            continue
        for i, x in enumerate(data if isinstance(data, list) else [data]):
            try:
                out.append(model.model_validate(x))
            except ValidationError as e:
                where = (
                    (x.get("profile_id") or x.get("event_id") or x.get("thesis_id"))
                    if isinstance(x, dict)
                    else None
                )
                what = "; ".join(
                    f"{'.'.join(map(str, err['loc'])) or '(object)'}: {err['msg']}" for err in e.errors()[:3]
                )
                note = f"{p}{f' [{where}]' if where else f' item {i}' if isinstance(data, list) else ''}: skipped — {what}"
                if problems is None:
                    raise DataError(note) from None
                problems.append(note)
    return out


def _unique(items: list, key: str, kind: str, problems: list[str]) -> list:
    """The last object of an id wins, as before; a repeated id is reported."""
    seen: dict[str, int] = {}
    for item in items:
        ident = getattr(item, key)
        seen[ident] = seen.get(ident, 0) + 1
    problems += [
        f"{kind} {ident} appears {n} times; the last one is used" for ident, n in seen.items() if n > 1
    ]
    return items


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
        embed_model: str = "",
    ) -> None:
        """With `embed`, research, news and theses are searched by BM25 and embeddings together (vectors cached
        per text and model in `cache_dir`); if the embedding model fails, search stays lexical and says so in
        `problems`."""
        self.problems: list[
            str
        ] = []  # files and objects `from_dir` skipped, each naming the file and the reason
        self.profiles = {p.profile_id: p for p in profiles}
        self.events = {e.event_id: e for e in events}
        self.theses = list(theses)
        self.chunks = chunks or []
        self.today = today
        self.embeddings = Embeddings(embed, embed_model, cache_dir) if embed else None
        self._chunk_index = (
            HybridIndex([f"{c.doc_title}\n{c.text}" for c in self.chunks], self.embeddings)
            if self.chunks
            else None
        )
        self._thesis_texts = [f"{t.statement} {t.quote}" for t in self.theses]
        self._thesis_index = BM25(self._thesis_texts) if self.theses else None
        self._note_embedding_error()

    @classmethod
    def from_dir(
        cls,
        root: str | Path,
        profile_ids: Iterable[str] | None = None,
        *,
        embed: Embedder | None = None,
        cache_dir: Path | None = None,
        embed_model: str = "",
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
        problems: list[str] = []
        docs = read_documents(docs_dir, problems) if docs_dir.exists() else []
        store = cls(
            _unique(load_models(profile_files, Profile, problems), "profile_id", "profile", problems),
            _unique(
                load_models(sorted((root / "events").glob("*.json")), NewsEvent, problems),
                "event_id",
                "event",
                problems,
            ),
            load_models(sorted(knowledge.glob("theses*.json")), Thesis, problems),
            [c for d in docs for c in split_chunks(d)],
            embed=embed,
            cache_dir=cache_dir,
            embed_model=embed_model,
        )
        store.problems = problems + store.problems
        return store

    def _note_embedding_error(self) -> None:
        error = self.embeddings.error if self.embeddings else None
        if error and error not in self.problems:
            self.problems.append(error)

    # ---------- profiles and events

    def add_profile(self, profile: Profile) -> None:
        self.profiles[profile.profile_id] = profile

    def add_event(self, event: NewsEvent) -> None:
        self.events[event.event_id] = event

    def search_events(self, query: str, k: int = 3, exclude: str | None = None) -> list[NewsEvent]:
        evs = [e for e in self.events.values() if e.event_id != exclude]
        if not evs:
            return []
        index = HybridIndex([f"{e.title} {e.summary}" for e in evs], self.embeddings)
        found = [evs[i] for i, _ in index.top(query, k)]
        self._note_embedding_error()
        return found

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
        # meaning, not just words: a thesis in Russian for an English query (embeddings, if configured)
        dense = [0.0] * len(self.theses)
        if self.embeddings and query and self.theses:
            vectors, qv = self.embeddings.vectors(self._thesis_texts), self.embeddings.query(query)
            if vectors and qv:
                dense = [max(0.0, cosine(qv, v)) for v in vectors]
            self._note_embedding_error()
        ranked = []
        for t, ts, ds in zip(self.theses, text_scores, dense, strict=True):
            if not applies_to(t.countries, country):
                continue
            score = 2.0 * len(set(t.factors) & set(factors)) + 1.5 * (t.topic in topics) + 0.5 * ts + 1.0 * ds
            if score > 0:
                stale = t.is_stale or self.is_stale(t.doc_date)
                ranked.append((stale, -score, -(t.doc_date.toordinal() if t.doc_date else 0), t))
        ranked.sort(key=lambda x: x[:3])
        return [t for *_, t in ranked[:k]]

    def is_stale(self, d: date | None) -> bool:
        return bool(d and ((self.today or date.today()) - d).days > STALE_AFTER_DAYS)
