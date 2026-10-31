"""Agent tools — an allow-list, read-only, bound to one subject.

The subject is fixed when the `Toolbox` is created and is never an argument, so the agent physically cannot read
another subject's data. Every result returns its sources for citations.
"""

from __future__ import annotations

from typing import Any

from ..config import Institution
from ..knowledge.store import DataStore
from ..schemas import SourceRef, ToolCall, ToolName, ToolResult
from .context import (
    chunk_source,
    event_sources,
    payments_source,
    profile_features,
    profile_source,
    thesis_source,
)

TOOL_SPECS: dict[str, dict[ToolName, str]] = {
    "ru": {
        ToolName.GET_PROFILE: "Профиль субъекта: род деятельности, масштаб, продукты, ВЭД, чувствительности. Аргументов нет.",
        ToolName.GET_COUNTERPARTIES: "Крупнейшие контрагенты по платежам: отрасль, регион, доля потока. args: role ('supplier' | 'customer' | не задан), limit.",
        ToolName.GET_GROUP: "Группа компаний и возможные связи через общих физлиц. Аргументов нет.",
        ToolName.GET_HOUSE_VIEW: "Позиция {institution_gen}: тезисы аналитики с цифрами и цитатами. args: factors (список из словаря), topics (список тем), query (текст), limit.",
        ToolName.SEARCH_RESEARCH: "Поиск по полному тексту аналитики {institution_gen}. args: query (конкретный запрос), limit.",
        ToolName.SEARCH_NEWS: "Поиск связанных новостей в базе событий. args: query, limit.",
    },
    "en": {
        ToolName.GET_PROFILE: "Subject profile: activity, scale, products, foreign trade, sensitivities. No arguments.",
        ToolName.GET_COUNTERPARTIES: "Largest counterparties by payments: industry, region, share of flow. args: role ('supplier' | 'customer' | omitted), limit.",
        ToolName.GET_GROUP: "Group of companies and possible links through shared persons. No arguments.",
        ToolName.GET_HOUSE_VIEW: "{institution} view: research theses with figures and quotes. args: factors (list from the vocabulary), topics (list of topics), query (text), limit.",
        ToolName.SEARCH_RESEARCH: "Full-text search over {institution} research. args: query (a specific query), limit.",
        ToolName.SEARCH_NEWS: "Search related news in the event base. args: query, limit.",
    },
}


def tool_specs(lang: str, institution: Institution) -> str:
    lang = "en" if lang == "en" else "ru"
    names = {"institution": institution.display(lang), "institution_gen": institution.genitive(lang)}
    return "\n".join(f"- {name.value}: {desc.format(**names)}" for name, desc in TOOL_SPECS[lang].items())


def as_str_list(value) -> list[str]:
    """Planner arguments may be a string, a list of strings or a list of objects — normalise to strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    out = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, dict):
            item = item.get("id") or item.get("key") or item.get("name") or next(iter(item.values()), "")
        if str(item).strip():
            out.append(str(item).strip())
    return out


def cap(limit, default: int, maximum: int) -> int:
    try:
        return max(1, min(int(limit), maximum))
    except (TypeError, ValueError):
        return default


class Toolbox:
    def __init__(
        self,
        store: DataStore,
        profile_id: str,
        current_event_id: str | None = None,
        *,
        lang: str = "ru",
        institution: Institution | None = None,
    ) -> None:
        if profile_id not in store.profiles:
            raise KeyError(f"No profile {profile_id}")
        self.store, self.profile_id, self.current_event_id = store, profile_id, current_event_id
        self.lang, self.institution = lang, institution or Institution()

    @property
    def profile(self):
        return self.store.profiles[self.profile_id]

    def call(self, call: ToolCall) -> ToolResult:
        handler = getattr(self, f"_{call.tool.value}")
        try:
            data, sources = handler(**call.args)
        except TypeError as e:  # wrong arguments from the planner
            return ToolResult(call_id=call.call_id, tool=call.tool, ok=False, error=f"arguments: {e}")
        return ToolResult(call_id=call.call_id, tool=call.tool, ok=True, data=data, sources=sources)

    # ---------- tools

    def _get_profile(self) -> tuple[Any, list[SourceRef]]:
        return self.profile, [profile_source(self.lang)]

    def _get_counterparties(self, role: str | None = None, limit: int = 5) -> tuple[Any, list[SourceRef]]:
        cps = [c for c in self.profile.counterparties if role in (None, c.role)]
        cps = sorted(cps, key=lambda c: c.share_of_flow, reverse=True)[: cap(limit, 5, 10)]
        return cps, [payments_source(self.lang)] if cps else []

    def _get_group(self) -> tuple[Any, list[SourceRef]]:
        p = self.profile
        data = {"group_id": p.group_id, "members": p.group_members, "affiliations": p.affiliations}
        return data, [profile_source(self.lang)]

    def _get_house_view(
        self,
        factors: list[str] | None = None,
        topics: list[str] | None = None,
        query: str = "",
        limit: int = 6,
    ) -> tuple[Any, list[SourceRef]]:
        theses = self.store.find_theses(
            as_str_list(factors), as_str_list(topics), str(query or ""), cap(limit, 6, 8)
        )
        return theses, [thesis_source(th, self.institution, self.lang) for th in theses]

    def _search_research(self, query: str = "", limit: int = 3) -> tuple[Any, list[SourceRef]]:
        chunks = self.store.search_chunks(str(query), cap(limit, 3, 3))
        return chunks, [chunk_source(c, self.institution, self.lang) for c in chunks]

    def _search_news(self, query: str = "", limit: int = 3) -> tuple[Any, list[SourceRef]]:
        events = self.store.search_events(str(query), cap(limit, 3, 3), exclude=self.current_event_id)
        return events, [s for e in events for s in event_sources(e)]


def summarize(result: ToolResult, lang: str = "ru") -> str:
    """A short description of a tool result for the trace."""
    en = lang == "en"
    if not result.ok:
        return f"{'error' if en else 'ошибка'}: {result.error}"
    d = result.data
    if result.tool is ToolName.GET_PROFILE:
        label = d.industry.name[:60] if d.industry else d.kind.value
        return f"{label}; {'features' if en else 'признаков'}: {len(profile_features(d, lang))}"
    if isinstance(d, list):
        if not d:
            return "nothing found" if en else "ничего не найдено"
        first = d[0]
        label = (
            getattr(first, "statement", None)
            or getattr(first, "title", None)
            or getattr(first, "doc_title", None)
            or (
                f"{first.counterparty_id} ({first.role}, {first.share_of_flow:.0%})"
                if hasattr(first, "counterparty_id")
                else ""
            )
        )
        return f"{'found' if en else 'найдено'}: {len(d)}; {'first' if en else 'первое'}: {str(label)[:90]}"
    if isinstance(d, dict):
        n_m, n_a = len(d.get("members", [])), len(d.get("affiliations", []))
        return (
            f"group members: {n_m}, hypothetical links: {n_a}"
            if en
            else f"участников группы: {n_m}, связей-гипотез: {n_a}"
        )
    return str(d)[:120]
