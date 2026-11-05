"""MCP server: Refract as tools for Claude Desktop, Cursor and other MCP clients.

Install with `pip install 'refract-agent[mcp]'` and register the command `refract mcp` (stdio transport):

    {"mcpServers": {"refract": {"command": "refract", "args": ["mcp", "--data", "/path/to/data"],
                                "env": {"LLM_API_KEY": "…", "LLM_BASE_URL": "…", "REFRACT_MODEL": "…"}}}}

Tools: list_profiles, get_profile, news_feed, list_events, analyze_news, ask_followup, get_trace, search_research.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .facade import Refract
from .render import chat_markdown, feed_markdown, insight_markdown, trace_markdown


def build_server(rf: Refract):
    try:  # mcp ≥ 2
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server

    server = Server(
        "refract",
        instructions=(
            "Refract explains what a news item means for a specific subject (a company, a sole proprietor, a "
            "self-employed person or a private individual). Call list_profiles first, then analyze_news with a "
            "profile_id and either an event_id or the news title and text. Every statement in the result is tagged "
            "as a fact, the institution's view, an AI inference or a hypothesis, with its sources."
        ),
    )

    @server.tool()
    def list_profiles() -> str:
        """Subjects available for analysis: id, kind and a short summary."""
        rows = [
            f"- `{p.profile_id}` ({p.kind.value}) — {p.name or ''}: {p.summary[:160]}"
            for p in rf.store.profiles.values()
        ]
        return "\n".join(rows) or "No profiles in the store."

    @server.tool()
    def get_profile(profile_id: str) -> str:
        """The full profile of a subject as JSON."""
        p = rf.store.profiles.get(profile_id)
        return p.model_dump_json(indent=1) if p else f"Unknown profile {profile_id}"

    @server.tool()
    def news_feed(profile_id: str, top: int = 10, since: str | None = None) -> str:
        """Which stored news concern the subject, strongest links first — ranked by code, with the reason for each.
        since: YYYY-MM-DD. Use analyze_news on an event_id from the list to get the full insight."""
        if profile_id not in rf.store.profiles:
            return f"Unknown profile {profile_id}; call list_profiles."
        try:
            start = datetime.fromisoformat(since) if since else None
        except ValueError:
            return "since must be YYYY-MM-DD."
        return feed_markdown(rf.feed(profile_id, top=max(1, min(top, 50)), since=start))

    @server.tool()
    def list_events(limit: int = 20) -> str:
        """Latest news events in the store."""
        evs = sorted(rf.store.events.values(), key=lambda e: e.published_at, reverse=True)[:limit]
        return (
            "\n".join(f"- `{e.event_id}` {e.published_at:%Y-%m-%d} [{e.language}] {e.title}" for e in evs)
            or "No events."
        )

    @server.tool()
    def analyze_news(
        profile_id: str,
        event_id: str | None = None,
        title: str | None = None,
        text: str | None = None,
        publisher: str = "user",
        url: str | None = None,
        role: str = "cfo",
        language: str = "ru",
    ) -> str:
        """Run the agent: what the news means for the subject. Give an event_id, or a title and text.
        role: owner, ceo, cfo, chief_accountant, treasurer, entrepreneur, individual, other. language: ru or en."""
        if profile_id not in rf.store.profiles:
            return f"Unknown profile {profile_id}; call list_profiles."
        if event_id:
            if event_id not in rf.store.events:
                return f"Unknown event {event_id}; call list_events."
            event = rf.store.events[event_id]
        elif title and text:
            event = rf.news_from_text(title, text, publisher=publisher, url=url)
        else:
            return "Give an event_id, or both title and text."
        run = rf.analyze(profile_id, event, role=role, language=language)
        if not run.insight:
            return "The agent produced no insight:\n\n" + trace_markdown(run.trace)
        return f"insight_id: `{run.insight.insight_id}`\n\n" + insight_markdown(
            run.insight, rf.settings.institution
        )

    @server.tool()
    def ask_followup(insight_id: str, question: str) -> str:
        """A follow-up question about an insight produced in this session."""
        if insight_id not in rf.runs:
            return f"Unknown insight {insight_id}."
        return chat_markdown(rf.chat(insight_id, question).response, rf.settings.institution)

    @server.tool()
    def get_trace(insight_id: str) -> str:
        """How an insight was produced: every tool call and model call with timing, cost and review findings."""
        run = rf.runs.get(insight_id)
        return trace_markdown(run.trace) if run else f"Unknown insight {insight_id}."

    @server.tool()
    def search_research(query: str, limit: int = 3) -> str:
        """Search the institution's research documents."""
        chunks = rf.store.search_chunks(query, k=max(1, min(limit, 5)))
        return json.dumps([c.model_dump(mode="json") for c in chunks], ensure_ascii=False, indent=1) or "[]"

    return server


def run(data_dir: str | Path | None = None) -> None:  # pragma: no cover
    build_server(Refract.from_dir(data_dir)).run()
