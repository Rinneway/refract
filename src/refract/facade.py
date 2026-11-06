"""`Refract` — the high-level entry point used by the CLI, the REST API and the MCP server.

rf = Refract.from_dir("examples/demo")
event = rf.news_from_text("ЦБ сохранил ставку", "Совет директоров…", publisher="Пресс-служба")
run = rf.analyze("demo_importer", event, role="cfo", language="ru")
answer = rf.chat(run.insight.insight_id, "А что с поставщиками?")
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import Agent, AgentConfig, AgentRun, ChatAgent, ChatRun
from .agent.feed import rank as rank_feed
from .agent.feed import tag_event
from .agent.impact import ImpactPolicy
from .agent.policy import DEFAULT_POLICY, Policy
from .config import Settings, load_settings
from .i18n import norm_lang
from .knowledge.store import DataStore
from .llm.client import LLM, LLMClient
from .recent import Recent
from .schemas import (
    AgentState,
    Brief,
    BriefItem,
    ChatTurn,
    Feed,
    Insight,
    NewsArticle,
    NewsEvent,
    Profile,
    Trace,
)
from .textutil import read_json


def _hash(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()[:12]


def news_from_text(
    title: str,
    text: str,
    *,
    publisher: str = "user",
    url: str | None = None,
    published_at: datetime | None = None,
    language: str | None = None,
) -> NewsEvent:
    """A news event from a pasted text. Factors are empty, so the agent tags it with the model first."""
    h = _hash(title, text)
    when = published_at or datetime.now(timezone.utc)
    lang = (
        norm_lang(language)
        if language
        else ("ru" if sum("а" <= c.lower() <= "я" for c in title + text) > len(title + text) / 4 else "en")
    )
    return NewsEvent(
        event_id=f"ev_{h}",
        title=title,
        summary=text[:600],
        published_at=when,
        articles=[
            NewsArticle(
                article_id=f"art_{h}",
                url=url or f"user://{h}",
                publisher=publisher,
                title=title,
                published_at=when,
                text=text,
                language=lang,
            )
        ],
        language=lang,
    )


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


class Refract:
    def __init__(
        self,
        store: DataStore,
        *,
        llm: LLM | None = None,
        settings: Settings | None = None,
        config: AgentConfig | None = None,
        policy: Policy | None = None,
        impact_policy: ImpactPolicy | None = None,
        keep_runs: int = 500,
        replay=None,
    ) -> None:
        """`keep_runs` — how many recent runs stay in memory for chat and the API; older ones are forgotten, so a
        long-running server does not grow without bound (save runs with `save_run` to keep them)."""
        self.store = store
        self.settings = settings or getattr(llm, "settings", None) or load_settings()
        self.config = config or AgentConfig(language=self.settings.language)
        self.policy = policy or (
            Policy(self.settings.roles_file, self.settings.actions_file)
            if (self.settings.roles_file or self.settings.actions_file)
            else DEFAULT_POLICY
        )
        self.impact_policy = impact_policy
        self._llm = llm
        self._lock = threading.Lock()
        self.runs: Recent[str, AgentRun] = Recent(keep_runs)
        self.replay = replay  # demo mode (`Refract.demo()`): model answers come from recordings

    @classmethod
    def demo(cls, settings: Settings | None = None, **kwargs: Any) -> Refract:
        """The agent without a model: the packaged demo data, and every model call answered from a recording of a
        real run or a scripted scenario (`demo.Replay`). Checks, the impact ceiling and the policy are live code.
        Pairs without a recording, pasted texts and follow-up questions raise `DemoError`."""
        from .demo import DEMO_DIR, DEMO_NOW, Replay

        store = DataStore.from_dir(DEMO_DIR)
        store.today = DEMO_NOW.date()
        settings = replace(settings or load_settings(), api_key=None, embedding_model=None)
        return cls(store, settings=settings, replay=Replay(), **kwargs)

    @property
    def is_demo(self) -> bool:
        return self.replay is not None

    @classmethod
    def from_dir(cls, path: str | Path | None = None, **kwargs: Any) -> Refract:
        settings = kwargs.pop("settings", None) or (
            kwargs["llm"].settings if kwargs.get("llm") else load_settings()
        )
        root = Path(path or settings.data_dir)
        if not root.exists() and path is None:  # installed from a wheel: fall back to the packaged demo data
            from .demo import DEMO_DIR

            root = DEMO_DIR
        embed = None
        if settings.embedding_model and settings.api_key and not kwargs.get("llm"):
            client = LLMClient(settings)
            embed = client.embed
            kwargs.setdefault("llm", client)
        store = DataStore.from_dir(
            root,
            embed=embed,
            cache_dir=Path(settings.cache_dir) / "embeddings",
            embed_model=settings.embedding_model or "",
        )
        return cls(store, settings=settings, **kwargs)

    @property
    def llm(self) -> LLM:
        with self._lock:
            if self._llm is None:
                self._llm = LLMClient(self.settings)
            return self._llm

    # ------------------------------------------------------------------ agent

    def agent(self, config: AgentConfig | None = None, on_progress=None, llm: LLM | None = None) -> Agent:
        extra = {}
        if self.replay is not None:
            from .demo import DEMO_NOW

            extra["now"] = lambda: DEMO_NOW
        return Agent(
            llm or self.llm,
            self.store,
            **extra,
            config=config or self.config,
            settings=self.settings,
            policy=self.policy,
            impact_policy=self.impact_policy,
            on_progress=on_progress,
        )

    def analyze(
        self,
        profile: str | Profile,
        event: str | NewsEvent | dict,
        *,
        role: str = "cfo",
        language: str | None = None,
        config: AgentConfig | None = None,
        on_progress=None,
    ) -> AgentRun:
        """Run the agent for one subject × event pair. `event` may be an id, a `NewsEvent` or
        `{"title": …, "text": …, "publisher": …, "url": …}`."""
        if isinstance(profile, Profile):
            self.store.add_profile(profile)
            profile = profile.profile_id
        if isinstance(event, dict):
            event = news_from_text(**event)
        lang = norm_lang(language or (config or self.config).language)
        llm, rec = None, None
        if self.replay is not None:
            from .demo import DemoError

            if isinstance(event, NewsEvent) and self.store.events.get(event.event_id) == event:
                event = event.event_id
            if not isinstance(event, str):
                raise DemoError("demo mode replays stored demo events only; a pasted news text needs a model")
            rec = self.replay.find(profile, event, role, lang)
            role, lang, llm = rec.role, rec.language, self.replay.llm(rec, self.settings)
        cfg = replace(config or self.config, language=lang)
        run = self.agent(cfg, on_progress, llm).run(profile, event, role=role, language=lang)
        if rec is not None:
            run.trace.config["replay"] = rec.key
        if run.insight:
            self.runs[run.insight.insight_id] = run
        return run

    def chat(
        self, insight_id: str, question: str, history: list[ChatTurn] | None = None, on_progress=None
    ) -> ChatRun:
        if self.replay is not None:
            from .demo import DemoError

            raise DemoError("demo mode has no recorded answers to follow-up questions; chat needs a model")
        run = self.runs[insight_id]
        agent = self.agent(on_progress=on_progress)
        return ChatAgent(self.llm, self.store, agent=agent).answer(run.state, run.insight, question, history)

    def feed(
        self,
        profile_id: str,
        *,
        top: int | None = 10,
        since: datetime | None = None,
        language: str | None = None,
        tag: bool = True,
        include_unrelated: bool = False,
    ) -> Feed:
        """Stored events ranked for the subject by code, strongest links first (`agent/feed.py`). Events without
        factors are tagged once by the model if `tag` and a model is configured; without one, they are ranked by
        activity and country links only."""
        profile = self.store.profiles[profile_id]
        lang = norm_lang(language or self.settings.language)
        untagged: list[str] = []
        for event in list(self.store.events.values()):
            if event.macro_factors or (since and _aware(event.published_at) < _aware(since)):
                continue
            tagged = tag_event(self.llm, self.settings, event, lang) if tag and self._can_tag() else None
            if tagged is not None:
                self.store.add_event(tagged)
            else:
                untagged.append(event.event_id)
        return rank_feed(
            profile,
            self.store.events.values(),
            lang=lang,
            policy=self.impact_policy,
            top=top,
            since=since,
            include_unrelated=include_unrelated,
            untagged=untagged,
        )

    def brief(
        self,
        profile_id: str,
        *,
        top: int = 3,
        since: datetime | None = None,
        role: str | None = None,
        language: str | None = None,
    ) -> Brief:
        """The `top` strongest direct news of the feed, each explained by the agent, cut down to a paragraph by
        code (headline, impact, the first “why it matters” statement, the first action), highest impact first."""
        from .eval.harness import DEFAULT_ROLE

        lang = norm_lang(language or self.settings.language)
        profile = self.store.profiles[profile_id]
        feed = self.feed(profile_id, top=None, since=since, language=lang)
        direct = [it for it in feed.items if it.relevance == "direct"]
        role = role or DEFAULT_ROLE.get(profile.kind.value, "other")
        items = []
        if self.replay is not None:  # demo mode: only the news that has a recording for this subject
            recorded = {e for p, e, *_ in self.replay.pairs() if p == profile_id}
            direct = [it for it in direct if it.event_id in recorded]
        for it in direct[:top]:
            run = self.analyze(profile_id, it.event_id, role=role, language=lang)
            ins = run.insight
            items.append(
                BriefItem(
                    event_id=it.event_id,
                    title=it.title,
                    relevance=it.relevance,
                    insight_id=ins.insight_id if ins else None,
                    headline=ins.headline if ins else None,
                    overall_impact=ins.overall_impact if ins else None,
                    overall_direction=ins.overall_direction.value if ins else None,
                    why=ins.why_it_matters[0].text if ins and ins.why_it_matters else None,
                    action=ins.actions[0].text if ins and ins.actions else None,
                    error=None if ins else run.trace.failure_reason(),
                )
            )
        order = {"high": 0, "medium": 1, "low": 2}
        items.sort(key=lambda b: order.get(b.overall_impact.value, 3) if b.overall_impact else 4)
        return Brief(
            profile_id=profile_id,
            language=lang,
            created_at=datetime.now(timezone.utc),
            items=items,
            considered=len(direct),
        )

    def _can_tag(self) -> bool:
        if self.replay is not None:
            return False
        return self._llm is not None or bool(self.settings.api_key)

    news_from_text = staticmethod(news_from_text)

    # ------------------------------------------------------------------ persistence

    @staticmethod
    def save_run(run: AgentRun, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        name = run.insight.insight_id if run.insight else run.trace.trace_id
        path = folder / f"{name}.json"
        data = {
            "insight": run.insight.model_dump(mode="json") if run.insight else None,
            "trace": run.trace.model_dump(mode="json"),
            "state": run.state.model_dump(mode="json"),
            "issues": [i.model_dump(mode="json") for i in run.issues],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    def load_run(self, path: str | Path) -> AgentRun:
        data = read_json(path)
        insight = Insight.model_validate(data["insight"]) if data.get("insight") else None
        state = AgentState.model_validate(data["state"])
        run = AgentRun(insight=insight, trace=Trace.model_validate(data["trace"]), state=state)
        if state.context.profile:
            self.store.add_profile(state.context.profile)
        if state.context.event:
            self.store.add_event(state.context.event)
        if insight:
            self.runs[insight.insight_id] = run
        return run
