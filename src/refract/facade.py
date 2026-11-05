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
from .agent.impact import ImpactPolicy
from .agent.policy import DEFAULT_POLICY, Policy
from .config import Settings, load_settings
from .i18n import norm_lang
from .knowledge.store import DataStore
from .llm.client import LLM, LLMClient
from .recent import Recent
from .schemas import AgentState, ChatTurn, Insight, NewsArticle, NewsEvent, Profile, Trace


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
        store = DataStore.from_dir(root, embed=embed, cache_dir=Path(settings.cache_dir) / "embeddings")
        return cls(store, settings=settings, **kwargs)

    @property
    def llm(self) -> LLM:
        with self._lock:
            if self._llm is None:
                self._llm = LLMClient(self.settings)
            return self._llm

    # ------------------------------------------------------------------ agent

    def agent(self, config: AgentConfig | None = None, on_progress=None) -> Agent:
        return Agent(
            self.llm,
            self.store,
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
        cfg = replace(config or self.config, language=lang)
        run = self.agent(cfg, on_progress).run(profile, event, role=role, language=lang)
        if run.insight:
            self.runs[run.insight.insight_id] = run
        return run

    def chat(
        self, insight_id: str, question: str, history: list[ChatTurn] | None = None, on_progress=None
    ) -> ChatRun:
        run = self.runs[insight_id]
        agent = self.agent(on_progress=on_progress)
        return ChatAgent(self.llm, self.store, agent=agent).answer(run.state, run.insight, question, history)

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
        data = json.loads(Path(path).read_text(encoding="utf-8"))
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
