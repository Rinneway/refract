"""REST API with live progress over Server-Sent Events. Install with `pip install 'refract-agent[api]'`.

    refract serve --port 8000            # docs at http://127.0.0.1:8000/docs

| Method | Path | What it does |
| --- | --- | --- |
| GET | /health | version, language, whether an LLM key is configured |
| GET | /profiles, /profiles/{id} | subjects in the store |
| GET | /events, /events/{id} | news events in the store |
| POST | /events | add a news item from text |
| POST | /insights | start the agent → job id (`?wait=true` returns the insight) |
| GET | /jobs/{job_id} | progress snapshot |
| GET | /jobs/{job_id}/stream | progress as SSE until the insight is ready |
| GET | /insights/{id}, /insights/{id}/trace, /insights/{id}/markdown | the result |
| POST | /insights/{id}/chat | follow-up question |

Set `REFRACT_API_TOKEN` to require `Authorization: Bearer <token>` on every request except /health.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import uuid
from datetime import datetime
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .agent import ABLATIONS, ablation
from .agent.progress import InsightProgress, ProgressTracker
from .facade import Refract, news_from_text
from .recent import Recent
from .render import insight_markdown
from .schemas import ChatResponse, ChatTurn, Feed, Insight, NewsEvent, Profile, Trace


class NewsIn(BaseModel):
    title: str = Field(min_length=3, max_length=300)
    text: str = Field(min_length=20, max_length=20000)
    publisher: str = "user"
    url: str | None = None
    published_at: datetime | None = None
    language: Literal["ru", "en"] | None = None


class InsightRequest(BaseModel):
    profile_id: str
    event_id: str | None = None
    news: NewsIn | None = None
    role: str = "cfo"
    language: Literal["ru", "en"] | None = None
    ablation: str = Field(default="full", description=f"One of {sorted(ABLATIONS)}")


class JobOut(BaseModel):
    job_id: str
    status: str
    insight_id: str | None = None


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class _Job:
    def __init__(self, tracker: ProgressTracker) -> None:
        self.tracker = tracker
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.done = threading.Event()
        self.insight_id: str | None = None
        self.error: str | None = None

    def __call__(self, event: dict[str, Any]) -> None:
        self.tracker(event)
        self.events.put(event)


def create_app(rf: Refract | None = None) -> FastAPI:
    app = FastAPI(
        title="Refract API", version=__version__, description="Grounded, personalised insights from news."
    )
    state: dict[str, Any] = {"rf": rf}
    jobs: Recent[str, _Job] = Recent(1000)  # the newest jobs; older ones answer 404
    token = os.getenv("REFRACT_API_TOKEN")

    def get_rf() -> Refract:
        if state["rf"] is None:
            state["rf"] = Refract.from_dir()
        return state["rf"]

    def auth(authorization: str | None = Header(default=None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "invalid or missing token")

    @app.get("/health")
    def health() -> dict[str, Any]:
        r = get_rf()
        return {
            "status": "ok",
            "version": __version__,
            "language": r.settings.language,
            "llm_configured": bool(r.settings.api_key),
            "data_problems": r.store.problems,
        }

    @app.get("/profiles", dependencies=[Depends(auth)])
    def profiles() -> list[dict[str, Any]]:
        return [
            {"profile_id": p.profile_id, "kind": p.kind.value, "name": p.name, "summary": p.summary}
            for p in get_rf().store.profiles.values()
        ]

    @app.get("/profiles/{profile_id}", response_model=Profile, dependencies=[Depends(auth)])
    def profile(profile_id: str) -> Profile:
        p = get_rf().store.profiles.get(profile_id)
        if not p:
            raise HTTPException(404, "profile not found")
        return p

    @app.get("/profiles/{profile_id}/feed", response_model=Feed, dependencies=[Depends(auth)])
    def feed(
        profile_id: str,
        top: int = Query(10, ge=1, le=200),
        since: datetime | None = None,
        tag: bool = Query(True, description="tag news without factors with the model first"),
        all: bool = Query(False, description="also news with nothing in common with the subject"),
        language: Literal["ru", "en"] | None = None,
    ) -> Feed:
        """Stored news ranked for the subject by code, strongest links first."""
        r = get_rf()
        if profile_id not in r.store.profiles:
            raise HTTPException(404, "profile not found")
        return r.feed(profile_id, top=top, since=since, tag=tag, include_unrelated=all, language=language)

    @app.get("/events", dependencies=[Depends(auth)])
    def events(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
        evs = sorted(get_rf().store.events.values(), key=lambda e: e.published_at, reverse=True)[:limit]
        return [
            {
                "event_id": e.event_id,
                "title": e.title,
                "published_at": e.published_at,
                "language": e.language,
                "event_type": e.event_type.value,
            }
            for e in evs
        ]

    @app.get("/events/{event_id}", response_model=NewsEvent, dependencies=[Depends(auth)])
    def event(event_id: str) -> NewsEvent:
        e = get_rf().store.events.get(event_id)
        if not e:
            raise HTTPException(404, "event not found")
        return e

    @app.post("/events", response_model=NewsEvent, dependencies=[Depends(auth)])
    def add_event(news: NewsIn) -> NewsEvent:
        e = news_from_text(**news.model_dump())
        get_rf().store.add_event(e)
        return e

    @app.post("/insights", dependencies=[Depends(auth)])
    def start(req: InsightRequest, wait: bool = False) -> Any:
        r = get_rf()
        if req.profile_id not in r.store.profiles:
            raise HTTPException(404, "profile not found")
        if req.ablation not in ABLATIONS:
            raise HTTPException(422, f"unknown ablation; choose one of {sorted(ABLATIONS)}")
        if req.news is not None:
            ev = news_from_text(**req.news.model_dump())
        elif req.event_id and req.event_id in r.store.events:
            ev = r.store.events[req.event_id]
        else:
            raise HTTPException(404, "give an existing event_id or a news item")
        cfg = ablation(req.ablation, r.config)
        job_id = uuid.uuid4().hex[:12]
        job = _Job(ProgressTracker(ev.event_id, req.profile_id, req.role))
        jobs[job_id] = job

        def work() -> None:
            try:
                run = r.analyze(
                    req.profile_id, ev, role=req.role, language=req.language, config=cfg, on_progress=job
                )
                job.insight_id = run.insight.insight_id if run.insight else None
                if not run.insight:
                    job.error = job.tracker.snapshot().error or "the agent produced no insight"
            except Exception as e:  # noqa: BLE001 — reported through the job
                job.error = f"{type(e).__name__}: {e}"[:300]
            finally:
                job.done.set()
                job.events.put({"status": "closed"})

        if wait:
            work()
            if not job.insight_id:
                raise HTTPException(502, job.error or "the agent produced no insight")
            return r.runs[job.insight_id].insight
        threading.Thread(target=work, daemon=True).start()
        return JobOut(job_id=job_id, status="processing")

    def _job(job_id: str) -> _Job:
        if job_id not in jobs:
            raise HTTPException(404, "job not found")
        return jobs[job_id]

    @app.get("/jobs/{job_id}", response_model=InsightProgress, dependencies=[Depends(auth)])
    def progress(job_id: str) -> InsightProgress:
        return _job(job_id).tracker.snapshot()

    @app.get("/jobs/{job_id}/stream", dependencies=[Depends(auth)])
    def stream(job_id: str) -> StreamingResponse:
        job = _job(job_id)

        def gen():
            while True:
                try:
                    ev = job.events.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    if job.done.is_set():
                        break
                    continue
                if ev.get("status") == "closed":
                    final = {"status": "closed", "insight_id": job.insight_id, "error": job.error}
                    yield f"event: closed\ndata: {json.dumps(final, ensure_ascii=False)}\n\n"
                    break
                yield f"event: {ev.get('status')}\ndata: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    def _run(insight_id: str):
        run = get_rf().runs.get(insight_id)
        if not run:
            raise HTTPException(404, "insight not found")
        return run

    @app.get("/insights/{insight_id}", response_model=Insight, dependencies=[Depends(auth)])
    def insight(insight_id: str) -> Insight:
        return _run(insight_id).insight

    @app.get("/insights/{insight_id}/trace", response_model=Trace, dependencies=[Depends(auth)])
    def trace(insight_id: str) -> Trace:
        return _run(insight_id).trace

    @app.get(
        "/insights/{insight_id}/markdown", response_class=PlainTextResponse, dependencies=[Depends(auth)]
    )
    def markdown(insight_id: str) -> str:
        return insight_markdown(_run(insight_id).insight, get_rf().settings.institution)

    @app.post("/insights/{insight_id}/chat", response_model=ChatResponse, dependencies=[Depends(auth)])
    def chat(insight_id: str, body: ChatIn) -> ChatResponse:
        _run(insight_id)
        return get_rf().chat(insight_id, body.question, body.history).response

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, rf: Refract | None = None) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(rf), host=host, port=port)
