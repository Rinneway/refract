"""Progress of an insight for a UI: the agent's `on_progress` events → an `InsightProgress` snapshot.

    tracker = ProgressTracker(event_id, profile_id, role)
    run = Agent(llm, store, on_progress=tracker).run(profile_id, event, role=role)
    tracker.snapshot()  # status processing → ready / failed, steps pending / running / done

Steps not started yet are `pending`; tools appear as they are called. Steps that were never needed (revise, a new
headline) are dropped at the end.
"""

from __future__ import annotations

import threading
from typing import Any, Literal

from pydantic import Field

from ..config import Institution
from ..i18n import STRINGS, t
from ..schemas import Contract

SKELETON = ("get_profile", "plan", "draft_insight", "verify", "finalize")


class ProgressStep(Contract):
    name: str
    label: str
    status: Literal["pending", "running", "done", "failed"]


class InsightProgress(Contract):
    event_id: str
    profile_id: str
    role: str | None = None
    status: Literal["processing", "ready", "failed"] = "processing"
    insight_id: str | None = None
    error: str | None = None
    steps: list[ProgressStep] = Field(default_factory=list)


def step_labels(lang: str = "ru", institution: Institution | None = None) -> dict[str, str]:
    """Labels of the agent's steps in the run language, as the agent's own progress events name them."""
    inst = institution or Institution()
    names = {"institution": inst.display(lang), "institution_gen": inst.genitive(lang)}
    return {n: t(f"step.{n}", lang, **names) for n in SKELETON if f"step.{n}" in STRINGS}


class ProgressTracker:
    def __init__(
        self,
        event_id: str,
        profile_id: str,
        role: str | None = None,
        labels: dict[str, str] | None = None,
        *,
        lang: str = "ru",
        institution: Institution | None = None,
    ) -> None:
        """Steps not started yet are shown with `labels` (default: the agent's own labels in `lang`)."""
        self._lock = threading.Lock()
        labels = labels if labels is not None else step_labels(lang, institution)
        self._state = InsightProgress(
            event_id=event_id,
            profile_id=profile_id,
            role=role,
            steps=[ProgressStep(name=n, label=labels.get(n, n), status="pending") for n in SKELETON],
        )

    def __call__(self, event: dict[str, Any]) -> None:
        with self._lock:
            status, name = event.get("status"), event.get("name", "")
            label = event.get("label") or name
            steps = self._state.steps
            if status == "started":
                pending = next((s for s in steps if s.name == name and s.status == "pending"), None)
                if pending:
                    pending.status, pending.label = "running", label
                else:
                    first = next((i for i, s in enumerate(steps) if s.status == "pending"), len(steps))
                    steps.insert(first, ProgressStep(name=name, label=label, status="running"))
            elif status == "done":
                failed = (event.get("step") or {}).get("verdict") == "failed"
                running = next((s for s in reversed(steps) if s.name == name and s.status == "running"), None)
                running = running or next(
                    (s for s in steps if s.name == name and s.status == "pending"), None
                )
                if running is None:
                    running = ProgressStep(name=name, label=label, status="running")
                    steps.append(running)
                running.status = "failed" if failed else "done"
            elif status == "finished":
                ok = bool(event.get("ok"))
                self._state.status = "ready" if ok else "failed"
                self._state.insight_id = event.get("insight_id")
                self._state.error = event.get("error")
                kept = []
                for s in steps:
                    if s.status == "pending":
                        continue
                    if s.status == "running":
                        s.status = "done" if ok else "failed"
                    kept.append(s)
                self._state.steps = kept

    def snapshot(self) -> InsightProgress:
        with self._lock:
            return self._state.model_copy(deep=True)
