"""Experiments: systems × subject–event pairs → outputs → judges → report.

    refract eval pairs  --data examples/demo --out runs/exp1/pairs.jsonl
    refract eval run    --pairs runs/exp1/pairs.jsonl --out runs/exp1 --systems full,no_verifier,single_call,retelling
    refract eval judge  --out runs/exp1 --judges verifier,main
    refract eval report --out runs/exp1 --reference full

Systems:
- every name in `refract.agent.ABLATIONS` — the agent with components switched off (`full` is the whole agent);
- `single_call` — one call of the same model with the same insight prompt and the house view matched by the event's
  factors: no planning, tools, review or revision;
- `retelling` — one call with the event and the subject data and a plain prompt, without the house view or rules.

Every step is resumable: finished outputs are not recomputed. Objective metrics (grounding violations, invented
numbers, impact above the computed ceiling) are computed by code for every system, so they do not depend on a judge.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..agent import ABLATIONS, ablation
from ..agent import checks as rule_checks
from ..agent.chat import context_from_state
from ..agent.context import InsightContext
from ..agent.impact import ORDER, impact_cap
from ..doctor import error_hint
from ..facade import Refract
from ..llm import LLMError
from ..prompts import classify as classify_prompt
from ..prompts import insight as insight_prompt
from ..schemas import ImpactLevel, InsightContent, NewsEvent
from ..textutil import invented_numbers, script_share
from . import judge as jp

BASELINES = ("single_call", "retelling")
SYSTEMS = (*ABLATIONS, *BASELINES)
DEFAULT_ROLE = {
    "company": "cfo",
    "sole_proprietor": "entrepreneur",
    "self_employed": "entrepreneur",
    "individual": "individual",
}


@dataclass
class Pair:
    profile_id: str
    event_id: str
    role: str = "cfo"
    language: str = "ru"
    gold_impact: str | None = None  # human label: low | medium | high
    gold_relevant: bool | None = None
    note: str = ""

    @property
    def id(self) -> str:
        return f"{self.profile_id}__{self.event_id}__{self.role}__{self.language}"


def load_pairs(path: str | Path) -> list[Pair]:
    return [
        Pair(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_pairs(pairs: Iterable[Pair], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "\n".join(json.dumps(asdict(p), ensure_ascii=False) for p in pairs) + "\n", encoding="utf-8"
    )


def make_pairs(
    rf: Refract, languages: Iterable[str] = ("ru",), roles: dict[str, str] | None = None
) -> list[Pair]:
    """Every profile × every event × every language, with the default reader role for the subject's kind."""
    roles = roles or DEFAULT_ROLE
    return [
        Pair(p.profile_id, e.event_id, roles.get(p.kind.value, "other"), lang)
        for p in rf.store.profiles.values()
        for e in rf.store.events.values()
        for lang in languages
    ]


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


@dataclass
class Experiment:
    rf: Refract
    out: Path
    pairs: list[Pair] = field(default_factory=list)
    workers: int = 4
    log: Callable[[str], None] = print

    def __post_init__(self) -> None:
        self.out = Path(self.out)

    # ------------------------------------------------------------------ helpers

    def _parallel(self, fn, items: list) -> None:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(fn, x) for x in items]
            for f in as_completed(futures):
                try:
                    if msg := f.result():
                        self.log(msg)
                except Exception as e:  # noqa: BLE001 — one failed pair must not stop the experiment
                    self.log(f"error: {type(e).__name__}: {e}")

    def classified(self, event: NewsEvent, lang: str) -> NewsEvent:
        """One tagging per event, shared by all systems (so they differ only in what is being compared)."""
        if event.macro_factors:
            return event
        path = self.out / "_events" / f"{event.event_id}.json"
        if (cached := _read(path)) is not None:
            return NewsEvent.model_validate(cached)
        r = self.rf.settings.route("classify_news")
        res = self.rf.llm.chat_json(
            r.model,
            classify_prompt.build_messages(event, lang),
            classify_prompt.NewsTags,
            max_tokens=r.max_tokens,
            step="eval_classify",
        )
        tagged = classify_prompt.apply(event, res.value) if res.value else event
        _write(path, tagged.model_dump(mode="json"))
        return tagged

    def _ctx(self, pair: Pair, event: NewsEvent, theses=None) -> InsightContext:
        rf = self.rf
        return InsightContext(
            profile=rf.store.profiles[pair.profile_id],
            event=event,
            theses=list(theses or []),
            role=pair.role,
            lang=pair.language,
            institution=rf.settings.institution,
            policy=rf.policy,
            offer_products=rf.settings.offer_products,
        )

    def _objective(self, content: dict | None, ctx: InsightContext, structured: bool) -> dict[str, Any]:
        """Code-computed metrics of what the reader would see."""
        if not content:
            return {}
        cap = impact_cap(ctx.profile, ctx.event, lang=ctx.lang)
        level = str(content.get("overall_impact", "")).lower()
        over = level in {lvl.value for lvl in ImpactLevel} and ORDER[ImpactLevel(level)] > ORDER[cap.level]
        out: dict[str, Any] = {"impact": level, "cap": cap.level.value, "over_cap": over}
        share = script_share(_text_of(content), ctx.lang)
        out["language_share"], out["language_ok"] = round(share, 3), share >= rule_checks.LANGUAGE_MIN
        if structured:
            try:
                ic = InsightContent.model_validate(
                    {k: content[k] for k in InsightContent.model_fields if k in content}
                )
            except Exception:  # noqa: BLE001
                return {**out, "rule_errors": None, "invented_numbers": None}
            errors = rule_checks.errors(rule_checks.rule_check(ic, ctx))
            out["rule_errors"] = [e.model_dump(mode="json") for e in errors]
            out["invented_numbers"] = sum(
                len(invented_numbers([c.text], ctx.render())) for c in ic.all_claims()
            )
            out["claims"] = len(ic.all_claims())
        return out

    # ------------------------------------------------------------------ run

    def run(self, systems: Iterable[str], force: bool = False) -> None:
        systems = list(systems)
        unknown = [s for s in systems if s not in SYSTEMS]
        if unknown:
            raise ValueError(f"unknown systems {unknown}; choose from {list(SYSTEMS)}")
        tasks = [(s, p) for s in systems for p in self.pairs]

        def one(task) -> str:
            system, pair = task
            path = self.out / "runs" / system / f"{pair.id}.json"
            if not force and (_read(path) or {}).get("content"):
                return f"{system} {pair.id}: cached"  # a failed run is retried
            event = self.rf.store.events[pair.event_id]
            started = time.perf_counter()
            if system in ABLATIONS:
                data = self._run_agent(system, pair, event)
            else:
                data = self._run_baseline(system, pair, event)
            data.update(
                {"system": system, "pair": asdict(pair), "wall_s": round(time.perf_counter() - started, 2)}
            )
            _write(path, data)
            return (
                f"{system} {pair.id}: {'ok' if data.get('content') else 'NO ANSWER'} ({data['wall_s']:.0f} s)"
            )

        self._parallel(one, tasks)

    def _run_agent(self, system: str, pair: Pair, event: NewsEvent) -> dict[str, Any]:
        cfg = ablation(system, self.rf.config)
        event = self.classified(event, pair.language)
        agent = self.rf.agent(cfg)
        run = agent.run(pair.profile_id, event, role=pair.role, language=pair.language)
        ctx = context_from_state(run.state, agent)
        content = run.insight.model_dump(mode="json") if run.insight else None
        full_ctx = self._ctx(pair, run.state.context.event or event, ctx.theses)
        full_ctx.research, full_ctx.related = ctx.research, ctx.related
        return {
            "content": content,
            "context": ctx.render(),
            "cost": run.trace.totals.cost,
            "latency_ms": run.trace.totals.latency_ms,
            "llm_calls": run.trace.totals.llm_calls,
            "iterations": run.trace.totals.iterations,
            "trace": run.trace.model_dump(mode="json"),
            "state": run.state.model_dump(mode="json"),
            "objective": self._objective(content, full_ctx, structured=True),
        }

    def _run_baseline(self, system: str, pair: Pair, event: NewsEvent) -> dict[str, Any]:
        rf = self.rf
        r = rf.settings.route("draft_insight")
        if system == "retelling":
            ctx = self._ctx(pair, event)
            res = rf.llm.chat_json(
                r.model,
                jp.baseline_messages(ctx.render(), pair.language),
                jp.BaselineAnswer,
                max_tokens=3000,
                step="eval_retelling",
                timeout_s=r.timeout_s,
            )
            structured = False
        else:  # single_call
            ev = self.classified(event, pair.language)
            country = rf.store.profiles[pair.profile_id].country
            theses = rf.store.find_theses(ev.macro_factors, [], ev.title, k=6, country=country)
            ctx = self._ctx(pair, ev, theses)
            res = rf.llm.chat_json(
                r.model,
                insight_prompt.build_messages(ctx),
                InsightContent,
                max_tokens=r.max_tokens,
                step="eval_single_call",
                timeout_s=r.timeout_s,
            )
            structured = True
        content = res.value.model_dump(mode="json") if res.value else None
        return {
            "content": content,
            "error": res.error,
            "context": ctx.render(),
            "cost": res.cost,
            "latency_ms": res.latency_ms,
            "llm_calls": len(res.calls),
            "objective": self._objective(content, ctx, structured=structured),
        }

    # ------------------------------------------------------------------ judge

    def judge(self, judges: Iterable[str], systems: Iterable[str] | None = None, force: bool = False) -> None:
        """Score every system's answers. Only systems that were run are judged; a call that fails (no key, no
        model) is not saved, so the next `judge` retries it; an authorisation error stops the whole step."""
        judges = list(judges)
        ran = (
            sorted(p.name for p in (self.out / "runs").iterdir() if p.is_dir())
            if (self.out / "runs").exists()
            else []
        )
        systems = [s for s in (systems or ran) if s in ran]
        tasks = [(j, s, p) for j in judges for s in systems for p in self.pairs]
        stop: list[str] = []
        inst = {
            "institution": self.rf.settings.institution.display("en"),
            "institution_gen": self.rf.settings.institution.genitive("ru"),
        }
        r = self.rf.settings.route("judge")

        def one(task) -> str:
            judge, system, pair = task
            if stop:
                return ""
            path = self.out / "judge" / _safe(judge) / system / f"{pair.id}.json"
            if not force and (_read(path) or {}).get("score") is not None:
                return f"{judge}/{system} {pair.id}: cached"
            data = _read(self.out / "runs" / system / f"{pair.id}.json")
            if data is None:
                return ""  # the pair was not run for this system
            if not data.get("content"):
                _write(path, {"score": None, "error": "no answer from the system"})
                return f"{judge}/{system} {pair.id}: no answer"
            lang = pair.language
            names = {**inst, "institution": self.rf.settings.institution.display(lang)}
            msgs = jp.judge_messages(
                data["context"], jp.render_answer(data["content"], lang, names), lang, names
            )
            try:
                res = self.rf.llm.chat_json(
                    judge,
                    msgs,
                    jp.JudgeScore,
                    max_tokens=r.max_tokens,
                    step=f"judge_{system}",
                    timeout_s=r.timeout_s,
                )
            except LLMError as e:
                if "HTTP 401" in str(e) or "HTTP 403" in str(e):
                    stop.append(error_hint(str(e), self.rf.llm) or str(e))
                    return f"stopped: {judge} — {stop[0]}"
                raise
            if res.value is None:
                if res.error and ("HTTP 401" in res.error or "HTTP 403" in res.error):
                    stop.append(error_hint(res.error, self.rf.llm) or res.error)
                    return f"stopped: {judge} — {stop[0]}"
                return f"{judge}/{system} {pair.id}: no score ({res.error}) — will be retried"
            _write(
                path,
                {
                    "score": res.value.model_dump(mode="json") if res.value else None,
                    "error": res.error,
                    "cost": res.cost,
                    "judge": judge,
                    "judge_version": jp.VERSION,
                },
            )
            return f"{judge}/{system} {pair.id}: {res.value.overall if res.value else '—'}"

        self._parallel(one, tasks)
        if stop:
            self.log(f"judging stopped, nothing was saved for the remaining answers: {stop[0]}")


def _text_of(content: dict) -> str:
    """Visible text of any system's answer (structured insight or a baseline's lists of strings)."""
    parts: list[str] = [str(content.get("headline", ""))]
    for key in (
        "what_happened",
        "why_it_matters",
        "risks",
        "opportunities",
        "house_view",
        "scenarios",
        "actions",
    ):
        for x in content.get(key) or []:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict):
                parts += [str(x.get(k) or "") for k in ("text", "caveat", "name", "description", "rationale")]
    return " ".join(parts)


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")
