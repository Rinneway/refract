"""Command line: `refract --help`.

refract demo                                   # scripted run, no API key needed
refract analyze --profile demo_coffee_ip --event ev_usn_vat_ru --role entrepreneur --lang en
refract analyze --profile demo_family_saver --title "…" --text "…"
refract chat --run runs/ins_x.json --q "What about my mortgage?"
refract theses --docs examples/demo/knowledge/docs --out theses.json
refract news fetch --feeds examples/feeds.example.json --data my_data
refract eval pairs|run|judge|report …
refract serve --port 8000                      # REST API (extra: api)
refract mcp                                    # MCP server over stdio (extra: mcp)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")


def _rf(args):
    from .config import load_settings
    from .facade import Refract

    settings = load_settings(getattr(args, "config", None))
    if getattr(args, "lang", None):
        settings = settings.with_language(args.lang)
    if getattr(args, "demo", False):  # no model: recorded answers, the packaged demo data
        if getattr(args, "data", None):
            sys.stderr.write("refract: --demo uses the packaged demo data; --data is ignored\n")
        return Refract.demo(settings=settings)
    rf = Refract.from_dir(getattr(args, "data", None), settings=settings)
    for problem in rf.store.problems:  # skipped files: say so instead of silently working with less data
        sys.stderr.write(f"refract: {problem}\n")
    return rf


def cmd_demo(args) -> int:
    from .demo import run_demo
    from .render import insight_markdown, trace_markdown

    if args.list:
        from .demo import Replay

        for rec in Replay().items:
            _out(f"{rec.profile_id}\t{rec.event_id}\t{rec.role}\t{rec.language}\t{rec.source}")
        return 0
    run = run_demo(args.scenario)
    if args.json:
        _out(
            json.dumps(
                {
                    "insight": run.insight.model_dump(mode="json") if run.insight else None,
                    "trace": run.trace.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        return 0
    _out(
        f"# Scripted demo: {args.scenario}\n\nModel answers are scripted; tools, checks, impact ceiling and policy are real.\n"
    )
    if run.insight:
        _out(insight_markdown(run.insight))
    _out("\n" + trace_markdown(run.trace))
    return 0


def _fail(msg: str) -> int:
    sys.stderr.write(f"refract: {msg}\n")
    return 2


def cmd_analyze(args) -> int:
    from .agent import ABLATIONS, ablation
    from .render import insight_markdown, trace_markdown
    from .textutil import read_text

    rf = _rf(args)
    if args.profile not in rf.store.profiles:
        return _fail(f"unknown profile {args.profile!r}; `refract profiles` lists them")
    if args.ablation not in ABLATIONS:
        return _fail(f"unknown ablation {args.ablation!r}; choose one of {', '.join(ABLATIONS)}")
    if args.event and args.event not in rf.store.events:
        return _fail(f"unknown event {args.event!r}; `refract events` lists them")
    if args.event:
        event = args.event
    elif args.title and (args.text or args.text_file):
        if args.text_file and not Path(args.text_file).is_file():
            return _fail(f"no such file: {args.text_file}")
        text = read_text(args.text_file, fallbacks=("cp1251",)) if args.text_file else args.text
        event = rf.news_from_text(args.title, text, publisher=args.publisher, url=args.url)
    else:
        return _fail("give --event ID, or --title with --text / --text-file")
    cfg = (
        replace(ablation(args.ablation, rf.config), max_iterations=args.iterations)
        if args.iterations is not None
        else ablation(args.ablation, rf.config)
    )

    def progress(ev: dict) -> None:
        if ev.get("status") == "started" and not args.quiet:
            sys.stderr.write(f"… {ev.get('label')}\n")

    run = rf.analyze(
        args.profile, event, role=args.role, language=args.lang, config=cfg, on_progress=progress
    )
    if run.trace.config.get("replay"):
        sys.stderr.write(_replay_note(run, args.role, args.lang))
    if args.out:
        path = rf.save_run(run, args.out)
        sys.stderr.write(f"saved: {path}\n")
    if not run.insight:
        sys.stderr.write(f"no insight — {run.trace.failure_reason()}\n")
    if args.json:
        _out(run.insight.model_dump_json(indent=1) if run.insight else "null")
    else:
        if run.insight:
            _out(insight_markdown(run.insight, rf.settings.institution))
        _out("\n" + trace_markdown(run.trace))
    return 0 if run.ok else 1


def _replay_note(run, role: str | None, lang: str | None) -> str:
    """What demo mode replayed, and in which role and language if not the asked ones."""
    t = run.trace
    got = (t.role.value if t.role else None, t.language)
    note = f"demo: model answers replayed from {t.config['replay']} — not a live model\n"
    if (role and role != got[0]) or (lang and lang != got[1]):
        note += f"demo: the recording is for role {got[0]}, language {got[1]}; shown as recorded\n"
    return note


def cmd_chat(args) -> int:
    from .render import chat_markdown
    from .schemas import ChatTurn

    if not Path(args.run).is_file():
        return _fail(f"no such run file: {args.run}")
    rf = _rf(args)
    run = rf.load_run(args.run)
    if run.insight is None:
        return _fail(f"{args.run} holds a failed run without an insight — {run.trace.failure_reason()}")
    history = []

    for q in args.q:
        res = rf.chat(run.insight.insight_id, q, history).response
        _out(f"\n> {q}\n\n{chat_markdown(res, rf.settings.institution)}")
        history += [ChatTurn(role="user", text=q), ChatTurn(role="assistant", text=res.answer)]
    return 0


def cmd_list(args) -> int:
    rf = _rf(args)
    if args.what == "profiles":
        for p in rf.store.profiles.values():
            _out(f"{p.profile_id:28} {p.kind.value:16} {p.name or ''}")
    else:
        for e in sorted(rf.store.events.values(), key=lambda e: e.published_at):
            _out(f"{e.event_id:22} {e.published_at:%Y-%m-%d} [{e.language}] {e.title}")
    return 0


def cmd_feed(args) -> int:
    from datetime import datetime

    from .eval.harness import DEFAULT_ROLE
    from .render import feed_markdown, insight_markdown

    rf = _rf(args)
    if args.profile not in rf.store.profiles:
        return _fail(f"unknown profile {args.profile!r}; `refract profiles` lists them")
    try:
        since = datetime.fromisoformat(args.since) if args.since else None
    except ValueError:
        return _fail(f"--since {args.since!r}: use YYYY-MM-DD")
    feed = rf.feed(args.profile, top=args.top, since=since, tag=not args.no_tag, include_unrelated=args.all)
    if args.json and not args.analyze:
        _out(feed.model_dump_json(indent=1))
        return 0
    _out(feed_markdown(feed))
    direct = [it for it in feed.items if it.relevance == "direct"][: args.analyze]
    role = args.role or DEFAULT_ROLE.get(rf.store.profiles[args.profile].kind.value, "other")
    for it in direct:
        run = rf.analyze(args.profile, it.event_id, role=role)
        if args.out:
            sys.stderr.write(f"saved: {rf.save_run(run, args.out)}\n")
        _out("\n---\n")
        _out(
            insight_markdown(run.insight, rf.settings.institution)
            if run.insight
            else f"{it.event_id}: no insight — {run.trace.failure_reason()}"
        )
    return 0


def cmd_brief(args) -> int:
    from datetime import datetime

    from .render import brief_markdown

    rf = _rf(args)
    if args.profile not in rf.store.profiles:
        return _fail(f"unknown profile {args.profile!r}; `refract profiles` lists them")
    try:
        since = datetime.fromisoformat(args.since) if args.since else None
    except ValueError:
        return _fail(f"--since {args.since!r}: use YYYY-MM-DD")
    brief = rf.brief(args.profile, top=args.top, since=since, role=args.role)
    if args.out:
        for item in brief.items:
            if item.insight_id:
                sys.stderr.write(f"saved: {rf.save_run(rf.runs[item.insight_id], args.out)}\n")
    _out(brief.model_dump_json(indent=1) if args.json else brief_markdown(brief))
    return 0


def cmd_theses(args) -> int:
    from .config import load_settings
    from .knowledge.theses import extract_theses
    from .llm import LLMClient

    theses, stats = extract_theses(
        LLMClient(load_settings(args.config)), Path(args.docs), workers=args.workers
    )
    Path(args.out).write_text(
        json.dumps([t.model_dump(mode="json") for t in theses], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    _out(json.dumps(stats, ensure_ascii=False, indent=1))
    return 0


def cmd_news(args) -> int:
    from datetime import datetime, timezone

    from .knowledge.store import load_models
    from .news.rss import collect, load_feeds
    from .schemas import NewsEvent

    folder = Path(args.data) / "events"
    stored = load_models(sorted(folder.glob("*.json")), NewsEvent, problems=[]) if folder.exists() else []
    known = {a.url for e in stored for a in e.articles if a.url}
    events, status = collect(load_feeds(args.feeds), limit_per_feed=args.limit, known_urls=known)
    for name, s in status.items():
        sys.stderr.write(f"{name}: {s}\n")
    if not events:
        _out("no new events")
        return 0
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"rss_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    path.write_text(
        json.dumps([e.model_dump(mode="json") for e in events], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    _out(f"{len(events)} events → {path}")
    return 0


def cmd_eval(args) -> int:
    import random

    from .eval import Experiment, load_pairs, make_pairs, save_pairs, write_report
    from .eval.labels import export_labels, import_labels, is_human

    if args.step == "pairs":
        rf = _rf(args)
        pairs = make_pairs(rf, languages=args.languages.split(","))
        total = len(pairs)
        if args.sample is not None and args.sample < total:
            pairs = sorted(random.Random(args.seed).sample(pairs, args.sample), key=lambda p: p.id)
        save_pairs(pairs, args.out)
        _out(
            f"{len(pairs)} pairs"
            + (f" of {total} (seed {args.seed})" if len(pairs) < total else "")
            + f" → {args.out}"
        )
        return 0
    out = Path(args.out)
    pairs_path = Path(args.pairs) if args.pairs else out / "pairs.jsonl"
    if not pairs_path.is_file():
        return _fail(f"no pairs file {pairs_path}: run `refract eval pairs --out {pairs_path}` first")
    pairs = load_pairs(pairs_path)
    if args.step == "label":
        name = args.name or "human"
        if args.action == "export":
            try:
                folder = export_labels(
                    out,
                    pairs,
                    args.systems.split(",") if args.systems else None,
                    sample=args.sample,
                    seed=args.seed,
                    name=name,
                    institution=_rf(args).settings.institution,
                    lang=args.lang or "ru",
                    force=args.force,
                )
            except (FileExistsError, ValueError) as e:
                return _fail(str(e))
            n = len(list((folder / "items").glob("*.md")))
            _out(
                f"{n} answers to label → {folder}/sheet.csv (pages in {folder}/items, read {folder}/README.md)"
            )
            return 0
        if args.action == "import":
            try:
                saved, empty, problems = import_labels(out, name, args.file)
            except FileNotFoundError as e:
                return _fail(str(e))
            for p in problems:
                _out(f"skipped {p}")
            _out(f"{saved} labels saved as judge `{name}`; {empty} rows empty, {len(problems)} with problems")
            return 0 if not problems else 1
        return _fail("`eval label` needs `export` or `import`")
    if args.step == "report":
        found = sorted(p.name for p in (out / "judge").iterdir()) if (out / "judge").exists() else []
        # LLM judges first (the first one is primary), human labels after them
        found = [j for j in found if not is_human(out, j)] + [j for j in found if is_human(out, j)]
        judges = args.judges.split(",") if args.judges else found
        S = write_report(out, pairs, judges, args.reference)
        _out((out / "report.md").read_text(encoding="utf-8"))
        return 0 if S["systems"] else 1
    rf = _rf(args)
    for note in rf.settings.notes:
        if "overrides" in note or "not set" in note:
            _out(f"note: {note}")
    exp = Experiment(rf, out, pairs, workers=args.workers)
    if not (out / "pairs.jsonl").exists():
        save_pairs(pairs, out / "pairs.jsonl")
    if args.step == "run":
        exp.run((args.systems or "full,single_call,retelling").split(","), force=args.force)
    elif args.step == "judge":
        exp.judge(
            args.judges.split(",") if args.judges else ["verifier"],
            args.systems.split(",") if args.systems else None,
            force=args.force,
        )
    return 0


def cmd_serve(args) -> int:  # pragma: no cover
    from .api import serve

    serve(args.host, args.port, _rf(args))
    return 0


def cmd_mcp(args) -> int:  # pragma: no cover
    from .mcp_server import build_server

    build_server(_rf(args)).run()
    return 0


def cmd_models(args) -> int:
    from .config import load_settings
    from .llm import LLMClient

    s = load_settings(args.config)
    _out(f"endpoints: {', '.join(s.base_urls)}")
    for alias, spec in s.models.items():
        _out(f"  {alias}: {spec.id}")
    for url, models in LLMClient(s).list_models().items():
        _out(f"\n{url}:")
        _out("\n".join(f"  {m}" for m in models) if isinstance(models, list) else f"  {models}")
    return 0


def cmd_check(args) -> int:
    from .config import load_settings
    from .doctor import run_check
    from .llm import LLMClient

    ok, report = run_check(LLMClient(load_settings(args.config)), args.models.split(","))
    _out(report)
    return 0 if ok else 1


def cmd_schemas(args) -> int:
    from . import schemas as S
    from .agent.progress import InsightProgress

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = [
        S.Profile,
        S.NewsEvent,
        S.Thesis,
        S.KnowledgeChunk,
        S.Insight,
        S.InsightContent,
        S.Trace,
        S.AgentState,
        S.ChatRequest,
        S.ChatResponse,
        S.Feed,
        S.Brief,
        InsightProgress,
    ]
    for m in models:
        name = "".join("_" + c.lower() if c.isupper() else c for c in m.__name__).lstrip("_")
        (out / f"{name}.schema.json").write_text(
            json.dumps(m.model_json_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    _out(f"{len(models)} schemas → {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="refract", description="Grounded, personalised insights from news.")
    p.add_argument("--version", action="version", version=f"refract {__version__}")
    p.add_argument("--config", help="path to refract.toml (default: ./refract.toml or $REFRACT_CONFIG)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_arg(sp, demo: bool = True):
        sp.add_argument("--data", help="data folder (profiles/, events/, knowledge/); default examples/demo")
        if demo:
            sp.add_argument(
                "--demo",
                action="store_true",
                help="no model: replay recorded answers of a real run on the demo data (checks are live code)",
            )

    sp = sub.add_parser("demo", help="scripted end-to-end run without an API key")
    sp.add_argument("scenario", nargs="?", default="rate", choices=["rate", "injection", "dollar"])
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--list", action="store_true", help="the pairs `--demo` can replay, and exit")
    sp.set_defaults(fn=cmd_demo)

    sp = sub.add_parser("analyze", help="run the agent for one subject × event")
    data_arg(sp)
    sp.add_argument("--profile", required=True)
    sp.add_argument("--event", help="event id from the store")
    sp.add_argument("--title")
    sp.add_argument("--text")
    sp.add_argument("--text-file")
    sp.add_argument("--publisher", default="user")
    sp.add_argument("--url")
    sp.add_argument(
        "--role",
        default="cfo",
        help="owner, ceo, cfo, chief_accountant, treasurer, entrepreneur, individual, other",
    )
    sp.add_argument("--lang", choices=["ru", "en"], help="language of the insight")
    sp.add_argument(
        "--ablation",
        default="full",
        help="full, no_plan, no_verifier, no_rules, no_revise, no_self_check, no_cap, no_policy, no_attribution, no_reconcile",
    )
    sp.add_argument("--iterations", type=int, help="revise rounds (default 1)")
    sp.add_argument("--out", help="folder to save insight, trace and state")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(fn=cmd_analyze)

    sp = sub.add_parser("chat", help="follow-up questions about a saved run")
    data_arg(sp)
    sp.add_argument("--run", required=True, help="JSON saved by `analyze --out`")
    sp.add_argument("--q", action="append", required=True, help="question; repeat for several turns")
    sp.add_argument("--lang", choices=["ru", "en"])
    sp.set_defaults(fn=cmd_chat)

    for what in ("profiles", "events"):
        sp = sub.add_parser(what, help=f"list {what} in the store")
        data_arg(sp)
        sp.set_defaults(fn=cmd_list, what=what)

    sp = sub.add_parser("feed", help="stored news ranked for one subject by code, strongest links first")
    data_arg(sp)
    sp.add_argument("--profile", required=True)
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument("--since", help="only news published on or after this date, YYYY-MM-DD")
    sp.add_argument("--all", action="store_true", help="also news with nothing in common with the subject")
    sp.add_argument("--no-tag", action="store_true", help="do not tag untagged news with the model")
    sp.add_argument(
        "--analyze", type=int, default=0, metavar="K", help="run the agent on the K strongest direct items"
    )
    sp.add_argument("--role", help="reader role for --analyze (default: by the subject's kind)")
    sp.add_argument("--out", help="with --analyze: folder to save the runs")
    sp.add_argument("--lang", choices=["ru", "en"])
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_feed)

    sp = sub.add_parser(
        "brief", help="the strongest news for one subject, each explained, in one short brief"
    )
    data_arg(sp)
    sp.add_argument("--profile", required=True)
    sp.add_argument("--top", type=int, default=3, help="how many news to explain (each is one agent run)")
    sp.add_argument("--since", help="only news published on or after this date, YYYY-MM-DD")
    sp.add_argument("--role", help="reader role (default: by the subject's kind)")
    sp.add_argument("--out", help="folder to save the full runs")
    sp.add_argument("--lang", choices=["ru", "en"])
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_brief)

    sp = sub.add_parser("theses", help="extract the institution's theses from documents (uses the LLM)")
    sp.add_argument("--docs", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--workers", type=int, default=4)
    sp.set_defaults(fn=cmd_theses)

    sp = sub.add_parser("news", help="collect news")
    news = sp.add_subparsers(dest="news_cmd", required=True)
    nf = news.add_parser("fetch", help="RSS/Atom feeds → events")
    nf.add_argument("--feeds", required=True)
    nf.add_argument("--data", required=True)
    nf.add_argument("--limit", type=int, default=30)
    nf.set_defaults(fn=cmd_news)

    sp = sub.add_parser("eval", help="experiments: pairs → run → judge → (label) → report")
    sp.add_argument("step", choices=["pairs", "run", "judge", "label", "report"])
    sp.add_argument("action", nargs="?", choices=["export", "import"], help="for `label`")
    data_arg(sp, demo=False)
    sp.add_argument("--out", required=True, help="experiment folder (for `pairs`: the pairs file)")
    sp.add_argument("--pairs", help="pairs file (default <out>/pairs.jsonl)")
    sp.add_argument("--languages", default="ru", help="for `pairs`: ru,en")
    sp.add_argument(
        "--systems",
        help="comma-separated; `run` default: full,single_call,retelling; `judge` default: every system that was run",
    )
    sp.add_argument("--judges", help="model aliases or ids, comma-separated; the first is primary")
    sp.add_argument("--reference", default="full")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--force", action="store_true")
    sp.add_argument(
        "--lang", choices=["ru", "en"], help="default language; for `label export` also of the instructions"
    )
    sp.add_argument("--sample", type=int, help="for `pairs` and `label export`: a random sample of this size")
    sp.add_argument("--seed", type=int, default=0, help="seed of the sample")
    sp.add_argument("--name", help="for `label`: the judge name of the labels (default human)")
    sp.add_argument("--file", help="for `label import`: the filled sheet (default labels/<name>/sheet.csv)")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("serve", help="REST API (pip install 'refract-agent[api]')")
    data_arg(sp)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("mcp", help="MCP server over stdio (pip install 'refract-agent[mcp]')")
    data_arg(sp)
    sp.set_defaults(fn=cmd_mcp)

    sp = sub.add_parser("models", help="show configured models and what the endpoints offer")
    sp.set_defaults(fn=cmd_models)

    sp = sub.add_parser(
        "check", help="check the endpoint: models, structured output, context window, reasoning"
    )
    sp.add_argument("--models", default="main,verifier", help="model aliases or ids, comma-separated")
    sp.set_defaults(fn=cmd_check)

    sp = sub.add_parser("schemas", help="export JSON Schemas of the contracts")
    sp.add_argument("--out", default="docs/schemas")
    sp.set_defaults(fn=cmd_schemas)
    return p


def utf8_streams() -> None:
    """Russian text on a Windows pipe or file (`refract … > out.md`, CI logs) — stdout there is cp1252/cp866 and
    would fail on the first Cyrillic letter. A console already takes Unicode; elsewhere it is UTF-8 anyway."""
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        reconfigure = getattr(stream, "reconfigure", None)
        if enc != "utf8" and reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):  # a stream that cannot change stays as it is
                reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    from .demo import DemoError
    from .textutil import DataError

    utf8_streams()
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except BrokenPipeError:  # `refract … | head`
        return 0
    except DataError as e:  # a broken config or data file: the message names it
        return _fail(str(e))
    except DemoError as e:  # demo mode cannot do it: the message lists what it can
        return _fail(str(e))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
