"""Human relevance labels for the feed, without a model.

    refract eval relevance export --out runs/rel          # relevance/sheet.csv: every subject × news item, blind
    refract eval relevance import --out runs/rel          # the filled sheet against the links computed by code

The feed calls a news item relevant to a subject when the strongest link computed by code (`agent/impact.py`) is at
least moderate. The sheet shows the annotator the subject and the news, never the code's verdict; the report gives
precision, recall, F1, κ and the share judged relevant at each link strength, plus every disagreement with its links,
which is where the impact links need work. No model is called: the demo events are tagged already, and an event
without factors is reported as untagged rather than guessed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from ..agent.impact import DEFAULT_IMPACT, event_links
from ..i18n import norm_lang
from .stats import weighted_kappa

YES = {"1", "yes", "y", "да", "д", "true", "+"}
NO = {"0", "no", "n", "нет", "н", "false", "-"}

README = {
    "ru": """# Разметка релевантности

В `sheet.csv` по строке на пару «субъект × новость». Прочитайте описание субъекта и новость и поставьте в колонке
`relevant`:

- `1` (или `да`) — новость касается субъекта настолько, что ему стоит о ней узнать: она меняет его доходы, расходы,
  долг, налоги, спрос, поставщиков или планы;
- `0` (или `нет`) — не касается или касается только как общий фон.

`comment` — по желанию. Пустые строки не учитываются. Файл в UTF-8 с BOM, разделитель «;» — открывается в Excel.
Не смотрите в выдачу агента до разметки: так оценка остаётся независимой.
""",
    "en": """# Relevance labels

`sheet.csv` has a row per subject × news item. Read the subject and the news item and fill the `relevant` column:

- `1` (or `yes`) — the news concerns the subject enough that they should hear about it: it changes their income,
  costs, debt, taxes, demand, suppliers or plans;
- `0` (or `no`) — it does not, or only as general background.

`comment` is optional. Empty rows are skipped. UTF-8, comma-separated. Do not look at the agent's feed before
labelling, so the labels stay independent.
""",
}


def _delimiter(lang: str) -> str:
    return ";" if lang == "ru" else ","


def export_relevance(rf: Any, out: Path, *, lang: str = "ru", force: bool = False) -> Path:
    """`out/relevance/sheet.csv` and `README.md`: every subject × every stored news item, without the verdict."""
    lang = norm_lang(lang)
    folder = out / "relevance"
    sheet = folder / "sheet.csv"
    if sheet.exists() and not force:
        raise FileExistsError(f"{sheet} exists; --force overwrites it (and the labels in it)")
    folder.mkdir(parents=True, exist_ok=True)
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=_delimiter(lang))
        w.writerow(["profile_id", "event_id", "subject", "news", "relevant", "comment"])
        for p in rf.store.profiles.values():
            for e in sorted(rf.store.events.values(), key=lambda e: e.published_at):
                subject = f"{p.name or p.profile_id}. {p.summary or ''}".strip()
                news = f"{e.title}. {e.summary or ''}".strip()
                w.writerow([p.profile_id, e.event_id, subject, news, "", ""])
    (folder / "README.md").write_text(README[lang], encoding="utf-8")
    return sheet


def read_sheet(path: Path) -> tuple[dict[tuple[str, str], bool], list[str]]:
    """Filled rows → {(profile_id, event_id): relevant}; problems name the row."""
    text = path.read_text(encoding="utf-8-sig")
    delimiter = ";" if text.split("\n", 1)[0].count(";") > text.split("\n", 1)[0].count(",") else ","
    labels: dict[tuple[str, str], bool] = {}
    problems: list[str] = []
    for i, row in enumerate(csv.DictReader(text.splitlines(), delimiter=delimiter), start=2):
        value = (row.get("relevant") or "").strip().lower()
        if not value:
            continue
        if value not in YES | NO:
            problems.append(f"row {i}: relevant={value!r} — use 1/0, yes/no, да/нет")
            continue
        labels[(row.get("profile_id") or "", row.get("event_id") or "")] = value in YES
    return labels, problems


def relevance_stats(rows: list[tuple[bool, int]]) -> dict[str, Any] | None:
    """(human verdict, link strength 0–3) pairs → agreement of the feed's verdict (strength ≥ 2) with people."""
    if not rows:
        return None
    tp = sum(g and s >= 2 for g, s in rows)
    fp = sum(not g and s >= 2 for g, s in rows)
    fn = sum(g and s < 2 for g, s in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    by_strength = {
        level: {"n": len(xs), "relevant_share": mean([float(g) for g in xs])}
        for level in range(4)
        if (xs := [g for g, s in rows if s == level])
    }
    return {
        "n": len(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": mean([float(g == (s >= 2)) for g, s in rows]),
        "kappa": weighted_kappa([int(g) for g, _ in rows], [int(s >= 2) for _, s in rows], [0, 1]),
        "by_strength": by_strength,
    }


def score_relevance(rf: Any, labels: dict[tuple[str, str], bool], lang: str = "ru") -> dict[str, Any]:
    """The labels against the links computed by code, with every disagreement and its links."""
    lang = norm_lang(lang)
    policy = rf.impact_policy or DEFAULT_IMPACT
    rows: list[tuple[bool, int]] = []
    disagreements, unknown, untagged = [], [], []
    for (pid, eid), gold in sorted(labels.items()):
        profile, event = rf.store.profiles.get(pid), rf.store.events.get(eid)
        if profile is None or event is None:
            unknown.append(f"{pid} × {eid}")
            continue
        if not event.macro_factors:
            untagged.append(eid)
        links = event_links(profile, event, lang, policy)
        strength = max((x.level for x in links), default=0)
        rows.append((gold, strength))
        if gold != (strength >= 2):
            disagreements.append(
                {
                    "profile_id": pid,
                    "event_id": eid,
                    "human": gold,
                    "strength": strength,
                    "links": [x.render(lang) for x in sorted(links, key=lambda x: -x.level)[:3]],
                }
            )
    return {
        "stats": relevance_stats(rows),
        "disagreements": disagreements,
        "unknown": unknown,
        "untagged": sorted(set(untagged)),
    }


def relevance_markdown(result: dict[str, Any]) -> str:
    s = result["stats"]
    if not s:
        return "No filled rows."

    def f(x: float | None) -> str:
        return "—" if x is None else f"{x:.2f}"

    names = {0: "none", 1: "weak", 2: "moderate", 3: "strong"}
    lines = [
        "# Feed relevance against human labels",
        "",
        f"Labelled pairs: {s['n']}. The feed calls a pair relevant at a moderate or strong link.",
        "",
        "| Precision | Recall | F1 | Accuracy | κ |",
        "|---|---|---|---|---|",
        f"| {f(s['precision'])} | {f(s['recall'])} | {f(s['f1'])} | {f(s['accuracy'])} | {f(s['kappa'])} |",
        "",
        "| Link strength | Pairs | Judged relevant |",
        "|---|---|---|",
    ]
    for level, v in sorted(s["by_strength"].items()):
        lines.append(f"| {names[int(level)]} | {v['n']} | {100 * v['relevant_share']:.0f}% |")
    if result["disagreements"]:
        lines += [
            "",
            "## Disagreements",
            "",
            "| Subject | News | Human | Code | Strongest links |",
            "|---|---|---|---|---|",
        ]
        for d in result["disagreements"]:
            links = "<br>".join(d["links"]) or "—"
            human = "relevant" if d["human"] else "not relevant"
            lines.append(
                f"| {d['profile_id']} | {d['event_id']} | {human} | {names[d['strength']]} | {links} |"
            )
    if result["unknown"]:
        lines += ["", f"Not in the data folder, skipped: {', '.join(result['unknown'])}."]
    if result["untagged"]:
        lines += [
            "",
            f"Events without factors (only activity and country links count): {', '.join(result['untagged'])}.",
        ]
    return "\n".join(lines)


def write_relevance(out: Path, result: dict[str, Any]) -> Path:
    folder = out / "relevance"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.md").write_text(relevance_markdown(result) + "\n", encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return folder / "report.md"
