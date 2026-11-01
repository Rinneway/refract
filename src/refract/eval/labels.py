"""Human labels: a blinded sheet for annotators and its import as one more judge.

LLM judges disagree with each other and prefer their own family's style; a paper needs a human reference. `export`
writes, for a sample of answers, one page per answer — the context the system had, its answer in the same neutral
form the LLM judge sees, and the judge's own rubric — and a sheet to fill in. Answers are shuffled and the system
names are kept in `key.json`, which the annotator does not open. `import` turns the filled rows into scores of a
judge named `human` (or any name), so the report compares every LLM judge with the human one.

    labels/<name>/README.md      how to fill in the sheet
    labels/<name>/sheet.csv      one row per answer: scores to fill in (UTF-8 with BOM; “;” for Russian Excel)
    labels/<name>/items/<id>.md  the context and the answer
    labels/<name>/key.json       item → system and pair (for import; not for the annotator)
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import judge as jp
from .harness import Pair, _read, _safe, _write

FLAGS = ("retelling", "impact_calibrated")
COLUMNS = ("item", *jp.CRITERIA, *FLAGS, "overall", "unsupported", "comment")
VERSION = "human-v1"

_README = {
    "ru": """# Разметка ответов

Откройте `items/<item>.md`: там контекст, который был у сервиса, ответ сервиса и критерии. Заполните строку \
`<item>` в `sheet.csv`:

- {criteria} — 0 (нет или плохо), 1 (частично), 2 (хорошо);
- retelling — 1, если ответ только пересказывает новость, иначе 0;
- impact_calibrated — 1, если уровень влияния соответствует тому, насколько событие касается субъекта, иначе 0;
- overall — 1–5;
- unsupported — утверждения ответа, которых нет в контексте, через «;» (можно пусто);
- comment — главный недостаток одной фразой.

Пустую строку можно оставить — она не войдёт в оценку. Не открывайте `key.json`: там названия систем. Порядок \
ответов случайный. Когда закончите: `refract eval label import --out <эксперимент>`.
""",
    "en": """# Labelling the answers

Open `items/<item>.md`: it holds the context the service had, its answer and the criteria. Fill in row `<item>` of \
`sheet.csv`:

- {criteria} — 0 (missing or poor), 1 (partly), 2 (good);
- retelling — 1 if the answer only retells the news, else 0;
- impact_calibrated — 1 if the impact level matches how much the event concerns the subject, else 0;
- overall — 1–5;
- unsupported — statements of the answer absent from the context, separated by “;” (may be empty);
- comment — the main weakness in one sentence.

A row may stay empty — it is left out. Do not open `key.json`: it names the systems. The order of the answers is \
random. When done: `refract eval label import --out <experiment>`.
""",
}


def _folder(out: Path, name: str) -> Path:
    return out / "labels" / _safe(name)


def export_labels(
    out: str | Path,
    pairs: list[Pair],
    systems: Iterable[str] | None = None,
    *,
    sample: int | None = None,
    seed: int = 0,
    name: str = "human",
    institution: Any = None,
    lang: str = "ru",
    force: bool = False,
) -> Path:
    """Write the sheet for the answers of `systems` (default: every system that was run). With `sample`, a random
    sample of that many answers, the same for the same seed. Refuses to overwrite a sheet unless `force`."""
    out = Path(out)
    folder = _folder(out, name)
    if (folder / "sheet.csv").exists() and not force:
        raise FileExistsError(
            f"{folder / 'sheet.csv'} exists — it may hold labels; use --force to replace it"
        )
    ran = sorted(p.name for p in (out / "runs").iterdir() if p.is_dir()) if (out / "runs").exists() else []
    systems = [s for s in (systems or ran) if s in ran]
    items = [
        (system, pair)
        for system in systems
        for pair in pairs
        if ((_read(out / "runs" / system / f"{pair.id}.json") or {}).get("content"))
    ]
    if not items:
        raise ValueError("no answers to label: run `refract eval run` first")
    rng = random.Random(seed)
    rng.shuffle(items)
    if sample is not None:
        items = items[: max(0, sample)]
    width = len(str(len(items)))
    key: dict[str, dict[str, str]] = {}
    (folder / "items").mkdir(parents=True, exist_ok=True)
    for i, (system, pair) in enumerate(items, 1):
        item = f"i{i:0{width}d}"
        data = _read(out / "runs" / system / f"{pair.id}.json") or {}
        plang = pair.language
        names = (
            {"institution": institution.display(plang), "institution_gen": institution.genitive("ru")}
            if institution
            else None
        )
        rubric, page = jp.judge_messages(
            data["context"], jp.render_answer(data["content"], plang, names), plang, names
        )
        (folder / "items" / f"{item}.md").write_text(
            f"# {item}\n\n{page['content'].rsplit(chr(10) * 2, 1)[0]}\n\n---\n\n{rubric['content']}\n",  # no JSON task
            encoding="utf-8",
        )
        key[item] = {"system": system, "pair": pair.id}
    _write(folder / "key.json", {"seed": seed, "sample": sample, "items": key})
    with (folder / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter="," if lang == "en" else ";")  # Excel in a Russian locale splits on “;”
        w.writerow(COLUMNS)
        for item in key:
            w.writerow([item, *[""] * (len(COLUMNS) - 1)])
    readme = _README["en" if lang == "en" else "ru"].format(criteria=", ".join(jp.CRITERIA))
    (folder / "README.md").write_text(readme, encoding="utf-8")
    return folder


def _int(value: str, low: int, high: int, column: str) -> int:
    try:
        x = int(float(value.strip().replace(",", ".")))
    except ValueError:
        raise ValueError(f"{column}: “{value}” is not a number") from None
    if not low <= x <= high:
        raise ValueError(f"{column}: {x} is outside {low}–{high}")
    return x


def _score(row: dict[str, str]) -> dict[str, Any]:
    comment = (row.get("comment") or "").strip()
    score = {
        "unsupported": [s.strip() for s in (row.get("unsupported") or "").split(";") if s.strip()][:6],
        **{c: {"note": comment, "score": _int(row.get(c) or "", 0, 2, c)} for c in jp.CRITERIA},
        **{f: bool(_int(row.get(f) or "", 0, 1, f)) for f in FLAGS},
        "overall": _int(row.get("overall") or "", 1, 5, "overall"),
        "comment": comment,
    }
    return jp.JudgeScore.model_validate(score).model_dump(mode="json")


def import_labels(
    out: str | Path, name: str = "human", sheet: str | Path | None = None
) -> tuple[int, int, list[str]]:
    """Read the filled sheet into `judge/<name>/<system>/<pair>.json`. Returns (saved, empty rows, problems); a row
    with a problem is not saved, the others are."""
    out = Path(out)
    folder = _folder(out, name)
    key = (_read(folder / "key.json") or {}).get("items")
    if not key:
        raise FileNotFoundError(f"{folder / 'key.json'} not found: run `refract eval label export` first")
    path = Path(sheet) if sheet else folder / "sheet.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t") if sample else csv.excel
        rows = list(csv.DictReader(f, dialect=dialect))
    saved, empty, problems = 0, 0, []
    for row in rows:
        item = (row.get("item") or "").strip()
        if not item:
            continue
        if item not in key:
            problems.append(f"{item}: not in key.json")
            continue
        if not any((row.get(c) or "").strip() for c in (*jp.CRITERIA, *FLAGS, "overall")):
            empty += 1
            continue
        try:
            score = _score(row)
        except (ValueError, ValidationError) as e:
            problems.append(f"{item}: {e}")
            continue
        target = key[item]
        _write(
            out / "judge" / _safe(name) / target["system"] / f"{target['pair']}.json",
            {"score": score, "error": None, "cost": 0.0, "judge": name, "judge_version": VERSION},
        )
        saved += 1
    return saved, empty, problems


def is_human(out: str | Path, judge: str) -> bool:
    """A judge whose scores came from `import_labels`."""
    root = Path(out) / "judge" / _safe(judge)
    first = next(root.glob("*/*.json"), None) if root.exists() else None
    return bool(first) and json.loads(first.read_text(encoding="utf-8")).get("judge_version") == VERSION
