import math
import re

import pytest

from refract.demo import OK, RATE_PLAN, _rate_draft
from refract.eval import Experiment, Pair, write_report
from refract.eval.stats import bootstrap_ci, sign_test, spearman, weighted_kappa
from refract.facade import Refract
from refract.llm import ScriptedLLM

BASELINE = {
    "headline": "Ставка 14%",
    "what_happened": ["Ставку сохранили"],
    "why_it_matters": ["Кредиты не подешевеют"],
    "risks": [],
    "opportunities": [],
    "scenarios": [],
    "house_view": [],
    "actions": ["Следить за ставкой"],
    "overall_impact": "high",
    "overall_direction": "negative",
}


def score(total_each: int) -> dict:
    c = {"note": "ok", "score": total_each}
    return {
        "unsupported": [],
        "what_happened": c,
        "why_it_matters": c,
        "consequences": c,
        "actions": c,
        "house_view": c,
        "grounding": c,
        "retelling": False,
        "impact_calibrated": True,
        "overall": 4,
        "comment": "—",
    }


def test_stats() -> None:
    assert weighted_kappa([0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2]) == 1.0
    assert weighted_kappa([0, 0, 2, 2], [2, 2, 0, 0], [0, 1, 2]) < 0
    assert sign_test(10, 0) < 0.01 and sign_test(3, 3) == 1.0
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    lo, hi = bootstrap_ci([1, 2, 3, 4, 5])
    assert lo <= 3 <= hi and math.isnan(bootstrap_ci([1])[0])


def test_experiment_end_to_end(store, settings, tmp_path) -> None:
    def judge(messages, schema):
        return score(2 if "Демо-банк ожидает" in messages[-1]["content"] else 1)

    llm = ScriptedLLM(
        {
            "plan": [RATE_PLAN],
            "draft_insight": [_rate_draft(False)],
            "verify": [OK],
            "eval_single_call": [_rate_draft(True)],
            "eval_retelling": [BASELINE],
            "*": [judge],
        },
        settings,
    )
    rf = Refract(store, llm=llm, settings=settings)
    pairs = [
        Pair("demo_building_materials", "ev_rate_hold_ru", "cfo", "ru"),
        Pair("demo_building_materials", "ev_rate_hold_ru", "cfo", "en", gold_impact="high"),
    ]
    exp = Experiment(rf, tmp_path, pairs, workers=1, log=lambda _: None)
    exp.run(["full", "no_cap", "single_call", "retelling"])
    exp.judge(["verifier", "main"])
    S = write_report(tmp_path, pairs, ["verifier", "main"], "full")
    full, single, retell = S["systems"]["full"], S["systems"]["single_call"], S["systems"]["retelling"]
    assert full["answered"] == 2 and full["rule_error_rate"] == 0.0
    assert (
        single["rule_error_rate"] == 1.0 and single["invented_numbers_mean"] >= 1
    )  # the invented 1,2 bn reaches the reader
    assert retell["rule_error_rate"] is None and retell["over_cap_rate"] == 0.0
    assert full["judges"]["verifier"]["total_mean"] == 12 and retell["judges"]["verifier"]["total_mean"] == 6
    assert S["pairwise"]["retelling"]["reference_wins"] == 2
    assert [a["judges"] for a in S["agreement"]] == [["verifier", "main"]] and S["agreement"][0]["n"] == 8
    assert "gold_impact_accuracy" in full
    langs = full["by_language"]
    assert (
        langs["ru"]["language_ok_rate"] == 1.0 and langs["en"]["language_ok_rate"] == 0.0
    )  # scripted drafts are Russian
    assert full["language_parity"]["n"] == 1
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert (
        "`full`" in report
        and "Agreement" in report
        and "By output language" in report
        and "en − ru" in report
        and "## Process" in report
    )
    assert full["revisions_mean"] is not None and full["reviewer_errors_mean"] is not None
    assert retell["revisions_mean"] is None  # baselines have no agent process
    exp.run(["full"])  # resumable: nothing is recomputed


def test_judge_stops_on_a_rejected_key_and_saves_nothing(store, settings, tmp_path) -> None:
    from refract.llm import LLMError

    calls = []

    def unauthorised(messages, schema):
        calls.append(1)
        raise LLMError("gemma4:31b: HTTP 401: Error code: 401 - Unauthorized")

    llm = ScriptedLLM(
        {"plan": [RATE_PLAN], "draft_insight": [_rate_draft(False)], "verify": [OK], "*": [unauthorised]},
        settings,
    )
    rf = Refract(store, llm=llm, settings=settings)
    pairs = [Pair("demo_building_materials", "ev_rate_hold_ru", "cfo", lang) for lang in ("ru", "en")]
    log: list[str] = []
    exp = Experiment(rf, tmp_path, pairs, workers=1, log=log.append)
    exp.run(["full"])
    log.clear()
    exp.judge(["judge2", "verifier"])  # systems: only what was run (no single_call / retelling noise)
    assert len(calls) == 1 and not (tmp_path / "judge").exists()
    assert sum("stopped" in m for m in log) == 2 and not any("single_call" in m for m in log)


def test_judge_stops_without_credits(store, settings, tmp_path) -> None:
    from refract.llm import LLMError

    def no_credits(messages, schema):
        raise LLMError("gemma4:31b: HTTP 402: Add usage credits to use cloud models")

    llm = ScriptedLLM(
        {"plan": [RATE_PLAN], "draft_insight": [_rate_draft(False)], "verify": [OK], "*": [no_credits]},
        settings,
    )
    log: list[str] = []
    exp = Experiment(
        Refract(store, llm=llm, settings=settings),
        tmp_path,
        [Pair("demo_building_materials", "ev_rate_hold_ru", "cfo", "ru")],
        workers=1,
        log=log.append,
    )
    exp.run(["full"])
    exp.judge(["judge"])
    assert any("credits or a plan" in m for m in log) and not (tmp_path / "judge").exists()


def test_experiment_warns_when_models_change(store, settings, tmp_path) -> None:
    import json as _json

    llm = ScriptedLLM({"plan": [RATE_PLAN], "draft_insight": [_rate_draft(False)], "verify": [OK]}, settings)
    pairs = [Pair("demo_building_materials", "ev_rate_hold_ru", "cfo", "ru")]
    log: list[str] = []
    exp = Experiment(Refract(store, llm=llm, settings=settings), tmp_path, pairs, workers=1, log=log.append)
    exp.run(["full"])
    assert any(m.startswith("models: main=") for m in log) and (tmp_path / "manifest.json").exists()
    manifest = _json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["models"]["verifier"] = "gemma4:12b"  # the experiment was started with another reviewer
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest), encoding="utf-8")
    assert any("verifier was gemma4:12b" in w for w in exp.check_manifest())
    S = write_report(tmp_path, pairs, [], "full")
    assert "verify" in S["systems"]["full"]["models"]
    assert "## Models" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_manifest_of_an_older_folder_comes_from_its_traces(store, settings, tmp_path) -> None:
    llm = ScriptedLLM({"plan": [RATE_PLAN], "draft_insight": [_rate_draft(False)], "verify": [OK]}, settings)
    pairs = [Pair("demo_building_materials", "ev_rate_hold_ru", "cfo", "ru")]
    exp = Experiment(
        Refract(store, llm=llm, settings=settings), tmp_path, pairs, workers=1, log=lambda _: None
    )
    exp.run(["full"])
    (tmp_path / "manifest.json").unlink()  # a folder from before manifests existed
    assert exp.check_manifest() == [] and (tmp_path / "manifest.json").exists()


def test_human_labels_round_trip(store, settings, tmp_path, capsys) -> None:
    """export a blinded sheet → an annotator fills it in → import as judge `human` → the report compares judges."""
    import csv
    import json

    from refract.cli import main
    from refract.eval import save_pairs

    llm = ScriptedLLM(
        {
            "plan": [RATE_PLAN],
            "draft_insight": [_rate_draft(False)],
            "verify": [OK],
            "eval_retelling": [BASELINE],
            "*": [lambda m, s: score(2 if "Демо-банк ожидает" in m[-1]["content"] else 1)],
        },
        settings,
    )
    pairs = [
        Pair("demo_building_materials", "ev_rate_hold_ru", "cfo", "ru"),
        Pair("demo_building_materials", "ev_rate_hold_ru", "owner", "ru"),
    ]
    save_pairs(pairs, tmp_path / "pairs.jsonl")
    exp = Experiment(
        Refract(store, llm=llm, settings=settings), tmp_path, pairs, workers=1, log=lambda _: None
    )
    exp.run(["full", "retelling"])
    exp.judge(["verifier"])

    assert main(["eval", "label", "export", "--out", str(tmp_path), "--seed", "3"]) == 0
    folder = tmp_path / "labels" / "human"
    key = json.loads((folder / "key.json").read_text(encoding="utf-8"))["items"]
    page = (folder / "items" / "i1.md").read_text(encoding="utf-8")
    assert len(key) == 4 and "# КОНТЕКСТ" in page and "JSON" not in page.split("---")[0]
    head = page.split("\n---\n")[0]  # the context and the answer; the rubric follows
    assert not re.search(r"\b(full|retelling|single_call|no_\w+)\b", head)  # blinded: no system names
    assert main(["eval", "label", "export", "--out", str(tmp_path)]) == 2  # never overwrites labels silently

    with (folder / "sheet.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))  # a Russian sheet, for Excel in a Russian locale
    for row in rows:  # the annotator agrees with the judge on everything but one answer, leaves one row empty
        agent = key[row["item"]]["system"] == "full"
        row.update({c: "2" if agent else "1" for c in ("what_happened", "why_it_matters", "consequences")})
        row.update({c: "2" if agent else "0" for c in ("actions", "house_view", "grounding")})
        row.update(retelling="0" if agent else "1", impact_calibrated="1", overall="4" if agent else "2")
    rows[-1].update({c: "" for c in rows[-1] if c != "item"})
    rows[0]["overall"] = "7"
    with (folder / "sheet.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )  # saved by another tool with commas: import detects it
        w.writeheader()
        w.writerows(rows)

    capsys.readouterr()
    assert main(["eval", "label", "import", "--out", str(tmp_path)]) == 1  # one row has a problem
    out = capsys.readouterr().out
    assert (
        "overall: 7 is outside 1–5" in out and "2 labels saved as judge `human`; 1 rows empty, 1 with" in out
    )
    assert main(["eval", "report", "--out", str(tmp_path)]) == 0
    S = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert S["judges"] == ["verifier", "human"] and S["humans"] == ["human"]  # the LLM judge stays primary
    assert [a["judges"] for a in S["agreement"]] == [["human", "verifier"]] and S["agreement"][0]["n"] == 2
    assert "human labels" in (tmp_path / "report.md").read_text(encoding="utf-8")

    # a second annotator: the same seed gives the same items; the report adds both annotators' agreement
    assert main(["eval", "label", "export", "--out", str(tmp_path), "--seed", "3", "--name", "ann2"]) == 0
    key2 = json.loads((tmp_path / "labels" / "ann2" / "key.json").read_text(encoding="utf-8"))["items"]
    assert key2 == key
    (tmp_path / "labels" / "ann2" / "sheet.csv").write_text(
        (folder / "sheet.csv").read_text(encoding="utf-8-sig"), encoding="utf-8-sig"
    )
    assert main(["eval", "label", "import", "--out", str(tmp_path), "--name", "ann2"]) == 1
    assert main(["eval", "report", "--out", str(tmp_path)]) == 0
    S = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert [a["judges"] for a in S["agreement"]] == [
        ["ann2", "human"],
        ["ann2", "verifier"],
        ["human", "verifier"],
    ]
    assert S["agreement"][0]["criteria"]["actions"]["exact"] == 1.0


def test_pairs_sample_is_reproducible(tmp_path, capsys) -> None:
    from refract.cli import main
    from refract.eval import load_pairs
    from tests.conftest import DEMO

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for path in (a, b):
        assert (
            main(["eval", "pairs", "--data", str(DEMO), "--out", str(path), "--sample", "5", "--seed", "7"])
            == 0
        )
    assert load_pairs(a) == load_pairs(b) and len(load_pairs(a)) == 5
    assert "5 pairs of " in capsys.readouterr().out
