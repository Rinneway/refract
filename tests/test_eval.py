import math

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
    assert S["agreement"]["n"] == 8 and "gold_impact_accuracy" in full
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
