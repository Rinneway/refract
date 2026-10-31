"""Robustness to small local models: attribution repair, structural overrules, fallbacks, plan dedupe.

`fixtures/qwen_usn_run.json` holds the real answers of qwen3.5:9b (main model) and gemma4:e4b (verifier) on the
demo pair demo_coffee_ip × ev_usn_vat_ru. Before attribution repair this run failed in finalize: every
“why it matters” statement lacked `features`, so the block was pruned to nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from refract.agent import Agent, ablation, checks
from refract.agent.context import profile_features
from refract.llm import ScriptedLLM
from refract.schemas import (
    Action,
    ActionKind,
    Claim,
    Direction,
    ImpactLevel,
    InsightContent,
    IssueCode,
    Origin,
    SubjectClaim,
    VerificationIssue,
)
from tests.conftest import NOW

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "qwen_usn_run.json").read_text(encoding="utf-8"))


def replay(store, settings, config=None):
    llm = ScriptedLLM(
        {
            "plan": [FIXTURE["plan"]],
            "draft_insight": [FIXTURE["draft"]],
            "verify": [FIXTURE["verification"]],
            "revise": [FIXTURE["draft"]],
        },
        settings,
    )
    agent = Agent(llm, store, settings=settings, config=config, now=lambda: NOW)
    return agent.run("demo_coffee_ip", "ev_usn_vat_ru", role="entrepreneur")


def test_real_small_model_run_is_repaired(store, settings) -> None:
    run = replay(store, settings)
    assert run.ok and run.trace.status == "done"
    names = [s.name for s in run.trace.steps]
    assert names.count("attribution") == 2  # after the draft and after the revision
    feats = {k for k, _ in profile_features(store.profiles["demo_coffee_ip"], "ru")}
    ins = run.insight
    assert ins.why_it_matters and all(c.features and set(c.features) <= feats for c in ins.why_it_matters)
    # The verifier's “r1 has no sources” is lifted: code attached a source and found no structural problem.
    assert "r1" in {r.claim_id for r in ins.risks}
    assert all(r.source_ids for r in ins.risks)
    assert not any(r.text.lower().startswith("риск") for r in ins.risks)
    verify = next(s for s in run.trace.steps if s.name == "verify")
    assert any("unknown_source" in i and "снято" in i for i in verify.issues)
    # Actions of the model survive; nothing was invented by the fallback.
    assert {a.action_id for a in ins.actions} >= {"a1", "a2"}
    assert "c_link" not in {c.claim_id for c in ins.why_it_matters}


def test_without_attribution_the_fallback_keeps_the_insight(store, settings) -> None:
    run = replay(store, settings, ablation("no_attribution"))
    assert run.ok and "attribution" not in [s.name for s in run.trace.steps]
    (link,) = run.insight.why_it_matters
    assert link.claim_id == "c_link" and link.features == ["taxes"] and link.caveat
    assert link.confidence <= 0.5 and "src_profile" in link.source_ids


def test_infer_features_uses_concept_words(store) -> None:
    feats = profile_features(store.profiles["demo_coffee_ip"], "ru")
    assert set(checks.infer_features("Вы станете плательщиком НДС", feats)) == {"tax_regime", "taxes"}
    assert checks.infer_features("При выручке 18 млн рублей вы превысите порог", feats) == ["scale.revenue"]
    assert "counterparties.supplier.cp_roaster_77" in checks.infer_features(
        "Поставщик кофе работает без НДС", feats
    )
    assert checks.infer_features("Погода в выходные будет солнечной", feats) == []


def test_subject_claim_tolerates_missing_features_but_schema_requires_them() -> None:
    c = SubjectClaim.model_validate({"claim_id": "w", "text": "x", "origin": "ai", "confidence": 0.7})
    assert c.features == []
    assert "features" in SubjectClaim.model_json_schema()["required"]


def _issue(code: IssueCode, cid: str) -> VerificationIssue:
    return VerificationIssue(code=code, severity="error", claim_id=cid, message="m")


def test_structural_findings_need_rule_confirmation() -> None:
    llm = [
        _issue(IssueCode.UNKNOWN_SOURCE, "r1"),
        _issue(IssueCode.UNKNOWN_SOURCE, "r2"),
        _issue(IssueCode.CONTRADICTION, "w1"),
    ]
    rule = [_issue(IssueCode.UNKNOWN_SOURCE, "r2")]
    out, notes = checks.overrule_structural(llm, rule, "en")
    assert [i.severity for i in out] == ["warning", "error", "error"]
    assert len(notes) == 1 and "r1" in notes[0]


def _content() -> InsightContent:
    fact = Claim(claim_id="c1", text="x", origin=Origin.FACT, source_ids=["src_a"], confidence=0.9)
    why = SubjectClaim(claim_id="w1", text="y", origin=Origin.AI, features=["kind"], confidence=0.8)
    acts = [
        Action(action_id="a1", text="t", kind=ActionKind.ANALYSIS, rationale="r", claim_ids=["w1"]),
        Action(action_id="a2", text="t", kind=ActionKind.ANALYSIS, rationale="r", claim_ids=["w1"]),
    ]
    return InsightContent(
        headline="h",
        what_happened=[fact],
        why_it_matters=[why],
        actions=acts,
        overall_impact=ImpactLevel.LOW,
        overall_direction=Direction.MIXED,
    )


def test_restored_block_brings_back_its_actions() -> None:
    draft = _content()
    bad = {"w1", "a2"}
    pruned, notes = checks.keep_required(checks.drop_claims(draft, bad), draft, set(), "en", dropped=bad)
    assert notes and pruned.why_it_matters[0].caveat
    assert [a.action_id for a in pruned.actions] == ["a1"]  # a2 was flagged itself


def test_duplicate_plan_steps_run_once(store, settings) -> None:
    step = {"tool": "get_house_view", "args": {"factors": ["taxes"]}, "purpose": "p"}
    plan = {
        "rationale": "r",
        "steps": [{**step, "step_id": "s1"}, {**step, "step_id": "s2"}, {**step, "step_id": "s3"}],
    }
    llm = ScriptedLLM(
        {"plan": [plan], "draft_insight": [FIXTURE["draft"]], "verify": [{"passed": True, "issues": []}]},
        settings,
    )
    run = Agent(llm, store, settings=settings, now=lambda: NOW).run("demo_coffee_ip", "ev_usn_vat_ru")
    assert [s.name for s in run.trace.steps].count("get_house_view") == 1
