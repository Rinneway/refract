"""Robustness to small local models: attribution, reconciling the reviewer, fallbacks, headlines, plan dedupe.

`fixtures/qwen_usn_run.json` holds two real runs of qwen3.5:9b (main model) and gemma4:e4b (verifier) through
local Ollama on the demo pair demo_coffee_ip × ev_usn_vat_ru, every answer in call order:
- run_1 failed in finalize before attribution repair: no “why it matters” statement had `features`;
- run_2 cut statements the reviewer itself called “not a contradiction, only too categorical”, took an extra
  revision round for it, and rejected a good headline because it was 17 characters too long.
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
from refract.textutil import trim_sentence
from tests.conftest import NOW

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "qwen_usn_run.json").read_text(encoding="utf-8"))


def replay(store, settings, name: str = "run_1", config=None, llm_out: list | None = None):
    answers = {
        k: (v if isinstance(v, list) else [v]) for k, v in FIXTURE[name].items() if not k.startswith("_")
    }
    llm = ScriptedLLM(answers, settings)
    agent = Agent(llm, store, settings=settings, config=config, now=lambda: NOW)
    run = agent.run("demo_coffee_ip", "ev_usn_vat_ru", role="entrepreneur")
    if llm_out is not None:
        llm_out.extend(llm.steps)
    return run


def test_real_small_model_run_is_repaired(store, settings) -> None:
    run = replay(store, settings, "run_1")
    assert run.ok and run.trace.status == "done"
    names = [s.name for s in run.trace.steps]
    assert names.count("attribution") == 2  # after the draft and after the revision
    assert "w1" in {
        c.claim_id for c in run.insight.why_it_matters
    }  # “too categorical” is not a reason to cut
    feats = {k for k, _ in profile_features(store.profiles["demo_coffee_ip"], "ru")}
    ins = run.insight
    assert ins.why_it_matters and all(c.features and set(c.features) <= feats for c in ins.why_it_matters)
    # The verifier's “r1 has no sources” is lifted: code attached a source and found no structural problem.
    assert "r1" in {r.claim_id for r in ins.risks}
    assert all(r.source_ids for r in ins.risks)
    assert not any(r.text.lower().startswith("риск") for r in ins.risks)
    verify = [s for s in run.trace.steps if s.name == "verify"][-1]
    assert any("unknown_source" in i and "снято" in i for i in verify.issues)
    # Actions of the model survive; nothing was invented by the fallback.
    assert {a.action_id for a in ins.actions} >= {"a1", "a2"}
    assert "c_link" not in {c.claim_id for c in ins.why_it_matters}


def test_without_attribution_the_fallback_keeps_the_insight(store, settings) -> None:
    run = replay(store, settings, "run_1", ablation("no_attribution"))
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
        {
            "plan": [plan],
            "draft_insight": FIXTURE["run_2"]["draft_insight"],
            "verify": [{"passed": True, "issues": []}],
        },
        settings,
    )
    run = Agent(llm, store, settings=settings, now=lambda: NOW).run("demo_coffee_ip", "ev_usn_vat_ru")
    assert [s.name for s in run.trace.steps].count("get_house_view") == 1


def test_reviewer_wording_remarks_do_not_cut_statements(store, settings) -> None:
    steps: list[str] = []
    run = replay(store, settings, "run_2", llm_out=steps)
    assert steps == ["plan", "draft_insight", "verify"]  # no revision round for remarks on wording
    ins = run.insight
    why = {c.claim_id: c for c in ins.why_it_matters}
    assert set(why) == {"w1", "w2", "w3"}
    assert why["w1"].caveat and why["w1"].confidence <= 0.6
    assert why["w2"].origin is Origin.HYPOTHESIS and why["w2"].caveat  # “this is an assumption”
    assert "o1" in {o.claim_id for o in ins.opportunities}
    assert {a.action_id for a in ins.actions} == {"a1", "a2"}  # the model's own actions, not the generic one
    verify = next(s for s in run.trace.steps if s.name == "verify")
    assert verify.verdict == "ok" and sum("понижено до предупреждения" in i for i in verify.issues) == 4


def test_without_reconcile_the_old_path_still_ends_well(store, settings) -> None:
    steps: list[str] = []
    run = replay(store, settings, "run_2", ablation("no_reconcile"), llm_out=steps)
    assert steps[-1] == "headline" and steps.count("verify") == 2
    ins = run.insight
    # The model's 177-character headline is cut at “, но…”, not replaced by a repeated house view.
    assert ins.headline.startswith("Снижение порога НДС") and len(ins.headline) <= 160
    assert not ins.headline.endswith("…")
    assert all(c.origin is not Origin.HOUSE_VIEW for c in ins.why_it_matters)
    finalize = run.trace.steps[-1]
    assert any("повтор позиции" in n for n in finalize.issues)


def test_trim_sentence_cuts_at_a_clause() -> None:
    long = "Снижение порога может сделать вас плательщиком налога, но для общепита это требует " + "x " * 60
    assert trim_sentence(long, 120) == "Снижение порога может сделать вас плательщиком налога"
    assert trim_sentence("short", 120) == "short"
    assert trim_sentence("word " * 50, 40).endswith("…")
    c = _content().model_dump()
    c["headline"] = long
    assert (
        InsightContent.model_validate(c).headline == "Снижение порога может сделать вас плательщиком налога"
    )


def _finding(code: IssueCode, cid: str, message: str) -> VerificationIssue:
    return VerificationIssue(code=code, severity="error", claim_id=cid, message=message)


def test_reconcile_follows_the_reviewers_own_words(store) -> None:
    from refract.agent.context import InsightContext

    ctx = InsightContext(
        profile=store.profiles["demo_coffee_ip"], event=store.events["ev_usn_vat_ru"], lang="ru"
    )
    content = _content()
    issues = [
        _finding(IssueCode.CONTRADICTION, "w1", "Это не противоречит контексту, но слишком категорично."),
        _finding(IssueCode.CONTRADICTION, "c1", "Это является предположением: данных нет."),
        # Marker words inside a quoted fragment of the statement do not count.
        _finding(
            IssueCode.CONTRADICTION,
            "w1",
            'Утверждается, что "налог смягчит нагрузку". В контексте ставка выше.',
        ),
        _finding(IssueCode.NUMBER_NOT_IN_SOURCES, "w1", "слишком категорично, числа 7 нет"),
        _finding(IssueCode.CONTRADICTION, "headline", "не противоречит, но формулировка слишком сильная"),
    ]
    out, notes, kinds = checks.reconcile(issues, content, ctx)
    assert [i.severity for i in out] == ["warning", "warning", "error", "error", "warning"]
    assert kinds == {"w1": "wording", "c1": "assumption", "headline": "wording"} and len(notes) == 3
    soft, soft_notes = checks.soften(content, kinds, "ru")
    assert soft.what_happened[0].origin is Origin.HYPOTHESIS and soft.why_it_matters[0].caveat and soft_notes


def test_fallback_headline_prefers_the_subject() -> None:
    content = _content()
    house = SubjectClaim(
        claim_id="w0",
        text="Институт ожидает роста",
        origin=Origin.HOUSE_VIEW,
        features=["kind"],
        confidence=0.9,
    )
    content = content.model_copy(update={"why_it_matters": [house, *content.why_it_matters]})
    assert checks.short_headline(content) == "y"


def test_action_keeps_its_remaining_support() -> None:
    draft = _content()
    two = Action(action_id="a3", text="t", kind=ActionKind.ANALYSIS, rationale="r", claim_ids=["w1", "c1"])
    draft = draft.model_copy(update={"actions": [*draft.actions, two]})
    pruned = checks.drop_claims(draft, {"w1"})
    assert [(a.action_id, a.claim_ids) for a in pruned.actions] == [("a3", ["c1"])]
