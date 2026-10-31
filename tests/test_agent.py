"""The orchestrator on a scripted model: path, revision, pruning, ablations, languages, safety."""

from __future__ import annotations

import pytest

from refract.agent import Agent, AgentConfig, ablation
from refract.demo import INJECTION_DRAFT, OK, RATE_PLAN, _rate_draft, run_demo
from refract.llm import ScriptedLLM
from refract.schemas import ImpactLevel, StepKind
from tests.conftest import NOW


def agent(store, settings, responses, config=None) -> tuple[Agent, ScriptedLLM]:
    llm = ScriptedLLM(responses, settings)
    return Agent(llm, store, settings=settings, config=config, now=lambda: NOW), llm


def test_rate_demo_catches_invented_number_and_wrong_product() -> None:
    run = run_demo("rate")
    assert run.ok and run.trace.status == "done" and run.state.iteration == 1
    names = [s.name for s in run.trace.steps]
    assert names[:2] == ["get_profile", "plan"] and names[-1] == "finalize"
    assert names.count("rule_check") == 2 and "revise" in names
    assert "1,2" not in " ".join(r.text for r in run.insight.risks)
    assert [a.product for a in run.insight.actions] == ["Кредит с фиксированной ставкой"]
    assert {s.source_id for s in run.insight.sources} >= {"src_art_rate_hold_ru", "src_demo_macro_2026q4_t1"}
    assert run.insight.disclaimer and "Демо-банка" in run.insight.disclaimer


def test_injection_demo_is_neutralised() -> None:
    run = run_demo("injection")
    assert run.insight.overall_impact is ImpactLevel.MEDIUM
    assert all(r.impact is not ImpactLevel.HIGH for r in run.insight.risks)
    assert all(a.product is None for a in run.insight.actions)
    rule = next(s for s in run.trace.steps if s.name == "rule_check")
    assert any("предела" in i for i in rule.issues)


def test_unfixable_claim_is_removed(store, settings) -> None:
    bad = {
        "passed": False,
        "issues": [
            {"code": "contradiction", "severity": "error", "claim_id": "c3", "message": "противоречит"}
        ],
    }
    a, _llm = agent(
        store,
        settings,
        {
            "plan": [RATE_PLAN],
            "draft_insight": [_rate_draft(False)],
            "verify": [bad],
            "revise": [_rate_draft(False)],
        },
    )
    run = a.run("demo_building_materials", "ev_rate_hold_ru")
    assert run.ok and "c3" not in {c.claim_id for c in run.insight.all_claims()}
    assert run.trace.steps[-1].verdict == "issues"


def test_required_block_comes_back_with_caveat(store, settings) -> None:
    issues = [
        {"code": "contradiction", "severity": "error", "claim_id": cid, "message": "сомнение"}
        for cid in ("c2", "c3")
    ]
    a, _ = agent(
        store,
        settings,
        {
            "plan": [RATE_PLAN],
            "draft_insight": [_rate_draft(False)],
            "verify": [{"passed": False, "issues": issues}],
            "revise": [_rate_draft(False)],
        },
    )
    run = a.run("demo_building_materials", "ev_rate_hold_ru")
    assert run.insight.why_it_matters and all(
        c.confidence <= 0.5 and c.caveat for c in run.insight.why_it_matters
    )


def test_fresh_news_gets_classified(store, settings) -> None:
    from refract.facade import news_from_text

    event = news_from_text("Юань подорожал", "Курс юаня вырос на 3% за неделю, импортёры платят больше.")
    tags = {
        "is_business_relevant": True,
        "event_type": "fx",
        "macro_factors": ["fx:CNY", "nonsense"],
        "industry_sections": ["g"],
        "industry_codes": ["46", "4"],
        "regions": [],
        "countries": ["CN"],
        "commodities": [],
        "importance": 0.6,
        "key_facts": ["юань вырос на 3%"],
        "language": "ru",
    }
    draft = _rate_draft(False) | {
        "what_happened": [
            {
                "claim_id": "c1",
                "text": "Юань подорожал.",
                "origin": "fact",
                "source_ids": [f"src_{event.articles[0].article_id}"],
                "confidence": 0.9,
            }
        ]
    }
    a, llm = agent(
        store,
        settings,
        {"classify_news": [tags], "plan": [RATE_PLAN], "draft_insight": [draft], "verify": [OK]},
    )
    run = a.run("demo_building_materials", event)
    assert llm.steps[0] == "classify_news"
    ev = run.state.context.event
    assert ev.macro_factors == ["fx:CNY"] and ev.industry_codes == ["46"] and ev.industry_sections == ["G"]


def test_planner_failure_falls_back_to_default_plan(store, settings) -> None:
    a, _ = agent(store, settings, {"plan": [None], "draft_insight": [_rate_draft(False)], "verify": [OK]})
    run = a.run("demo_building_materials", "ev_rate_hold_ru")
    tools = [s.name for s in run.trace.steps if s.kind is StepKind.TOOL_CALL]
    assert tools == ["get_profile", "get_house_view", "get_counterparties", "search_research"]


@pytest.mark.parametrize(
    ("name", "absent"),
    [("no_verifier", "verify"), ("no_rules", "rule_check"), ("no_self_check", "verify")],
)
def test_ablations_skip_components(store, settings, name, absent) -> None:
    a, _llm = agent(
        store,
        settings,
        {
            "plan": [RATE_PLAN],
            "draft_insight": [_rate_draft(True)],
            "verify": [OK],
            "revise": [_rate_draft(False)],
        },
        ablation(name),
    )
    run = a.run("demo_building_materials", "ev_rate_hold_ru")
    assert absent not in [s.name for s in run.trace.steps]
    assert run.trace.config["llm_verifier"] is (name not in ("no_verifier", "no_self_check"))


def test_no_plan_ablation(store, settings) -> None:
    a, llm = agent(
        store, settings, {"draft_insight": [_rate_draft(False)], "verify": [OK]}, ablation("no_plan")
    )
    run = a.run("demo_building_materials", "ev_rate_hold_ru")
    assert "plan" not in llm.steps and run.trace.steps[1].verdict == "skipped"


def test_no_cap_and_no_policy_keep_model_choices(store, settings) -> None:
    plan = {
        "rationale": "r",
        "steps": [{"step_id": "s1", "tool": "get_house_view", "args": {}, "purpose": "p"}],
    }
    cfg = AgentConfig(impact_cap=False, action_policy=False)
    a, _ = agent(store, settings, {"plan": [plan], "draft_insight": [INJECTION_DRAFT], "verify": [OK]}, cfg)
    run = a.run("demo_coffee_ip", "ev_injection_ru", role="entrepreneur")
    assert run.insight.overall_impact is ImpactLevel.HIGH
    assert run.insight.actions[0].product is not None


def test_english_run_uses_english_prompts_and_labels(store, settings) -> None:
    draft = _rate_draft(False)
    draft["headline"] = "The 14% rate stays: cheaper loans are postponed"
    a, llm = agent(store, settings, {"plan": [RATE_PLAN], "draft_insight": [draft], "verify": [OK]})
    run = a.run("demo_building_materials", "ev_rate_hold_ru", language="en")
    system = llm.messages["draft_insight"][0]["content"]
    assert system.startswith("You are an analyst at Demo Bank") and "Write in English" in system
    assert run.insight.language == "en" and "Demo Bank" in run.insight.disclaimer
    assert run.insight.actions[0].product == "Fixed-rate loan"


def test_unknown_profile_reports_failure(store, settings) -> None:
    events = []
    a = Agent(ScriptedLLM({}, settings), store, settings=settings, on_progress=events.append)
    with pytest.raises(KeyError):
        a.run("nobody", "ev_rate_hold_ru")
    assert events[-1]["status"] == "finished" and events[-1]["ok"] is False


def test_progress_events(store, settings) -> None:
    from refract.agent.progress import ProgressTracker

    tracker = ProgressTracker("ev_rate_hold_ru", "demo_building_materials")
    llm = ScriptedLLM({"plan": [RATE_PLAN], "draft_insight": [_rate_draft(False)], "verify": [OK]}, settings)
    Agent(llm, store, settings=settings, on_progress=tracker).run(
        "demo_building_materials", "ev_rate_hold_ru"
    )
    snap = tracker.snapshot()
    assert snap.status == "ready" and snap.insight_id and all(s.status == "done" for s in snap.steps)


def test_dollar_demo_cuts_misattributed_view_and_rewrites_headline() -> None:
    run = run_demo("dollar")
    ins = run.insight
    assert ins.language == "en" and "5%" not in ins.headline
    assert [c.claim_id for c in ins.house_view] == ["b1"]  # b2 cited only the news
    assert ins.actions[0].product == "FX forward"  # named in Russian by the draft, canonical English name
    assert "headline" in [s.name for s in run.trace.steps]


def test_wrong_language_is_flagged_and_restore_keeps_the_run_language(store) -> None:
    from refract.agent import checks
    from refract.agent.context import InsightContext
    from refract.schemas import Claim, InsightContent, Origin

    ctx = InsightContext(
        store.profiles["demo_building_materials"],
        store.events["ev_rate_hold_ru"],
        theses=store.theses,
        lang="en",
    )
    draft = InsightContent.model_validate(_rate_draft(False))  # written in Russian
    assert any(
        i.code.value == "wrong_language" and i.severity == "warning" for i in checks.rule_check(draft, ctx)
    )
    removed = [
        Claim(
            claim_id="b1",
            text="x",
            origin=Origin.HOUSE_VIEW,
            source_ids=["src_demo_macro_2026q4_t1"],
            confidence=0.9,
        )
    ]
    assert (
        checks.house_view_from_theses(ctx, removed) == []
    )  # the thesis is in Russian, the insight in English
    ctx.lang = "ru"
    assert [c.source_ids for c in checks.house_view_from_theses(ctx, removed)] == [
        ["src_demo_macro_2026q4_t1"]
    ]
