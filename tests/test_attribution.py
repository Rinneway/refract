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
from refract.config import Settings
from refract.facade import Refract
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
from refract.textutil import invented_numbers, trim_sentence
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


EN_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "qwen_dollar_en_run.json").read_text(encoding="utf-8")
)


def test_percent_must_be_a_percent_in_the_context() -> None:
    context = "Опубликовано 2026-10-20. Ставка 5% без вычетов."
    assert invented_numbers(["общая ставка 20%"], context) == {20.0}
    assert invented_numbers(["ставка 5% без вычетов"], context) == set()
    assert invented_numbers(["20 октября"], context) == set()


def test_revision_sees_the_softened_remarks_and_the_invented_percent(store, settings) -> None:
    llm = ScriptedLLM({k: v if isinstance(v, list) else [v] for k, v in FIXTURE["run_3"].items()}, settings)
    run = Agent(llm, store, settings=settings, now=lambda: NOW).run(
        "demo_coffee_ip", "ev_usn_vat_ru", role="entrepreneur"
    )
    rule = next(s for s in run.trace.steps if s.name == "rule_check")
    assert any("20.0" in i for i in rule.issues)  # “общая ставка 20%”: the 20 of a date does not count
    findings = llm.messages["revise"][-1]["content"]
    assert "[warning] w1: contradiction" in findings and "Смягчи" in findings
    hyp = {c.claim_id for c in run.insight.all_claims() if c.origin is Origin.HYPOTHESIS}
    assert hyp == {"w2", "o1"}  # “в профиле нет данных…” — an assumption, labelled as such


def test_english_run_maps_keys_cites_numbers_and_drops_a_non_effect(store) -> None:
    settings = Settings(api_key="scripted", language="en")
    r = EN_FIXTURE["run_1"]
    llm = ScriptedLLM(
        {k: v if isinstance(v, list) else [v] for k, v in r.items() if not k.startswith("_")}, settings
    )
    run = Agent(llm, store, settings=settings, now=lambda: NOW).run(
        "demo_building_materials", "ev_dollar_en", role="cfo", language="en"
    )
    attribution = next(s for s in run.trace.steps if s.name == "attribution")
    assert any("'sector'] → [" in i and "'industry'" in i for i in attribution.issues)
    assert any(i.startswith("w2:") and "src_demo_macro_2026q4_t1" in i for i in attribution.issues)  # its 14%
    ins = run.insight
    assert "o1" not in {o.claim_id for o in ins.opportunities}  # “…this is not a direct risk”
    assert len(ins.headline) <= 160 and not ins.headline.endswith("…")


def test_resolve_feature_aliases(store) -> None:
    feats = profile_features(store.profiles["demo_building_materials"], "en")
    assert checks.resolve_feature("sector", feats) == "industry"
    assert checks.resolve_feature("loans", feats) == "products.loans"
    assert checks.resolve_feature("cp_dev_101", feats) == "counterparties.customer.cp_dev_101"
    assert checks.resolve_feature("weather", feats) is None


def test_trim_sentence_cuts_at_a_bare_while() -> None:
    text = (
        "Demo Bank expects the dollar to stay strong, raising your import costs while warning of " + "x " * 50
    )
    assert (
        trim_sentence(text, 160) == "Demo Bank expects the dollar to stay strong, raising your import costs"
    )


def test_stronger_reviewer_run_keeps_the_key_conclusion(store, settings) -> None:
    # gemma4:12b: “не свершившийся факт” + “число не из источников” about 18 млн, which is in the subject data.
    answers = {
        k: v if isinstance(v, list) else [v] for k, v in FIXTURE["run_4"].items() if not k.startswith("_")
    }
    llm = ScriptedLLM(answers, settings)
    run = Agent(llm, store, settings=settings, now=lambda: NOW).run(
        "demo_coffee_ip", "ev_usn_vat_ru", role="entrepreneur"
    )
    why = {c.claim_id: c for c in run.insight.why_it_matters}
    assert "w1" in why and why["w1"].caveat
    # Its real catch stays: a1 compared VAT 5% with “УСН доходы” while the subject is on “доходы минус расходы”.
    assert "a1" not in {a.action_id for a in run.insight.actions}
    verify = [s for s in run.trace.steps if s.name == "verify"][-1]
    assert any("number_not_in_sources" in i and "снято" in i for i in verify.issues)


def test_number_findings_need_the_number_check() -> None:
    llm = [_issue(IssueCode.NUMBER_NOT_IN_SOURCES, "w1"), _issue(IssueCode.NUMBER_NOT_IN_SOURCES, "w2")]
    rule = [_issue(IssueCode.NUMBER_NOT_IN_SOURCES, "w2")]
    out, _ = checks.overrule_structural(llm, rule, "ru")
    assert [i.severity for i in out] == ["warning", "error"]


# real answers recorded in October 2026; the regression cases below were found in them
SMOKE = json.loads(
    (Path(__file__).parent / "fixtures" / "smoke_2026-10_qwen9b_gemma12b.json").read_text(encoding="utf-8")
)


def smoke(store, key: str):
    profile, event, role, lang = key.split("__")[:4]
    settings = Settings(api_key="scripted", language=lang)
    llm = ScriptedLLM({k: v if isinstance(v, list) else [v] for k, v in SMOKE[key].items()}, settings)
    return Agent(llm, store, settings=settings, now=lambda: NOW).run(profile, event, role=role, language=lang)


def test_uk_subject_gets_no_russian_rate_forecast(store) -> None:
    run = smoke(store, "demo_london_studio__ev_dollar_en__cfo__en")
    cited = {s for c in run.insight.all_claims() for s in c.source_ids}
    assert not any(s.startswith("src_demo_macro_") for s in cited)
    assert "w2" not in {c.claim_id for c in run.insight.why_it_matters}  # “rates stay at 14%” — Russia's rate


def test_house_view_gets_its_thesis_back_across_languages(store) -> None:
    run = smoke(store, "demo_building_materials__ev_yuan_ru__cfo__en")
    house = {c.claim_id: c.source_ids for c in run.insight.house_view}
    # English statement, Russian thesis: matched by 11.5–12.5; the news' 11.35 does not block it. b2, the inflation
    # forecast, is cut: a thesis on inflation is not about the yuan (off_topic).
    assert house == {"b1": ["src_demo_macro_2026q4_t5"]}
    # w2 “your largest supplier (14%)” — the 22% one is cement: a wrong fact about the subject, cut with the action
    # that rested on it alone (the relabelling of such actions is tested in test_actions.py)
    assert "w2" not in {c.claim_id for c in run.insight.why_it_matters}
    assert [a.action_id for a in run.insight.actions] == ["a1"]


def test_product_named_in_the_text_is_made_explicit(store) -> None:
    run = smoke(store, "demo_grain_farm__ev_wheat_en__cfo__en")
    a2 = next(a for a in run.insight.actions if a.action_id == "a2")
    assert a2.product == "Factoring"  # “Apply for Factoring…” with product = null; the rule did fire
    assert "b2" not in {
        c.claim_id for c in run.insight.house_view
    }  # the UK-exporters thesis is not for a farm


def test_revision_is_a_patch_not_a_rewrite(store) -> None:
    # The reviser answered a single (false) finding on c1 with 324 tokens: no risks, house view or actions left.
    run = smoke(store, "demo_building_materials__ev_yuan_ru__cfo__en__rerun")
    revise = next(s for s in run.trace.steps if s.name == "revise")
    assert any("kept as they were" in i for i in revise.issues)
    ins = run.insight
    assert {r.claim_id for r in ins.risks} == {"r1", "r2"} and ins.opportunities and len(ins.house_view) == 1
    assert {a.action_id for a in ins.actions} == {"a1", "a2"}
    t = run.trace.totals
    assert t.restored >= 7 and t.pruned >= 1 and t.reviewer_errors >= 1  # b2 is off-topic: cut, not restored


def test_house_view_off_the_topic_of_the_news_is_cut(store) -> None:
    """Found in a local run: the bank's views on the VAT threshold and on soft loans attached to news about coffee
    shops closing for lack of demand. Only the view on spending in cafés stays."""
    from refract.agent.impact import thesis_fits

    theses = {th.thesis_id: th for th in store.theses}
    event = store.events["ev_injection_ru"]
    assert thesis_fits(theses["demo_sme_2026_t3"], event) is True  # consumer demand, catering
    assert (
        thesis_fits(theses["demo_sme_2026_t1"], event) is False
    )  # VAT threshold, even though tagged catering
    assert thesis_fits(theses["demo_sme_2026_t4"], event) is False  # soft loans
    run = Refract.demo().analyze("demo_coffee_ip", "ev_injection_ru", role="entrepreneur")
    assert [c.source_ids for c in run.insight.house_view] == [["src_demo_sme_2026_t3"]]
    rules = [i for s in run.trace.steps if s.name == "rule_check" for i in s.issues]
    assert any("b1:" in i and "taxes" in i for i in rules)


def test_merge_revision_respects_flags_and_renames() -> None:
    old = _content()
    a2 = Action(
        action_id="a2",
        text="Сравните налоговые режимы",
        kind=ActionKind.ANALYSIS,
        rationale="r",
        claim_ids=["w1"],
    )
    old = old.model_copy(update={"actions": [old.actions[0], a2]})
    renamed = SubjectClaim(claim_id="w9", text="y", origin=Origin.AI, features=["kind"], confidence=0.8)
    new = old.model_copy(update={"why_it_matters": [renamed], "actions": []})
    merged, restored = checks.merge_revision(old, new, flagged={"a2"})
    assert [c.claim_id for c in merged.why_it_matters] == ["w9"]  # w1 lives on as w9
    assert restored == []  # the actions relied on w1, now w9; a2 was flagged anyway
    merged, restored = checks.merge_revision(old, old.model_copy(update={"actions": old.actions[:1]}), set())
    assert restored == ["a2"] and [a.action_id for a in merged.actions] == ["a1", "a2"]


def test_trace_counts_what_code_did_with_the_review(store, settings) -> None:
    run = replay(store, settings, "run_2")
    t = run.trace.totals
    assert t.reviewer_errors == 4 and t.reviewer_errors_lifted == 4 and t.pruned == 0 and t.cached_calls == 0


def test_largest_counterparty_must_have_the_largest_share(store) -> None:
    """gemma4:12b on the yuan news: “the largest supplier (14%)” — the importer, while cement has 22%."""
    from refract.agent.context import InsightContext

    def check(text: str, lang: str = "en") -> list[str]:
        ctx = InsightContext(
            profile=store.profiles["demo_building_materials"], event=store.events["ev_yuan_ru"], lang=lang
        )
        return [i.message for i in checks.counterparty_superlatives(text, "w2", ctx)]

    real = (
        "Your largest supplier accounts for 14% of your outgoing payments; if they bill in yuan, costs fall."
    )
    assert check(real) == [
        "w2: “largest supplier … 14%” does not match the profile: 14% is the share of supplier "
        "cp_import_309, the largest supplier is cp_cement_205 (22%)"
    ]
    assert check("Ваш крупнейший поставщик-импортёр (14% исходящих платежей) закупает в юанях.", "ru")
    assert "доля покупателя cp_dev_101" in check("Крупнейший поставщик (31% платежей) — застройщик.", "ru")[0]
    # Not flagged: the right share, a plural, “one of”, a percentage that is no counterparty's share, no share
    for ok in (
        "Your largest customer (developer) accounts for 31% of incoming payments.",
        "Ваш крупнейший поставщик (22% исходящих платежей) — производитель цемента.",
        "Your main suppliers (14% and 22% of payments) may raise prices.",
        "One of the largest suppliers, with 14% of payments, imports from China.",
        "Sales of your largest customer fell 20% year on year.",
        "Your largest supplier may raise prices.",
    ):
        assert check(ok, "ru" if "Ваш" in ok else "en") == [], ok


def test_wrong_superlative_is_a_hard_error_of_the_rule_check(store) -> None:
    from refract.agent.context import InsightContext

    ctx = InsightContext(
        profile=store.profiles["demo_building_materials"], event=store.events["ev_yuan_ru"], lang="en"
    )
    content = _content().model_copy(
        update={"headline": "Cheaper yuan lowers costs at your largest supplier (14% of payments)"}
    )
    found = [i for i in checks.rule_check(content, ctx) if "does not match the profile" in i.message]
    assert [(i.claim_id, i.severity) for i in found] == [(checks.HEADLINE, "error")]


def test_advice_must_fit_the_kind_of_subject(store) -> None:
    from refract.agent.context import InsightContext
    from refract.schemas.insight import Action, ActionKind

    def unfit(profile_id: str, *texts: str, lang: str = "ru") -> list[str]:
        ctx = InsightContext(
            profile=store.profiles[profile_id], event=store.events["ev_rate_hold_ru"], lang=lang
        )
        actions = [
            Action(action_id=f"a{i}", text=t, kind=ActionKind.ANALYSIS, rationale="r", claim_ids=["c1"])
            for i, t in enumerate(texts, 1)
        ]
        content = _content().model_copy(update={"actions": actions})
        return [i.claim_id for i in checks.unfit_actions(content, ctx)]

    # gemma4:12b suggested an SME loan to a household saver
    assert unfit("demo_family_saver", "Оформить льготный кредит для МСБ", "Проверить условия ипотеки") == [
        "a1"
    ]
    assert unfit("demo_designer_npd", "Нанять сотрудника на проект", "Отложить резерв на налог НПД") == ["a1"]
    assert unfit(
        "demo_building_materials", "Пересмотреть семейный бюджет", "Пересчитать график платежей"
    ) == ["a1"]
    assert unfit("demo_london_studio", "Review the household budget", lang="en") == ["a1"]
    assert (
        unfit("demo_coffee_ip", "Оформить кредит для бизнеса", "Сверить НДС с поставщиками") == []
    )  # an ИП may


def test_counterparty_codes_never_reach_the_reader(store) -> None:
    """Payment data have no names; a code like cp_import_309 is replaced by what is known about the counterparty."""
    from refract.agent import checks

    profile = store.profiles["demo_building_materials"]
    run = smoke(store, "demo_building_materials__ev_yuan_ru__cfo__en")
    content = InsightContent.model_validate(
        {k: getattr(run.insight, k) for k in InsightContent.model_fields}
        | {"headline": "Imports from counterparty cp_import_309 get cheaper"}
    )
    named, notes = checks.name_counterparties(content, profile, "en")
    assert named.headline == "Imports from the supplier with 14% of outgoing payments get cheaper"
    assert notes and "cp_import_309" in notes[0]
    ru, _ = checks.name_counterparties(
        content.model_copy(update={"headline": "Платёж контрагенту cp_dev_101"}), profile, "ru"
    )
    assert ru.headline == "Платёж контрагенту (покупатель, 31% входящих платежей)"
    assert checks.name_counterparties(named, profile, "en") == (named, [])


def test_a_rule_from_the_sources_is_applied_with_the_right_arithmetic() -> None:
    """Found in a local run: “every 10% of freight → 1.5 pp of margin” became “a 15% jump cuts it by 1.5 pp”."""
    from refract.agent.checks.arithmetic import misapplied, rules_in

    rules = rules_in(
        "every 10% rise in freight cuts the exporter's margin by about 1.5 percentage points. "
        "Каждые 10% роста фрахта снижают маржу экспортёра на 1,5 п.п."
    )
    assert {(r.change, r.effect) for r in rules} == {(10.0, 1.5)}
    wrong = "A 15% jump in freight rates cuts your margin by approximately 1.5 percentage points."
    assert misapplied(wrong, rules)[1:] == (15.0, 1.5, 2.25)
    assert misapplied("Рост фрахта на 15% снизит маржу примерно на 1,5 п.п.", rules) is not None
    for fine in (
        "A 15% jump in freight rates cuts your margin by about 2.2 pp.",  # right, within rounding
        "Every 10% rise in freight cuts the margin by about 1.5 percentage points.",  # the rule itself
        "Inflation of 6.8% is 2.8 pp above the target.",  # another topic
        "Prices rose 6% and freight 15%, so the margin fell by 2 pp.",  # two changes: not the plain case
    ):
        assert misapplied(fine, rules) is None, fine
    run = Refract.demo().analyze("demo_grain_farm", "ev_wheat_en")
    rules_found = [i for s in run.trace.steps if s.name == "rule_check" for i in s.issues]
    assert any("2.25 pp, not 1.5" in i for i in rules_found)
    assert "w2" not in {c.claim_id for c in run.insight.why_it_matters}
