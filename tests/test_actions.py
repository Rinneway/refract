from refract.agent.actions import candidate_actions, enforce_actions
from refract.agent.policy import DEFAULT_POLICY
from refract.schemas import (
    Action,
    ActionKind,
    Claim,
    Direction,
    ImpactLevel,
    InsightContent,
    Origin,
    SubjectClaim,
)


def ids(cands) -> set[str]:
    return {c.rule.id for c in cands}


def content(actions, impact=ImpactLevel.MEDIUM, direction=Direction.NEGATIVE) -> InsightContent:
    c = Claim(claim_id="c1", text="x", origin=Origin.FACT, source_ids=["s"], confidence=1)
    w = SubjectClaim(claim_id="c2", text="y", origin=Origin.AI, features=["kind"], confidence=0.8)
    return InsightContent(
        headline="h",
        what_happened=[c],
        why_it_matters=[w],
        actions=actions,
        overall_impact=impact,
        overall_direction=direction,
    )


def test_rules_fire_by_features(store) -> None:
    rate = store.events["ev_rate_hold_ru"]
    assert ids(candidate_actions(store.profiles["demo_building_materials"], rate)) == {
        "fix_rate",
        "refinance",
        "deposit",
    }
    assert ids(candidate_actions(store.profiles["demo_family_saver"], rate)) == {
        "household_budget",
        "mortgage_check",
    }
    vat = store.events["ev_usn_vat_ru"]
    assert ids(candidate_actions(store.profiles["demo_coffee_ip"], vat)) == {
        "support_programs",
        "tax_regime_review",
        "compliance",
    }


def test_rule_texts_are_localised(store) -> None:
    rate = store.events["ev_rate_hold_ru"]
    ru = candidate_actions(store.profiles["demo_building_materials"], rate, lang="ru")[0].render("ru")
    en = candidate_actions(store.profiles["demo_building_materials"], rate, lang="en")[0].render("en")
    assert "продукт" in ru and "product" in en


def test_product_must_come_from_a_fired_rule(store) -> None:
    p, e = store.profiles["demo_building_materials"], store.events["ev_rate_hold_ru"]
    actions = [
        Action(
            action_id="a1",
            text="fix",
            kind=ActionKind.CONTACT_MANAGER,
            rationale="r",
            product="Fixed-rate loan",
        ),
        Action(
            action_id="a2",
            text="factoring",
            kind=ActionKind.APPLY_PRODUCT,
            rationale="r",
            product="Факторинг",
        ),
    ]
    out, notes = enforce_actions(content(actions), candidate_actions(p, e), e, lang="ru")
    assert [a.action_id for a in out.actions] == ["a1"]
    assert out.actions[0].product == "Кредит с фиксированной ставкой"  # canonical name in the run language
    assert any("a2" in n for n in notes)


def test_one_product_and_manager_first(store) -> None:
    p, e = store.profiles["demo_building_materials"], store.events["ev_rate_hold_ru"]
    actions = [
        Action(
            action_id="a1",
            text="deposit",
            kind=ActionKind.APPLY_PRODUCT,
            rationale="r",
            product="Депозит или неснижаемый остаток",
        ),
        Action(
            action_id="a2",
            text="refinance",
            kind=ActionKind.APPLY_PRODUCT,
            rationale="r",
            product="Рефинансирование кредита",
        ),
    ]
    out, _ = enforce_actions(content(actions, ImpactLevel.HIGH), candidate_actions(p, e), e)
    assert out.actions[0].kind is ActionKind.CONTACT_MANAGER  # high and negative
    assert sum(a.kind is ActionKind.APPLY_PRODUCT for a in out.actions) == 1
    # refinance is allowed only when the event is positive for the subject
    assert "a2" not in {a.action_id for a in out.actions}


def test_no_products_for_tax_events_or_when_switched_off(store) -> None:
    p, e = store.profiles["demo_coffee_ip"], store.events["ev_usn_vat_ru"]
    a = Action(
        action_id="a1",
        text="programme",
        kind=ActionKind.CONTACT_MANAGER,
        rationale="r",
        product="Консультация по программам поддержки",
    )
    out, _ = enforce_actions(content([a]), candidate_actions(p, e), e)
    assert out.actions[0].product is None
    rate = store.events["ev_rate_hold_ru"]
    b = Action(
        action_id="a1", text="deposit", kind=ActionKind.APPLY_PRODUCT, rationale="r", product="Term deposit"
    )
    out, _ = enforce_actions(
        content([b]),
        candidate_actions(store.profiles["demo_building_materials"], rate),
        rate,
        offer_products=False,
    )
    assert out.actions[0].product is None and out.actions[0].kind is ActionKind.CONTACT_MANAGER


def test_fallback_action(store) -> None:
    e = store.events["ev_wheat_en"]
    out, _notes = enforce_actions(content([]), [], e, lang="en")
    assert out.actions[0].kind is ActionKind.REQUEST_CALLBACK and "manager" in out.actions[0].text


def test_policy_files_are_complete() -> None:
    from refract.schemas import ReaderRole
    from refract.schemas.vocab import is_known_factor

    roles = DEFAULT_POLICY.roles()
    assert set(roles) == {r.value for r in ReaderRole}
    for r in roles.values():
        for lang in ("ru", "en"):
            assert r.name(lang) and r.main_question(lang) and r.depth(lang) and r.tone(lang)
    for rule in DEFAULT_POLICY.rules():
        assert rule.text("ru") and rule.text("en") and rule.rationale("ru") and rule.rationale("en")
        if rule.product_key:
            assert rule.product("ru") and rule.product("en")
        for f in rule.when.get("factors", []):
            assert is_known_factor(f.replace(":*", ":x")), f
