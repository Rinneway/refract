import pytest

from refract.agent.impact import ImpactPolicy, event_links, impact_cap
from refract.schemas import ImpactLevel
from refract.schemas.vocab import country_code, division, section_of


@pytest.mark.parametrize(
    ("profile", "event", "level"),
    [
        (
            "demo_building_materials",
            "ev_rate_hold_ru",
            ImpactLevel.HIGH,
        ),  # loans: strong exposure, strong event
        (
            "demo_building_materials",
            "ev_housing_en",
            ImpactLevel.HIGH,
        ),  # developer customer with 31% of the flow
        ("demo_coffee_ip", "ev_injection_ru", ImpactLevel.MEDIUM),  # own activity, but the event is weak
        (
            "demo_coffee_ip",
            "ev_rate_hold_ru",
            ImpactLevel.MEDIUM,
        ),  # a loan priced off the rate, no sensitivity
        (
            "demo_london_studio",
            "ev_rate_hold_ru",
            ImpactLevel.LOW,
        ),  # its rate is the Bank of England's, not this one
        ("demo_family_saver", "ev_wheat_en", ImpactLevel.LOW),  # nothing in common
        ("demo_designer_npd", "ev_yuan_ru", ImpactLevel.LOW),  # trades with KZ and AE, not CN
        ("demo_grain_farm", "ev_wheat_en", ImpactLevel.HIGH),  # own activity, exports to TR and EG
        ("demo_london_studio", "ev_dollar_en", ImpactLevel.HIGH),  # fx:USD 0.8, the main factor
    ],
)
def test_impact_ceiling(store, profile, event, level) -> None:
    assert impact_cap(store.profiles[profile], store.events[event]).level is level


def test_secondary_factor_is_weaker(store) -> None:
    # ev_dollar_en: fx:USD is primary, inflation is secondary (countries cleared: US inflation is not a Russian
    # family's inflation)
    family = store.profiles["demo_family_saver"]
    links = event_links(family, store.events["ev_dollar_en"].model_copy(update={"countries": []}), "en")
    assert links and all("secondary" in x.text for x in links if x.kind == "exposure")


def test_thresholds_are_configurable(store) -> None:
    p, e = store.profiles["demo_grain_farm"], store.events["ev_rate_hold_ru"]  # policy_rate 0.6
    assert impact_cap(p, e).level is ImpactLevel.MEDIUM
    assert impact_cap(p, e, policy=ImpactPolicy(strong_exposure=0.5)).level is ImpactLevel.HIGH


def test_links_are_localised(store) -> None:
    p, e = store.profiles["demo_grain_farm"], store.events["ev_wheat_en"]
    ru = " ".join(x.render("ru") for x in event_links(p, e, "ru"))
    en = " ".join(x.render("en") for x in event_links(p, e, "en"))
    assert "связь" in ru and "link" in en


def test_country_codes() -> None:
    assert country_code("Объединённые Арабские Эмираты") == "AE"
    assert country_code("ОАЭ") == country_code("uae") == country_code("AE") == "AE"
    assert country_code("КНР") == "CN" and country_code("Республика Казахстан") == "KZ"
    assert country_code("United Kingdom") == "GB" and country_code("Türkiye") == "TR"


def test_nace_divisions() -> None:
    assert division("46.73") == "46" and division("1.11") == "01" and division("abc") is None
    assert section_of("46.73") == "G" and section_of("01.11") == "A" and section_of("62.01") == "J"


def test_product_link_counts_once_and_not_over_a_sensitivity(store) -> None:
    coffee, builder = store.profiles["demo_coffee_ip"], store.profiles["demo_building_materials"]
    rate = store.events["ev_rate_hold_ru"]
    links = [x for x in event_links(coffee, rate) if x.kind == "product"]
    assert [x.feature for x in links] == ["products.loans"] and links[0].level == 2
    assert not [x for x in event_links(builder, rate) if x.kind == "product"]  # policy_rate is a sensitivity


def test_national_factors_stay_in_their_country(store) -> None:
    london, family = store.profiles["demo_london_studio"], store.profiles["demo_family_saver"]
    rate, dollar = store.events["ev_rate_hold_ru"], store.events["ev_dollar_en"]  # RU; US
    assert not [x for x in event_links(london, rate) if x.feature == "policy_rate"]
    assert [x.feature for x in event_links(london, dollar) if x.kind == "exposure"] == [
        "fx:USD"
    ]  # crosses borders
    assert not event_links(family, dollar)  # US inflation is not a Russian family's inflation
    unknown = rate.model_copy(
        update={"countries": []}
    )  # the country of the news is not known: no restriction
    assert [x.feature for x in event_links(london, unknown)] == ["policy_rate"]


def test_trade_matches_by_what_is_traded(store) -> None:
    """Division 46 is any wholesale: grain news (46.2) is not about a building-materials wholesaler (46.73)."""
    builder, farm = store.profiles["demo_building_materials"], store.profiles["demo_grain_farm"]
    wheat = store.events["ev_wheat_en"]  # 01 and 46.2
    assert impact_cap(builder, wheat).level is ImpactLevel.LOW
    assert not [x for x in event_links(builder, wheat) if x.kind == "own_division"]
    trader = [x for x in event_links(farm, wheat) if x.feature == "counterparties.customer.cp_trader_411"]
    assert [x.level for x in trader] == [3]  # the grain trader (46.21, 38%) is in the group the news names
    general = wheat.model_copy(update={"industry_codes": ["46"]})  # wholesale without the goods: weak
    assert [x.level for x in event_links(builder, general) if x.feature == "industry"] == [1]


def test_a_section_letter_is_a_weak_link_whatever_the_share(store) -> None:
    # USN news names retail (47) and food service (56); the farm's grain trader (46.21, 38%) shares only section G
    links = event_links(store.profiles["demo_grain_farm"], store.events["ev_usn_vat_ru"])
    assert [(x.kind, x.level) for x in links] == [("counterparty", 1)]


def test_activity_groups() -> None:
    from refract.schemas.vocab import group

    assert (
        group("46.73") == "46.7" and group("46.2") == "46.2" and group("46") is None and group("x.1") is None
    )


def test_tax_news_about_another_regime_is_background(store) -> None:
    """The VAT threshold of the simplified system: news for the coffee shop on УСН, background for the self-employed."""
    from refract.agent.impact import tax_regimes

    assert tax_regimes("Порог НДС на упрощёнке") == {"usn"}
    assert tax_regimes("НПД (налог на профессиональный доход)") == {"npd"}
    assert tax_regimes("self-employed on the professional income tax") == {"npd"}
    assert tax_regimes("Ключевая ставка сохранена") == set()
    event = store.events["ev_usn_vat_ru"]
    npd = impact_cap(store.profiles["demo_designer_npd"], event, lang="ru")
    usn = impact_cap(store.profiles["demo_coffee_ip"], event, lang="ru")
    assert npd.level is ImpactLevel.LOW and "режиме УСН, а субъект на НПД" in npd.reason
    assert usn.level is ImpactLevel.HIGH
