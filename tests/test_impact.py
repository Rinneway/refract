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
        ("demo_london_studio", "ev_rate_hold_ru", ImpactLevel.MEDIUM),  # sensitivity 0.5 — moderate
        ("demo_family_saver", "ev_wheat_en", ImpactLevel.LOW),  # nothing in common
        ("demo_designer_npd", "ev_yuan_ru", ImpactLevel.LOW),  # trades with KZ and AE, not CN
        ("demo_grain_farm", "ev_wheat_en", ImpactLevel.HIGH),  # own activity, exports to TR and EG
        ("demo_london_studio", "ev_dollar_en", ImpactLevel.HIGH),  # fx:USD 0.8, the main factor
    ],
)
def test_impact_ceiling(store, profile, event, level) -> None:
    assert impact_cap(store.profiles[profile], store.events[event]).level is level


def test_secondary_factor_is_weaker(store) -> None:
    # ev_dollar_en: fx:USD is primary, inflation is secondary
    family = store.profiles["demo_family_saver"]
    links = event_links(family, store.events["ev_dollar_en"], "en")
    assert links and all("secondary" in x.text for x in links if x.kind == "exposure")


def test_thresholds_are_configurable(store) -> None:
    p, e = store.profiles["demo_london_studio"], store.events["ev_rate_hold_ru"]
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
