"""The personal news feed: stored events ranked for a subject by code."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from refract.agent.feed import rank, relevance
from refract.config import Settings
from refract.facade import Refract, news_from_text
from refract.llm import ScriptedLLM

NEWS = news_from_text(
    "Банк России сохранил ключевую ставку",
    "Банк России сохранил ключевую ставку на уровне 14% годовых.",
    published_at=datetime(2026, 10, 30, tzinfo=timezone.utc),
)
TAGS = {
    "is_business_relevant": True,
    "event_type": "monetary_policy",
    "macro_factors": ["policy_rate"],
    "industry_sections": [],
    "industry_codes": [],
    "regions": [],
    "countries": ["RU"],
    "commodities": [],
    "importance": 0.8,
    "key_facts": ["Ставка 14%"],
    "language": "ru",
}


def ids(feed) -> list[str]:
    return [it.event_id for it in feed.items]


def test_each_subject_gets_its_own_order(store) -> None:
    coffee = rank(store.profiles["demo_coffee_ip"], store.events.values())
    assert (
        ids(coffee)[0] == "ev_usn_vat_ru" and coffee.items[0].impact_ceiling.value == "high"
    )  # its tax regime
    london = rank(store.profiles["demo_london_studio"], store.events.values(), lang="en")
    assert [(it.event_id, it.relevance) for it in london.items] == [
        ("ev_dollar_en", "direct"),
        ("ev_housing_en", "background"),
    ]  # the Bank of Russia's rate is not a UK studio's news
    assert (
        london.items[0].links[0].startswith("strong link — US dollar exchange rate: 45% of incoming payments")
    )
    builder = rank(store.profiles["demo_building_materials"], store.events.values(), top=None)
    assert {it.event_id: it.relevance for it in builder.items}[
        "ev_wheat_en"
    ] == "background"  # grain, not cement


def test_filters(store) -> None:
    p = store.profiles["demo_coffee_ip"]
    everything = rank(p, store.events.values(), top=None, include_unrelated=True)
    assert everything.considered == len(store.events) and everything.items[-1].relevance == "none"
    recent = rank(p, store.events.values(), since=datetime(2026, 10, 28), top=None)
    assert {it.event_id for it in recent.items} == {"ev_injection_ru", "ev_sme_loans_ru"}
    assert len(rank(p, store.events.values(), top=2).items) == 2
    assert [relevance(s) for s in range(4)] == ["none", "background", "direct", "direct"]


def test_untagged_news_is_tagged_once_or_reported(store, settings) -> None:
    llm = ScriptedLLM({"classify_news": [TAGS]}, settings)
    rf = Refract(store, llm=llm, settings=settings)
    rf.store.add_event(NEWS)
    feed = rf.feed("demo_family_saver", top=None)
    assert NEWS.event_id in ids(feed) and not feed.untagged and llm.steps == ["classify_news"]
    rf.feed("demo_family_saver")
    assert llm.steps == ["classify_news"]  # the tags stay in the store

    plain = Refract(store, settings=Settings(api_key=None))  # no model configured: ranked without tags
    plain.store.add_event(NEWS.model_copy(update={"event_id": "ev_plain"}))
    assert "ev_plain" in plain.feed("demo_family_saver").untagged


def test_cli_feed(capsys) -> None:
    from refract.cli import main
    from tests.conftest import DEMO

    assert main(["feed", "--data", str(DEMO), "--profile", "demo_coffee_ip", "--no-tag", "--top", "2"]) == 0
    out = capsys.readouterr().out
    assert out.index("ev_usn_vat_ru") < out.index("ev_sme_loans_ru") and "показано: 2" in out
    assert main(["feed", "--data", str(DEMO), "--profile", "demo_london_studio", "--no-tag", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["items"][0]["event_id"] == "ev_dollar_en"
    assert main(["feed", "--data", str(DEMO), "--profile", "nobody"]) == 2
    assert main(["feed", "--data", str(DEMO), "--profile", "demo_coffee_ip", "--since", "yesterday"]) == 2


def test_rest_and_mcp(store, settings) -> None:
    rf = Refract(store, llm=ScriptedLLM({}, settings), settings=settings)
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from refract.api import create_app

    api = TestClient(create_app(rf))
    body = api.get("/profiles/demo_coffee_ip/feed?top=3&tag=false").json()
    assert [i["event_id"] for i in body["items"]][:1] == ["ev_usn_vat_ru"] and len(body["items"]) == 3
    assert api.get("/profiles/nobody/feed").status_code == 404
    since = api.get("/profiles/demo_coffee_ip/feed?tag=false&since=2026-10-28T00:00:00Z").json()
    assert {i["event_id"] for i in since["items"]} == {"ev_injection_ru", "ev_sme_loans_ru"}

    pytest.importorskip("mcp")
    import asyncio

    from refract.mcp_server import build_server
    from tests.test_mcp import text_of

    server = build_server(rf)
    out = text_of(asyncio.run(server.call_tool("news_feed", {"profile_id": "demo_london_studio", "top": 1})))
    assert "ev_dollar_en" in out and "ev_rate_hold_ru" not in out


def test_brief_explains_the_strongest_news(store, settings, capsys) -> None:
    from refract.demo import OK, RATE_PLAN, _rate_draft
    from refract.render import brief_markdown

    llm = ScriptedLLM({"plan": [RATE_PLAN], "draft_insight": [_rate_draft(False)], "verify": [OK]}, settings)
    rf = Refract(store, llm=llm, settings=settings)
    family = rf.brief("demo_family_saver", top=3)
    assert family.considered == 1 and [i.event_id for i in family.items] == ["ev_rate_hold_ru"]
    item = family.items[0]
    assert item.insight_id in rf.runs and item.headline and item.why and item.action and item.error is None
    builder = rf.brief("demo_building_materials", top=2)
    assert builder.considered == 3 and len(builder.items) == 2  # one agent run per item, no more
    text = brief_markdown(family)
    assert text.startswith("## Бриф для demo_family_saver") and "--event ev_rate_hold_ru" in text

    broken = Refract(
        store, llm=ScriptedLLM({"plan": [RATE_PLAN], "draft_insight": [None]}, settings), settings=settings
    )
    failed = broken.brief("demo_family_saver").items[0]  # the model gave no draft: the brief says so
    assert (
        failed.insight_id is None
        and failed.error
        and "инсайт не получен" in brief_markdown(broken.brief("demo_family_saver"))
    )


def test_brief_over_rest_and_mcp(store, settings) -> None:
    from refract.demo import OK, RATE_PLAN, _rate_draft

    llm = ScriptedLLM({"plan": [RATE_PLAN], "draft_insight": [_rate_draft(False)], "verify": [OK]}, settings)
    rf = Refract(store, llm=llm, settings=settings)
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from refract.api import create_app

    api = TestClient(create_app(rf))
    body = api.post("/profiles/demo_family_saver/brief?top=1").json()
    assert (
        body["items"][0]["event_id"] == "ev_rate_hold_ru" and api.post("/profiles/x/brief").status_code == 404
    )
    pytest.importorskip("mcp")
    import asyncio

    from refract.mcp_server import build_server
    from tests.test_mcp import text_of

    out = text_of(asyncio.run(build_server(rf).call_tool("news_brief", {"profile_id": "demo_family_saver"})))
    assert "Ставка 14%" in out
