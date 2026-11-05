"""The whole tool over HTTP: the CLI and the real `LLMClient` against a fake OpenAI-compatible server.

The other tests drive the agent through `ScriptedLLM`. These go through everything a user runs — `check`,
`models`, `analyze` (a stored event and pasted news), `chat` on the saved run, `theses`, and the evaluation
pipeline — so the HTTP client, structured output, the cache, the call log and the CLI wiring are covered too.
"""

from __future__ import annotations

import json
import re

import pytest

from refract.cli import main
from refract.demo import NUMBER_FINDING, OK, RATE_PLAN, _rate_draft
from tests.conftest import DEMO
from tests.fake_openai import FakeOpenAI

QUOTE = "Мы ожидаем, что центральный банк сохранит ключевую ставку на уровне 14% до конца 2026 года."


def _draft(messages):
    revising = any("Замечания:" in (m.get("content") or "") for m in messages)
    return _rate_draft(invented=not revising)


def _verify(messages):
    return NUMBER_FINDING if "1,2 млрд" in messages[-1]["content"] else OK


def _code(messages):
    return {"code": re.search(r"cobalt-\d+", messages[-1]["content"]).group(0)}


def _theses(messages):
    if QUOTE not in messages[-1]["content"]:
        return {"theses": []}
    return {
        "theses": [
            {
                "topic": "policy_rate",
                "statement": "Ключевая ставка останется 14% до конца 2026 года.",
                "figures": [],
                "horizon": "конец 2026 года",
                "factors": ["policy_rate"],
                "quote": QUOTE,
            }
        ]
    }


def _score(n: int) -> dict:
    c = {"note": "ok", "score": n}
    return {
        "unsupported": [],
        **{
            k: c
            for k in ("what_happened", "why_it_matters", "consequences", "actions", "house_view", "grounding")
        },
        "retelling": False,
        "impact_calibrated": True,
        "overall": 4,
        "comment": "—",
    }


ANSWERS = {
    "Echo": {"answer": "ok", "number": 42},
    "Code": _code,
    "Plan": RATE_PLAN,
    "InsightContent": _draft,
    "VerificationReport": _verify,
    "HeadlineOut": {"headline": "Ставка 14% откладывает удешевление кредитов"},
    "NewsTags": {
        "is_business_relevant": True,
        "event_type": "monetary_policy",
        "macro_factors": ["policy_rate"],
        "industry_sections": [],
        "industry_codes": [],
        "regions": [],
        "countries": ["RU"],
        "commodities": [],
        "importance": 0.7,
        "key_facts": ["Ставка сохранена на уровне 14%"],
        "language": "ru",
    },
    "ChatPlan": {"intent": "clarify", "steps": []},
    "ChatAnswer": {
        "intent": "clarify",
        "answer": "Ставка сохранена на уровне 14%.",
        "claims": [
            {
                "claim_id": "q1",
                "text": "Ставка сохранена на уровне 14%.",
                "origin": "fact",
                "source_ids": ["src_art_rate_hold_ru"],
                "confidence": 0.9,
            }
        ],
        "follow_ups": [],
    },
    "ThesesOut": _theses,
    "JudgeScore": _score(2),
    "BaselineAnswer": {
        "headline": "Ставка 14%",
        "what_happened": ["Ставку сохранили"],
        "why_it_matters": ["Кредиты не подешевеют"],
        "risks": [],
        "opportunities": [],
        "scenarios": [],
        "house_view": [],
        "actions": ["Следить за ставкой"],
        "overall_impact": "medium",
        "overall_direction": "negative",
    },
}


@pytest.fixture
def server(monkeypatch, tmp_path):
    with FakeOpenAI(ANSWERS) as fake:
        for var, value in {
            "LLM_BASE_URL": fake.base_url,
            "LLM_API_KEY": "test-key",
            "REFRACT_MODEL": "fake-main",
            "REFRACT_VERIFIER_MODEL": "fake-verifier",
            "REFRACT_CONFIG": str(tmp_path / "no-config.toml"),  # never the developer's refract.toml
            "REFRACT_DATA_DIR": str(DEMO),
            "REFRACT_LANGUAGE": "ru",
            "LLM_CACHE_DIR": str(tmp_path / "cache"),
            "LLM_LOG_PATH": str(tmp_path / "log.jsonl"),
        }.items():
            monkeypatch.setenv(var, value)
        yield fake


def test_check_and_models(server, capsys) -> None:
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "structured output works" in out and "context holds" in out and "FAIL" not in out
    assert main(["models"]) == 0
    assert "fake-main" in capsys.readouterr().out


def test_analyze_save_chat_over_http(server, tmp_path, capsys) -> None:
    runs = tmp_path / "runs"
    code = main(
        ["analyze", "--profile", "demo_building_materials", "--event", "ev_rate_hold_ru", "--out", str(runs)]
    )
    out = capsys.readouterr().out
    assert code == 0 and "Как получен вывод" in out
    assert server.schemas()[:3] == ["Plan", "InsightContent", "VerificationReport"]
    assert "InsightContent" in server.schemas()[3:]  # the revision went over HTTP too
    saved = sorted(runs.glob("*.json"))
    assert len(saved) == 1
    run = json.loads(saved[0].read_text(encoding="utf-8"))
    assert "1,2 млрд" not in json.dumps(run["insight"], ensure_ascii=False)  # caught, revised and gone
    assert run["trace"]["totals"]["iterations"] == 1
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["step"] for r in log} >= {"plan", "draft_insight", "verify", "revise"}
    assert all(r["prompt_tokens"] > 0 for r in log)

    before = len(server.requests)  # the same run again: every answer comes from the disk cache
    assert (
        main(["analyze", "--profile", "demo_building_materials", "--event", "ev_rate_hold_ru", "--json"]) == 0
    )
    assert len(server.requests) == before
    insight = json.loads(capsys.readouterr().out)
    assert insight["profile_id"] == "demo_building_materials" and insight["what_happened"]

    assert main(["chat", "--run", str(saved[0]), "--q", "Что со ставкой?"]) == 0
    assert "14%" in capsys.readouterr().out
    assert server.schemas()[-2:] == ["ChatPlan", "ChatAnswer"]


def test_analyze_pasted_news(server, capsys) -> None:
    code = main(
        [
            "analyze",
            "--profile",
            "demo_building_materials",
            "--title",
            "ЦБ сохранил ставку",
            "--text",
            "Банк России сохранил ключевую ставку на уровне 14% годовых.",
        ]
    )
    assert code == 0 and server.schemas()[0] == "NewsTags"  # the pasted news is tagged first
    assert "Как получен вывод" in capsys.readouterr().out


def test_theses_from_documents(server, tmp_path, capsys) -> None:
    out = tmp_path / "theses.json"
    assert main(["theses", "--docs", str(DEMO / "knowledge" / "docs"), "--out", str(out)]) == 0
    theses = json.loads(out.read_text(encoding="utf-8"))
    assert [t["quote"] for t in theses] == [QUOTE]  # kept because the quote is verbatim in the document
    assert theses[0]["countries"] == ["RU"]  # inherited from the document's front matter


def test_evaluation_pipeline(server, tmp_path, capsys) -> None:
    exp = tmp_path / "exp"
    pairs = exp / "pairs.jsonl"
    exp.mkdir()
    pairs.write_text(
        json.dumps(
            {
                "profile_id": "demo_building_materials",
                "event_id": "ev_rate_hold_ru",
                "role": "cfo",
                "language": "ru",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "eval",
                "run",
                "--out",
                str(exp),
                "--systems",
                "full,no_reconcile,single_call,retelling",
                "--workers",
                "1",
            ]
        )
        == 0
    )
    assert (exp / "manifest.json").exists()
    assert main(["eval", "judge", "--out", str(exp), "--judges", "fake-verifier", "--workers", "1"]) == 0
    assert main(["eval", "report", "--out", str(exp)]) == 0
    report = (exp / "report.md").read_text(encoding="utf-8")
    for system in ("full", "no_reconcile", "single_call", "retelling"):
        assert f"`{system}`" in report
    assert "## Process" in report and "## Models" in report and "mixed" not in report
    summary = json.loads((exp / "summary.json").read_text(encoding="utf-8"))
    assert summary["systems"]["full"]["judges"]["fake-verifier"]["total_mean"] == 12


def test_python_api(server) -> None:
    from refract import Refract

    rf = Refract.from_dir()
    run = rf.analyze("demo_building_materials", "ev_rate_hold_ru", role="cfo")
    assert run.insight and run.trace.status == "done" and run.trace.totals.llm_calls >= 4
    assert run.trace.totals.tokens_in > 0  # usage is read from the HTTP response
    assert rf.chat(run.insight.insight_id, "Что со ставкой?").response.answer.startswith("Ставка")


def test_rest_api(server) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from refract.api import create_app

    api = TestClient(create_app())  # builds Refract from the environment, like `refract serve`
    health = api.get("/health").json()
    assert health["status"] == "ok" and health["llm_configured"] is True
    r = api.post(
        "/insights?wait=true", json={"profile_id": "demo_building_materials", "event_id": "ev_rate_hold_ru"}
    )
    assert r.status_code == 200
    iid = r.json()["insight_id"]
    assert api.get(f"/insights/{iid}/trace").json()["totals"]["iterations"] == 1
    assert api.post(f"/insights/{iid}/chat", json={"question": "Что дальше?"}).json()["answer"]
    news = {
        "title": "ЦБ сохранил ставку",
        "text": "Банк России сохранил ключевую ставку на уровне 14% годовых.",
    }
    job = api.post("/insights", json={"profile_id": "demo_coffee_ip", "news": news}).json()
    with api.stream("GET", f"/jobs/{job['job_id']}/stream") as s:
        body = "".join(s.iter_text())
    assert "event: closed" in body and api.get(f"/jobs/{job['job_id']}").json()["status"] == "ready"


def test_mcp_server(server) -> None:
    pytest.importorskip("mcp")
    import asyncio

    from refract import Refract
    from refract.mcp_server import build_server
    from tests.test_mcp import text_of

    mcp = build_server(Refract.from_dir())
    out = text_of(
        asyncio.run(
            mcp.call_tool(
                "analyze_news", {"profile_id": "demo_building_materials", "event_id": "ev_rate_hold_ru"}
            )
        )
    )
    iid = re.search(r"insight_id: `([^`]+)`", out).group(1)
    assert "Как получен вывод" in text_of(asyncio.run(mcp.call_tool("get_trace", {"insight_id": iid})))
    assert "14%" in text_of(
        asyncio.run(mcp.call_tool("ask_followup", {"insight_id": iid, "question": "Что дальше?"}))
    )


def test_feed_tags_untagged_news_and_analyzes_the_top_item(server, capsys) -> None:
    from refract import Refract

    rf = Refract.from_dir()
    news = rf.news_from_text(
        "ЦБ сохранил ставку", "Банк России сохранил ключевую ставку на уровне 14% годовых."
    )
    rf.store.add_event(news)
    feed = rf.feed("demo_family_saver", top=None)
    assert news.event_id in [it.event_id for it in feed.items] and server.schemas() == ["NewsTags"]

    assert main(["feed", "--profile", "demo_building_materials", "--analyze", "1"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("## Новости для demo_building_materials") and "Как получен вывод" not in out
    assert "InsightContent" in server.schemas()  # the strongest direct item went through the agent
