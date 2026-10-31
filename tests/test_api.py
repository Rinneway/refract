import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from refract.api import create_app
from refract.demo import OK, RATE_PLAN, _rate_draft
from refract.facade import Refract
from refract.llm import ScriptedLLM


@pytest.fixture
def api(store, settings):
    chat = {"intent": "clarify", "answer": "Ставка сохранена на уровне 14%.", "claims": []}
    llm = ScriptedLLM(
        {
            "plan": [RATE_PLAN],
            "draft_insight": [_rate_draft(False)],
            "verify": [OK],
            "chat_plan": [{"intent": "clarify", "steps": []}],
            "chat": [chat],
        },
        settings,
    )
    return TestClient(create_app(Refract(store, llm=llm, settings=settings)))


def test_read_endpoints(api) -> None:
    assert api.get("/health").json()["status"] == "ok"
    ids = {p["profile_id"] for p in api.get("/profiles").json()}
    assert "demo_coffee_ip" in ids and api.get("/profiles/demo_coffee_ip").json()["kind"] == "sole_proprietor"
    assert api.get("/events/ev_wheat_en").json()["language"] == "en"
    assert api.get("/profiles/nobody").status_code == 404


def test_insight_sync_trace_markdown_chat(api) -> None:
    r = api.post(
        "/insights?wait=true", json={"profile_id": "demo_building_materials", "event_id": "ev_rate_hold_ru"}
    )
    assert r.status_code == 200
    iid = r.json()["insight_id"]
    assert api.get(f"/insights/{iid}/trace").json()["status"] == "done"
    assert "Позиция Демо-банка" in api.get(f"/insights/{iid}/markdown").text
    answer = api.post(f"/insights/{iid}/chat", json={"question": "Что дальше?"}).json()
    assert answer["answer"].startswith("Ставка")


def test_job_stream(api) -> None:
    job = api.post(
        "/insights", json={"profile_id": "demo_building_materials", "event_id": "ev_rate_hold_ru"}
    ).json()
    with api.stream("GET", f"/jobs/{job['job_id']}/stream") as s:
        body = "".join(s.iter_text())
    assert "event: started" in body and "event: closed" in body
    assert api.get(f"/jobs/{job['job_id']}").json()["status"] == "ready"


def test_token(monkeypatch, store, settings) -> None:
    monkeypatch.setenv("REFRACT_API_TOKEN", "secret")
    c = TestClient(create_app(Refract(store, llm=ScriptedLLM({}, settings), settings=settings)))
    assert c.get("/profiles").status_code == 401
    assert c.get("/profiles", headers={"Authorization": "Bearer secret"}).status_code == 200
    assert c.get("/health").status_code == 200
