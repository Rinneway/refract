"""Demo mode: the agent without a model, every model call answered from a recording."""

from __future__ import annotations

import json

import pytest

from refract.cli import main
from refract.demo import DemoError, Replay
from refract.facade import Refract


@pytest.fixture(scope="module")
def rf() -> Refract:
    return Refract.demo()


def test_every_recorded_pair_replays_to_an_insight(rf) -> None:
    for profile, event, role, lang in Replay().pairs():
        run = rf.analyze(profile, event, role=role, language=lang)
        assert run.ok, (profile, event, run.trace.failure_reason())
        models = {s.model for s in run.trace.steps if s.model}
        assert models and all(m.startswith("recorded:") or m == "scripted" for m in models)
        assert run.trace.config["replay"]


def test_replay_keeps_the_recorded_role_and_language(rf) -> None:
    run = rf.analyze("demo_family_saver", "ev_rate_hold_ru", role="cfo", language="en")
    assert (run.trace.role.value, run.trace.language) == ("individual", "ru")


def test_what_demo_mode_cannot_do_is_said_plainly(rf) -> None:
    with pytest.raises(DemoError, match="replayable pairs"):
        rf.analyze("demo_family_saver", "ev_dollar_en")
    with pytest.raises(DemoError, match="pasted"):
        rf.analyze("demo_family_saver", {"title": "Новость", "text": "Текст новости для проверки."})
    run = rf.analyze("demo_coffee_ip", "ev_sme_loans_ru", role="entrepreneur")
    with pytest.raises(DemoError, match="follow-up"):
        rf.chat(run.insight.insight_id, "А что с кредитом?")


def test_demo_brief_explains_only_recorded_news(rf) -> None:
    brief = rf.brief("demo_building_materials", top=5)
    recorded = {e for p, e, *_ in Replay().pairs() if p == "demo_building_materials"}
    assert brief.items and {i.event_id for i in brief.items} <= recorded
    assert all(i.error is None for i in brief.items)


def test_cli_demo_flag(capsys) -> None:
    assert (
        main(["analyze", "--demo", "--profile", "demo_grain_farm", "--event", "ev_wheat_en", "--json"]) == 0
    )
    out = capsys.readouterr()
    assert json.loads(out.out)["language"] == "en" and "not a live model" in out.err
    assert main(["analyze", "--demo", "--profile", "demo_grain_farm", "--event", "ev_rate_hold_ru"]) == 2
    assert "no recording" in capsys.readouterr().err
    assert main(["demo", "--list"]) == 0
    assert "demo_london_studio\tev_dollar_en\tcfo\ten\trecorded" in capsys.readouterr().out


def test_api_in_demo_mode() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from refract.api import create_app

    c = TestClient(create_app(Refract.demo()))
    assert c.get("/health").json()["demo"] is True
    ok = c.post("/insights?wait=true", json={"profile_id": "demo_grain_farm", "event_id": "ev_wheat_en"})
    assert ok.status_code == 200 and ok.json()["headline"]
    bad = c.post("/insights", json={"profile_id": "demo_grain_farm", "event_id": "ev_rate_hold_ru"})
    assert bad.status_code == 409 and "replayable" in bad.json()["detail"]


def test_cyrillic_survives_a_windows_pipe() -> None:
    import io
    import sys

    from refract.cli import utf8_streams

    raw = io.BytesIO()
    pipe = io.TextIOWrapper(raw, encoding="cp1252")
    old, sys.stdout = sys.stdout, pipe
    try:
        utf8_streams()
        sys.stdout.write("Ставка 14%")
        sys.stdout.flush()
    finally:
        sys.stdout = old
    assert raw.getvalue().decode("utf-8") == "Ставка 14%"


def test_record_writes_what_replay_reads(tmp_path, monkeypatch) -> None:
    """`refract demo --record` with a model: here the model is itself a replay, so the new file must replay the same."""
    from refract.demo import record, recordings

    source = Refract.demo()
    key = "demo_grain_farm__ev_wheat_en__cfo__en"
    original = next(r for r in recordings() if r.key == key)

    class Live(Refract):  # the configured model is a replay of the original recording
        @property
        def llm(self):
            return source.replay.llm(original)

    rf = Live(source.store, settings=source.settings)
    out = tmp_path / "rec.json"
    assert record(rf, out, [("demo_grain_farm", "ev_wheat_en", "cfo", "en")]) == {key: "ok"}
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"_note", "_models", key} and isinstance(data[key]["plan"], dict)
    monkeypatch.setenv("REFRACT_RECORDINGS", str(out))
    again = Refract.demo().analyze("demo_grain_farm", "ev_wheat_en")
    first = source.analyze("demo_grain_farm", "ev_wheat_en")
    assert again.insight.headline == first.insight.headline


def test_record_without_a_model_is_a_message(tmp_path, monkeypatch, capsys) -> None:
    for var in ("LLM_API_KEY", "OPENAI_API_KEY", "REFRACT_CONFIG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no refract.toml, no .env
    assert main(["demo", "--record", str(tmp_path / "rec.json")]) == 2
    assert "needs a model" in capsys.readouterr().err
