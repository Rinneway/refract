from refract.agent import ChatAgent
from refract.demo import run_demo
from refract.llm import ScriptedLLM


def test_chat_drops_unsupported_claims(store, settings) -> None:
    run = run_demo("rate")
    answer = {
        "intent": "clarify",
        "answer": "Ставка 14% сохраняется, поэтому платежи по кредитам в ближайшие месяцы не снизятся.",
        "claims": [
            {
                "claim_id": "q1",
                "text": "Ставка сохранена на уровне 14%.",
                "origin": "fact",
                "source_ids": ["src_art_rate_hold_ru"],
                "confidence": 0.9,
            },
            {
                "claim_id": "q2",
                "text": "Банк снизит ставку до 9%.",
                "origin": "house_view",
                "source_ids": ["src_made_up"],
                "confidence": 0.9,
            },
        ],
        "follow_ups": ["Что с застройщиком?"],
    }
    llm = ScriptedLLM({"chat_plan": [{"intent": "clarify", "steps": []}], "chat": [answer]}, settings)
    res = ChatAgent(llm, store).answer(run.state, run.insight, "Что будет с платежами?").response
    assert [c.claim_id for c in res.claims] == ["q1"]
    assert res.sources and res.sources[0].source_id == "src_art_rate_hold_ru"


def test_chat_rebuilds_answer_with_invented_numbers(store, settings) -> None:
    run = run_demo("rate")
    answer = {
        "intent": "clarify",
        "answer": "Платёж вырастет на 37 000 рублей.",
        "claims": [
            {
                "claim_id": "q1",
                "text": "Ставка сохранена на уровне 14%.",
                "origin": "fact",
                "source_ids": ["src_art_rate_hold_ru"],
                "confidence": 0.9,
            }
        ],
    }
    llm = ScriptedLLM({"chat_plan": [{"intent": "clarify", "steps": []}], "chat": [answer]}, settings)
    res = ChatAgent(llm, store).answer(run.state, run.insight, "Насколько вырастет платёж?").response
    assert "37" not in res.answer and res.answer == "Ставка сохранена на уровне 14%."


def test_chat_never_returns_an_empty_answer(store, settings) -> None:
    run = run_demo("rate")
    claim = {
        "claim_id": "q1",
        "text": "Ставка сохранена на уровне 14%.",
        "origin": "fact",
        "source_ids": ["src_art_rate_hold_ru"],
        "confidence": 0.9,
    }
    for claims, expected in (([claim], "Ставка сохранена на уровне 14%."), ([], "")):
        answer = {"intent": "clarify", "answer": "  ", "claims": claims}
        llm = ScriptedLLM({"chat_plan": [{"intent": "clarify", "steps": []}], "chat": [answer]}, settings)
        res = ChatAgent(llm, store).answer(run.state, run.insight, "Что со ставкой?")
        assert res.response.answer.strip() and (not expected or res.response.answer == expected)
        assert any("текста ответа" in n for s in res.trace.steps for n in s.issues)


def test_chat_slice_calls_tools(store, settings) -> None:
    run = run_demo("rate")
    plan = {
        "intent": "slice",
        "steps": [
            {
                "step_id": "s1",
                "tool": "get_counterparties",
                "args": {"role": "supplier"},
                "purpose": "поставщики",
            }
        ],
    }
    llm = ScriptedLLM(
        {
            "chat_plan": [plan],
            "chat": [
                {"intent": "slice", "answer": "Крупнейший поставщик — производитель цемента.", "claims": []}
            ],
        },
        settings,
    )
    run_chat = ChatAgent(llm, store).answer(run.state, run.insight, "А что с поставщиками?")
    assert "get_counterparties" in [s.name for s in run_chat.trace.steps]
    assert "Контрагенты по платежам" in llm.messages["chat"][-1]["content"]


def test_only_recent_runs_stay_in_memory(store, settings) -> None:
    from refract.demo import OK, RATE_PLAN, _rate_draft
    from refract.facade import Refract
    from refract.recent import Recent

    recent = Recent(2)
    for k in "abc":
        recent[k] = k
    assert list(recent) == ["b", "c"]
    recent["b"] = "b2"  # a rewrite moves the entry to the newest end
    recent["d"] = "d"
    assert list(recent.items()) == [("b", "b2"), ("d", "d")]

    llm = ScriptedLLM({"plan": [RATE_PLAN], "draft_insight": [_rate_draft(False)], "verify": [OK]}, settings)
    rf = Refract(store, llm=llm, settings=settings, keep_runs=1)
    first = rf.analyze("demo_building_materials", "ev_rate_hold_ru")
    second = rf.analyze("demo_building_materials", "ev_rate_hold_ru", role="owner")
    assert list(rf.runs) == [second.insight.insight_id] and first.insight.insight_id not in rf.runs
