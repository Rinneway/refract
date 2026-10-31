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
