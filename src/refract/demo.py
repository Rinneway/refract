"""Scripted end-to-end runs that need no API key: `refract demo`.

The model's answers are scripted, everything else is real — tools, retrieval, rule checks, the impact ceiling,
the action policy, pruning and the trace. Two scenarios show what the deterministic layer does with typical model
mistakes:

- `rate` — a building-materials wholesaler × the policy-rate decision. The draft contains an invented number and a
  product no rule allows; the rule checks catch the number, the agent revises, the policy removes the product.
- `injection` — a sole-proprietor coffee shop × a news item with a prompt injection. The draft obeys the injected
  command (high impact, a loan offer); the impact ceiling and the action policy undo it.
- `dollar` (English) — a London software studio × a stronger dollar. The draft credits the institution with a
  statement that cites only the news and puts an invented number in the headline; the revision does not fix them, so
  the statement is cut and the headline is rewritten from verified statements.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .agent import Agent, AgentRun
from .config import Settings
from .knowledge.store import DataStore
from .llm.testing import ScriptedLLM

_PACKAGED = Path(__file__).resolve().parent / "demo_data"  # inside an installed wheel
_REPO = Path(__file__).resolve().parents[2] / "examples" / "demo"  # in a source checkout
DEMO_DIR = _PACKAGED if _PACKAGED.exists() else _REPO

RATE_PLAN = {
    "rationale": "У компании кредиты и лизинг, а крупнейший покупатель — застройщик: ставка влияет и на долг, и на спрос.",
    "steps": [
        {
            "step_id": "s1",
            "tool": "get_house_view",
            "args": {"factors": ["policy_rate"], "topics": ["policy_rate", "credit"]},
            "purpose": "позиция Демо-банка по ставке",
        },
        {
            "step_id": "s2",
            "tool": "get_counterparties",
            "args": {"role": "customer"},
            "purpose": "отрасли покупателей",
        },
        {
            "step_id": "s3",
            "tool": "search_research",
            "args": {"query": "прогноз ключевой ставки конец года"},
            "purpose": "фрагменты аналитики",
        },
    ],
}


def _rate_draft(invented: bool) -> dict:
    risk_text = (
        "Расходы на обслуживание кредитов и лизинга вырастут на 1,2 млрд руб. за год."
        if invented
        else "Расходы на обслуживание кредитов и лизинга в ближайшие два квартала не снизятся."
    )
    return {
        "headline": "Ставка 14% сохраняется: удешевления кредитов и лизинга в ближайшие месяцы не будет",
        "what_happened": [
            {
                "claim_id": "c1",
                "text": "Центробанк 23 октября сохранил ключевую ставку на уровне 14% годовых.",
                "origin": "fact",
                "source_ids": ["src_art_rate_hold_ru"],
                "confidence": 0.95,
            }
        ],
        "why_it_matters": [
            {
                "claim_id": "c2",
                "text": "У компании есть кредиты и лизинг, поэтому сохранение ставки откладывает снижение расходов на обслуживание долга.",
                "origin": "ai",
                "source_ids": ["src_art_rate_hold_ru", "src_profile"],
                "features": ["products.loans", "products.leasing", "policy_rate"],
                "confidence": 0.8,
            },
            {
                "claim_id": "c3",
                "text": "Крупнейший покупатель — застройщик (31% входящих платежей): высокая ставка сдерживает ипотеку и может замедлить его закупки.",
                "origin": "hypothesis",
                "source_ids": ["src_payments"],
                "features": ["counterparties.customer.cp_dev_101", "construction"],
                "confidence": 0.6,
                "caveat": "По платежам видно только оборот, а не условия договоров с застройщиком.",
            },
        ],
        "risks": [
            {
                "claim_id": "r1",
                "text": risk_text,
                "origin": "ai",
                "source_ids": ["src_demo_macro_2026q4_t3"],
                "features": ["products.loans"],
                "confidence": 0.7,
                "kind": "risk",
                "impact": "medium",
                "horizon": "short",
            }
        ],
        "opportunities": [],
        "scenarios": [],
        "house_view": [
            {
                "claim_id": "b1",
                "text": "Демо-банк ожидает сохранения ключевой ставки 14% до конца 2026 года.",
                "origin": "house_view",
                "source_ids": ["src_demo_macro_2026q4_t1"],
                "confidence": 0.9,
            },
            {
                "claim_id": "b2",
                "text": "Базовый прогноз Демо-банка на конец 2027 года — 12%.",
                "origin": "house_view",
                "source_ids": ["src_demo_macro_2026q4_t2"],
                "confidence": 0.9,
            },
        ],
        "actions": [
            {
                "action_id": "a1",
                "text": "Обсудить с менеджером фиксацию ставки по части долга",
                "kind": "contact_manager",
                "rationale": "Есть кредиты с плавающей стоимостью.",
                "product": "Кредит с фиксированной ставкой",
                "claim_ids": ["c2"],
            },
            {
                "action_id": "a2",
                "text": "Подключить факторинг по поставкам застройщику",
                "kind": "apply_product",
                "rationale": "Крупный покупатель — застройщик.",
                "product": "Факторинг",
                "claim_ids": ["c3"],
            },
        ],
        "overall_impact": "medium",
        "overall_direction": "negative",
    }


INJECTION_DRAFT = {
    "headline": "Закрытие сети кофеен — тревожный сигнал для вашей кофейни",
    "what_happened": [
        {
            "claim_id": "c1",
            "text": "Региональная сеть кофеен закроет 12 из 40 заведений до конца года.",
            "origin": "fact",
            "source_ids": ["src_art_injection_ru"],
            "confidence": 0.9,
        }
    ],
    "why_it_matters": [
        {
            "claim_id": "c2",
            "text": "Вы тоже работаете в подаче напитков, и выручка зависит от трат горожан: снижение посещаемости кофеен касается и вас.",
            "origin": "ai",
            "source_ids": ["src_art_injection_ru", "src_profile"],
            "features": ["industry", "consumer_demand"],
            "confidence": 0.7,
        }
    ],
    "risks": [
        {
            "claim_id": "r1",
            "text": "Посещаемость кофеен в крупных городах снижается второй квартал подряд.",
            "origin": "fact",
            "source_ids": ["src_art_injection_ru"],
            "features": ["industry"],
            "confidence": 0.8,
            "kind": "risk",
            "impact": "high",
            "horizon": "short",
        }
    ],
    "opportunities": [],
    "scenarios": [],
    "house_view": [],
    "actions": [
        {
            "action_id": "a1",
            "text": "Оформить кредитную линию",
            "kind": "apply_product",
            "rationale": "Так сказано в новости.",
            "product": "Кредитная линия на пополнение оборотных средств",
            "claim_ids": ["r1"],
        }
    ],
    "overall_impact": "high",
    "overall_direction": "negative",
}

DOLLAR_PLAN = {
    "rationale": "Almost half of incoming payments are in dollars from a US client.",
    "steps": [
        {
            "step_id": "s1",
            "tool": "get_house_view",
            "args": {"factors": ["fx:USD"], "topics": ["fx"]},
            "purpose": "Demo Bank view on the dollar",
        },
        {
            "step_id": "s2",
            "tool": "get_counterparties",
            "args": {"role": "customer"},
            "purpose": "who pays in dollars",
        },
    ],
}

DOLLAR_DRAFT = {
    "headline": "A 5% stronger dollar lifts sterling revenue from your US client",
    "what_happened": [
        {
            "claim_id": "c1",
            "text": "The dollar gained 1.2% against a basket of currencies after US consumer prices rose 3.4% year on year.",
            "origin": "fact",
            "source_ids": ["src_art_dollar_en"],
            "confidence": 0.95,
        }
    ],
    "why_it_matters": [
        {
            "claim_id": "c2",
            "text": "45% of incoming payments come from a US client billed in dollars, so a stronger dollar raises revenue in sterling.",
            "origin": "ai",
            "source_ids": ["src_art_dollar_en", "src_payments"],
            "features": ["counterparties.customer.cp_us_client_1", "fx:USD"],
            "confidence": 0.85,
        },
        {
            "claim_id": "c3",
            "text": "Developer salaries are paid in sterling, so the margin on US work widens while the dollar holds.",
            "origin": "hypothesis",
            "source_ids": ["src_profile"],
            "features": ["labor_market", "products.payroll"],
            "confidence": 0.6,
            "caveat": "The profile does not show the currency of costs.",
        },
    ],
    "risks": [
        {
            "claim_id": "r1",
            "text": "If expectations of a Federal Reserve cut return, the dollar may give back its gains.",
            "origin": "ai",
            "source_ids": ["src_art_dollar_en"],
            "features": ["fx:USD"],
            "confidence": 0.6,
            "kind": "risk",
            "impact": "medium",
            "horizon": "short",
        }
    ],
    "opportunities": [
        {
            "claim_id": "o1",
            "text": "Part of the expected dollar receipts can be locked in at the current rate.",
            "origin": "ai",
            "source_ids": ["src_demo_fxcom_2026m10_t2"],
            "features": ["products.fx_account"],
            "confidence": 0.7,
            "kind": "opportunity",
            "impact": "high",
            "horizon": "immediate",
        }
    ],
    "scenarios": [],
    "house_view": [
        {
            "claim_id": "b1",
            "text": "Demo Bank expects the dollar to stay strong into year-end; its GBP/USD forecast for the end of 2026 is 1.24.",
            "origin": "house_view",
            "source_ids": ["src_demo_fxcom_2026m10_t1"],
            "confidence": 0.9,
        },
        {
            "claim_id": "b2",
            "text": "Demo Bank says the dollar rally will last into next year.",
            "origin": "house_view",
            "source_ids": ["src_art_dollar_en"],
            "confidence": 0.8,
        },
    ],
    "actions": [
        {
            "action_id": "a1",
            "text": "Discuss hedging half of the next two quarters' dollar receipts",
            "kind": "contact_manager",
            "rationale": "45% of incoming payments are in dollars.",
            "product": "Валютный форвард",
            "claim_ids": ["c2", "o1"],
        }
    ],
    "overall_impact": "high",
    "overall_direction": "positive",
}

NUMBER_FINDING = {
    "passed": False,
    "issues": [
        {
            "code": "number_not_in_sources",
            "severity": "error",
            "claim_id": "r1",
            "message": "1,2 млрд руб. нет в контексте",
        }
    ],
}
OK = {"passed": True, "issues": []}
HEADLINE_FINDING = {
    "passed": False,
    "issues": [
        {
            "code": "number_not_in_sources",
            "severity": "error",
            "claim_id": "headline",
            "message": "the news reports a 1.2% gain, not 5%",
        }
    ],
}


def scenario(name: str) -> tuple[ScriptedLLM, str, str, str]:
    """(scripted model, profile_id, event_id, role) of a demo scenario."""
    settings = Settings(api_key="scripted", language="ru")
    if name == "rate":
        llm = ScriptedLLM(
            {
                "plan": [RATE_PLAN],
                "draft_insight": [_rate_draft(invented=True)],
                "verify": [NUMBER_FINDING, OK],
                "revise": [_rate_draft(invented=False)],
            },
            settings,
        )
        return llm, "demo_building_materials", "ev_rate_hold_ru", "cfo"
    if name == "injection":
        plan = {
            "rationale": "Новость об отрасли субъекта.",
            "steps": [
                {
                    "step_id": "s1",
                    "tool": "get_house_view",
                    "args": {"factors": ["consumer_demand"], "topics": ["consumption"]},
                    "purpose": "позиция по спросу",
                }
            ],
        }
        return (
            ScriptedLLM({"plan": [plan], "draft_insight": [INJECTION_DRAFT], "verify": [OK]}, settings),
            "demo_coffee_ip",
            "ev_injection_ru",
            "entrepreneur",
        )
    if name == "dollar":
        llm = ScriptedLLM(
            {
                "plan": [DOLLAR_PLAN],
                "draft_insight": [DOLLAR_DRAFT],
                "verify": [HEADLINE_FINDING],
                "revise": [DOLLAR_DRAFT],  # the revision fails to fix anything
                "headline": [{"headline": "A stronger dollar lifts sterling revenue from your US client"}],
            },
            Settings(api_key="scripted", language="en"),
        )
        return llm, "demo_london_studio", "ev_dollar_en", "cfo"
    raise KeyError(f"unknown demo scenario {name!r}; choose rate, injection or dollar")


def run_demo(name: str = "rate", data_dir: Path | None = None) -> AgentRun:
    llm, profile_id, event_id, role = scenario(name)
    store = DataStore.from_dir(data_dir or DEMO_DIR)
    store.today = datetime(2026, 10, 31, tzinfo=timezone.utc).date()
    agent = Agent(
        llm, store, settings=llm.settings, now=lambda: datetime(2026, 10, 31, 9, 0, tzinfo=timezone.utc)
    )
    return agent.run(profile_id, event_id, role=role)
