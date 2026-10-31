"""Every step has a prompt in both languages; English system prompts contain no Russian text."""

import re

import pytest

from refract.agent.context import InsightContext
from refract.config import Institution
from refract.demo import _rate_draft
from refract.eval import judge as jp
from refract.knowledge.theses import page_messages
from refract.prompts import chat, classify, headline, insight, plan, revise, verify
from refract.schemas import InsightContent, IssueCode, VerificationIssue

CYR = re.compile(r"[а-яА-ЯёЁ]")


@pytest.fixture
def ctx(store):
    def make(lang: str) -> InsightContext:
        return InsightContext(
            store.profiles["demo_london_studio"],
            store.events["ev_dollar_en"],
            theses=store.theses[:2],
            lang=lang,
        )

    return make


def systems(store, ctx, lang: str) -> dict[str, str]:
    c = ctx(lang)
    draft = InsightContent.model_validate(_rate_draft(False))
    issue = VerificationIssue(code=IssueCode.CONTRADICTION, severity="error", claim_id="c2", message="x")
    from refract.demo import run_demo

    ins = run_demo("rate").insight
    inst = c.inst()
    return {
        "classify": classify.build_messages(c.event, lang)[0]["content"],
        "plan": plan.build_messages(c.profile, c.event, lang, Institution())[0]["content"],
        "insight": insight.build_messages(c)[0]["content"],
        "verify": verify.build_messages(c, draft)[0]["content"],
        "revise": revise.build_messages(c, draft, [issue])[-1]["content"],
        "headline": headline.build_messages(draft, ["x"], lang, inst)[0]["content"],
        "chat_plan": chat.plan_messages(ins, c, "q", [])[0]["content"],
        "chat": chat.answer_messages(ins, c, "q", [], "")[0]["content"],
        "judge": jp.judge_messages("ctx", "ans", lang, inst)[0]["content"],
        "baseline": jp.baseline_messages("ctx", lang)[0]["content"],
        "theses": page_messages("text", 1, "doc", lang)[0]["content"],
    }


def test_english_prompts_are_english(store, ctx) -> None:
    for step, text in systems(store, ctx, "en").items():
        assert not CYR.search(text), f"{step}: {CYR.search(text).group(0)!r} in the English prompt"


def test_russian_prompts_are_russian(store, ctx) -> None:
    for step, text in systems(store, ctx, "ru").items():
        assert len(CYR.findall(text)) > 50, step


def test_context_is_localised(ctx) -> None:
    en, ru = ctx("en").render(), ctx("ru").render()
    assert "## Subject" in en and "Impact ceiling" in en and "Demo Bank view" in en
    assert "## Субъект" in ru and "Предельный уровень влияния" in ru and "Позиция Демо-банка" in ru
