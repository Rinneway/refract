"""The revise step: fix the draft by the review findings."""

from __future__ import annotations

from ..i18n import norm_lang
from ..schemas import InsightContent, VerificationIssue
from . import insight

VERSION = "revise-v2"

TASK = {
    "ru": """Ниже — твой черновик и замечания проверки. Исправь черновик:
- каждое замечание уровня error устрани: исправь утверждение по источнику, поменяй origin, добавь признаки \
субъекта или ссылки — а если подтвердить нельзя, удали утверждение и действия, которые на него ссылаются;
- замечания warning учти, если это не требует новых фактов;
- не добавляй фактов и чисел, которых нет в контексте; сохрани claim_id исправленных утверждений.
Верни полный исправленный инсайт строго по JSON-схеме.""",
    "en": """Below are your draft and the review findings. Fix the draft:
- resolve every error-level finding: correct the statement by its source, change its origin, add subject features \
or citations — and if it cannot be supported, delete the statement and the actions that rely on it;
- take warnings into account if that needs no new facts;
- do not add facts or numbers absent from the context; keep the claim_id of corrected statements.
Return the full corrected insight strictly by the JSON schema.""",
}
FINDINGS = {
    "ru": ("Замечания", "весь инсайт", "Как исправить"),
    "en": ("Findings", "whole insight", "How to fix"),
}


def build_messages(ctx, draft: InsightContent, issues: list[VerificationIssue]) -> list[dict[str, str]]:
    lang = norm_lang(ctx.lang)
    title, whole, how = FINDINGS[lang]
    notes = "\n".join(
        f"- [{i.severity}] {i.claim_id or whole}: {i.code.value} — {i.message}"
        + (f" {how}: {i.fix_hint}" if i.fix_hint else "")
        for i in issues
    )
    return [
        *insight.build_messages(ctx),
        {"role": "assistant", "content": draft.model_dump_json()},
        {"role": "user", "content": f"{TASK[lang]}\n\n{title}:\n{notes}"},
    ]
