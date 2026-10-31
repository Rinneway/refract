"""Rule checks of a draft — free, deterministic and immune to model errors.

They catch structural violations (sources, origins, subject features) and numbers absent from the context. Semantic
contradictions are checked by a second model (`prompts/verify.py`); the orchestrator merges both.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from ..i18n import norm_lang, t
from ..schemas import (
    Claim,
    ImpactLevel,
    Insight,
    InsightContent,
    IssueCode,
    Origin,
    VerificationIssue,
    violations,
)
from ..textutil import invented_numbers, lexical_support, script_share
from .context import InsightContext, known_feature
from .impact import above

HEADLINE = "headline"
LANGUAGE_MIN = 0.8  # share of letters in the script of the run language


def visible_text(content: InsightContent) -> str:
    """Everything the reader sees as text: headline, statements, caveats, scenarios, actions."""
    parts = [content.headline]
    for c in content.all_claims():
        parts += [c.text, c.caveat or ""]
    parts += [f"{sc.name} {sc.description}" for sc in content.scenarios]
    parts += [f"{a.text} {a.rationale}" for a in content.actions]
    return " ".join(parts)


_MSG = {
    "features": {"ru": "неизвестные признаки субъекта: {bad}", "en": "unknown subject features: {bad}"},
    "features_hint": {
        "ru": "используй ключи из списка «Признаки»",
        "en": "use keys from the “Features” list",
    },
    "headline_numbers": {
        "ru": "в заголовке числа, которых нет в контексте: {nums}",
        "en": "numbers in the headline that are not in the context: {nums}",
    },
    "headline_hint": {
        "ru": "возьми числа из утверждений инсайта или убери их из заголовка",
        "en": "take numbers from the insight's statements or drop them from the headline",
    },
    "numbers": {
        "ru": "числа, которых нет в контексте: {nums}",
        "en": "numbers that are not in the context: {nums}",
    },
    "numbers_hint": {
        "ru": "убери число или возьми его из источника",
        "en": "drop the number or take it from a source",
    },
    "cap": {
        "ru": "уровень влияния выше предела {level} ({reason})",
        "en": "impact above the ceiling {level} ({reason})",
    },
    "cap_claims": {"ru": "; утверждения {ids}", "en": "; statements {ids}"},
    "cap_hint": {
        "ru": "понизь overall_impact и impact до {level} или ниже",
        "en": "lower overall_impact and impact to {level} or below",
    },
    "house_empty": {
        "ru": "есть тезисы института, но блок его позиции пуст",
        "en": "theses are available but the house_view block is empty",
    },
    "house_hint": {
        "ru": "добавь позицию института, если тезисы относятся к событию",
        "en": "add the house view if the theses concern the event",
    },
    "language": {
        "ru": "инсайт написан не по-русски: доля кириллицы {share:.0%}",
        "en": "the insight is not in English: {share:.0%} of letters are Latin",
    },
    "language_hint": {"ru": "перепиши все утверждения по-русски", "en": "rewrite every statement in English"},
    "confirmed": {
        "ru": "{cid}: замечание «{code}» снято — утверждение совпадает с источником",
        "en": "{cid}: finding “{code}” lifted — the statement matches its source",
    },
    "lifted": {"ru": " [снято кодом: совпадает с источником]", "en": " [lifted by code: matches the source]"},
}


def _m(key: str, lang: str, **kw: object) -> str:
    return _MSG[key][norm_lang(lang)].format(**kw)


def as_insight(content: InsightContent, ctx: InsightContext) -> Insight:
    """An insight built without validation, to list every violation instead of failing on the first."""
    return Insight.model_construct(
        **{k: getattr(content, k) for k in InsightContent.model_fields},
        insight_id="check",
        profile_id=ctx.profile.profile_id,
        event_id=ctx.event.event_id,
        language=ctx.lang,
        created_at=datetime.now(timezone.utc),
        sources=ctx.sources(),
    )


def rule_check(content: InsightContent, ctx: InsightContext) -> list[VerificationIssue]:
    lang = ctx.lang
    issues = list(violations(as_insight(content, ctx)))

    feats = ctx.features()
    for c in content.why_it_matters:
        bad = [k for k in c.features if not known_feature(k, feats)]
        if bad:
            issues.append(
                VerificationIssue(
                    code=IssueCode.NO_SUBJECT_LINK,
                    severity="error" if len(bad) == len(c.features) else "warning",
                    claim_id=c.claim_id,
                    message=_m("features", lang, bad=bad),
                    fix_hint=_m("features_hint", lang),
                )
            )

    context = ctx.render()
    if extra := invented_numbers([content.headline], context):
        issues.append(
            VerificationIssue(
                code=IssueCode.NUMBER_NOT_IN_SOURCES,
                severity="error",
                claim_id=HEADLINE,
                message=_m("headline_numbers", lang, nums=sorted(extra)),
                fix_hint=_m("headline_hint", lang),
            )
        )
    for c in content.all_claims():
        if extra := invented_numbers([c.text], context):
            issues.append(
                VerificationIssue(
                    code=IssueCode.NUMBER_NOT_IN_SOURCES,
                    severity="error",
                    claim_id=c.claim_id,
                    message=_m("numbers", lang, nums=sorted(extra)),
                    fix_hint=_m("numbers_hint", lang),
                )
            )

    if ctx.use_cap:
        cap = ctx.cap()
        over = [e.claim_id for e in [*content.risks, *content.opportunities] if above(e.impact, cap.level)]
        if above(content.overall_impact, cap.level) or over:
            issues.append(
                VerificationIssue(
                    code=IssueCode.NO_SUBJECT_LINK,
                    severity="warning",
                    message=_m("cap", lang, level=cap.level.value, reason=cap.reason)
                    + (_m("cap_claims", lang, ids=over) if over else ""),
                    fix_hint=_m("cap_hint", lang, level=cap.level.value),
                )
            )

    share = script_share(visible_text(content), lang)
    if share < LANGUAGE_MIN:
        issues.append(
            VerificationIssue(
                code=IssueCode.WRONG_LANGUAGE,
                severity="warning",
                message=_m("language", lang, share=share),
                fix_hint=_m("language_hint", lang),
            )
        )

    if not content.house_view and ctx.theses:
        issues.append(
            VerificationIssue(
                code=IssueCode.HOUSE_VIEW_WITHOUT_RESEARCH,
                severity="warning",
                message=_m("house_empty", lang),
                fix_hint=_m("house_hint", lang),
            )
        )
    return issues


def merge(*groups: list[VerificationIssue]) -> list[VerificationIssue]:
    seen, out = set(), []
    for group in groups:
        for i in group:
            key = (i.claim_id, i.code)
            if key not in seen:
                seen.add(key)
                out.append(i)
    return out


def errors(issues: list[VerificationIssue]) -> list[VerificationIssue]:
    return [i for i in issues if i.severity == "error"]


def drop_claims(content: InsightContent, claim_ids: set[str]) -> InsightContent:
    """Last line of defence: remove statements with unresolved errors and the actions that rely on them."""

    def keep(items):
        return [c for c in items if c.claim_id not in claim_ids]

    actions = [
        a for a in content.actions if a.action_id not in claim_ids and not set(a.claim_ids) & claim_ids
    ]
    return content.model_copy(
        update={
            "what_happened": keep(content.what_happened),
            "why_it_matters": keep(content.why_it_matters),
            "risks": keep(content.risks),
            "opportunities": keep(content.opportunities),
            "house_view": keep(content.house_view),
            "actions": actions,
        }
    )


def house_view_from_theses(ctx: InsightContext, removed: list[Claim]) -> list[Claim]:
    """The review removed the whole house view — return the theses the draft's author cited.

    The thesis statement is used, not the removed claim: thesis quotes were verified against the document. If the
    author left the house view empty (no research on the topic), nothing is added — theses picked only by a
    generic factor tend not to concern the event. Only theses in the run language are used.
    """
    cited = {s for c in removed for s in c.source_ids}
    # A thesis in another language would put, say, Russian text into an English insight — skip it.
    theses = [
        th
        for th in ctx.theses
        if f"src_{th.thesis_id}" in cited
        and not th.is_stale
        and norm_lang(th.language) == norm_lang(ctx.lang)
    ]
    return [
        Claim(
            claim_id=f"b{i}",
            text=th.statement,
            origin=Origin.HOUSE_VIEW,
            source_ids=[f"src_{th.thesis_id}"],
            confidence=1.0,
        )
        for i, th in enumerate(theses[:3], 1)
    ]


def short_headline(content: InsightContent, limit: int = 160) -> str:
    """Fallback headline: the first verified “why it matters” statement."""
    blocks = content.why_it_matters or content.what_happened
    if not blocks:
        return content.headline[:limit]
    text = blocks[0].text
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:—- ") + "…"


REQUIRED_BLOCKS = ("what_happened", "why_it_matters")


def keep_required(
    pruned: InsightContent, draft: InsightContent, hard: set[str], lang: str = "ru"
) -> tuple[InsightContent, list[str]]:
    """Required blocks must not become empty because of the second model's doubts.

    If the review removed all of “what happened” or “why it matters”, statements with only model findings come back
    with lower confidence and a caveat. `hard` are rule-check errors (unknown source, invented number): never restored.
    """
    notes = []
    for block in REQUIRED_BLOCKS:
        if getattr(pruned, block):
            continue
        back = [
            c.model_copy(
                update={"confidence": min(c.confidence, 0.5), "caveat": c.caveat or t("note.doubt", lang)}
            )
            for c in getattr(draft, block)
            if c.claim_id not in hard
        ]
        if back:
            pruned = pruned.model_copy(update={block: back})
            notes.append(t("note.restored_block", lang, block=block))
    return pruned, notes


SUPPORT_MIN = 0.85
# Connectives after which a statement turns into the author's own inference rather than a retelling of the source.
INFERENCE = re.compile(
    r"(то есть|что означает|это означает|а значит|значит|поэтому|следовательно|что будет|что может"
    r"|which means|this means|meaning that|therefore|hence|thus|so that|which will|which may|implying)",
    re.IGNORECASE,
)
_DISPUTABLE = {
    IssueCode.CONTRADICTION,
    IssueCode.NUMBER_NOT_IN_SOURCES,
    IssueCode.HOUSE_VIEW_WITHOUT_RESEARCH,
}


def cited_text(ctx: InsightContext, source_ids: list[str]) -> str:
    """Text of the sources a statement cites: the whole article, a thesis with its quote, a research excerpt."""
    texts: dict[str, str] = {}
    for a in ctx.event.articles:
        texts[f"src_{a.article_id}"] = f"{a.title}. {a.text or ''} {ctx.event.summary}"
    for r in ctx.related:
        for a in r.articles:
            texts[f"src_{a.article_id}"] = f"{a.title}. {r.summary}"
    for th in ctx.theses:
        texts[f"src_{th.thesis_id}"] = f"{th.statement} {th.quote}"
    for c in ctx.research:
        texts[f"src_{c.chunk_id}"] = c.text
    return " ".join(texts.get(s, "") for s in source_ids)


def confirm_by_sources(
    issues: list[VerificationIssue], content: InsightContent, ctx: InsightContext
) -> tuple[list[VerificationIssue], list[str]]:
    """Second-model findings against facts and the house view are checked against the cited source.

    If a statement is nearly verbatim from its source (≥ SUPPORT_MIN of stems), all its numbers are there, and it
    contains no inference connective, the finding is treated as nitpicking a paraphrase and becomes a warning: the
    statement stays. Other findings are untouched.
    """
    claims = {c.claim_id: c for c in content.all_claims()}
    out, notes = [], []
    for i in issues:
        c = claims.get(i.claim_id or "")
        if (
            i.severity == "error"
            and i.code in _DISPUTABLE
            and c is not None
            and c.origin in (Origin.FACT, Origin.HOUSE_VIEW)
            and not INFERENCE.search(c.text)
        ):
            source = cited_text(ctx, c.source_ids)
            if (
                source
                and not invented_numbers([c.text], source)
                and lexical_support(c.text, source) >= SUPPORT_MIN
            ):
                out.append(
                    i.model_copy(
                        update={"severity": "warning", "message": i.message + _m("lifted", ctx.lang)}
                    )
                )
                notes.append(_m("confirmed", ctx.lang, cid=c.claim_id, code=i.code.value))
                continue
        out.append(i)
    return out, notes


def clamp_impact(content: InsightContent, cap: ImpactLevel) -> tuple[InsightContent, list[str]]:
    """Impact never above the ceiling computed from the event–subject links."""
    changed: list[str] = []

    def fix(effects: list) -> list:
        out = []
        for e in effects:
            if above(e.impact, cap):
                changed.append(e.claim_id)
                e = e.model_copy(update={"impact": cap})
            out.append(e)
        return out

    update: dict = {"risks": fix(content.risks), "opportunities": fix(content.opportunities)}
    if above(content.overall_impact, cap):
        changed.insert(0, "overall_impact")
        update["overall_impact"] = cap
    return (content.model_copy(update=update) if changed else content), changed
