"""Rule checks of a draft — free, deterministic and immune to model errors.

They catch structural violations (sources, origins, subject features), numbers absent from the context, a risk
that says there is no effect, impact above the ceiling and text in the wrong language.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from ...schemas import Insight, InsightContent, IssueCode, Origin, VerificationIssue, violations
from ...textutil import first_percent, invented_numbers, normalize, script_share
from ..context import InsightContext, known_feature
from ..impact import above
from .messages import finding

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


# A risk or an opportunity that says there is none (“…but this is not a direct risk”).
_NO_EFFECT = re.compile(
    r"не является (?:прямым |существенным |реальным )?риском|прямого риска нет|риска нет|не несет риска"
    r"|is not a (?:direct |real |significant )?risk|not a direct risk|poses no (?:direct )?risk|no direct risk",
    re.IGNORECASE,
)


# “The largest supplier (14% of payments)” — a superlative about one counterparty followed by its share. Plurals
# (“the main suppliers”) and “one of the largest” are not checked: they do not name the top one.
_SUPERLATIVE = re.compile(
    r"(?<!один из )(?<!одного из )(?<!одному из )(?<!одним из )(?<!одном из )(?<!one of the )(?<!one of your )"
    r"(?<!one of its )\b(крупнейш\w*|самы\w* крупн\w*|главн\w*|основн\w*|largest|biggest|main|principal|top)"
    r"\s+(?:[\w-]+\s+){0,2}?(поставщик\w*|покупател\w*|клиент\w*|заказчик\w*|supplier|customer|client|buyer)"
    r"(s?)\b"
)
_RU_PLURAL = re.compile(r"(?:поставщик|покупател|клиент|заказчик)(?:и|ов|ам|ами|ах|ей|ям|ями|ях)$")
_SHARE_WINDOW = 70  # characters after the role word where its share is written


def counterparty_superlatives(text: str, cid: str, ctx: InsightContext) -> list[VerificationIssue]:
    """A statement that calls a counterparty the largest must give the share of the largest one.

    Small models write “the largest supplier (14%)” about the supplier the news concerns, while another one has
    22%. Flagged only when the share given belongs to another counterparty of the subject — a percentage that is no
    counterparty's share is about something else (“the largest customer's sales fell 20%”).
    """
    cps = ctx.profile.counterparties
    if not cps:
        return []
    lang, out = ctx.lang, []
    low = normalize(text)
    for m in _SUPERLATIVE.finditer(low):
        word = m.group(2)
        if m.group(3) or _RU_PLURAL.search(word):
            continue
        role = "supplier" if word.startswith(("поставщик", "supplier")) else "customer"
        written = first_percent(low[m.end() : m.end() + _SHARE_WINDOW])
        if written is None:
            continue
        token, readings = written

        def share_is(c, readings=readings) -> bool:
            return any(abs(x - c.share_of_flow * 100) < 0.5 for x in readings)

        mine = [c for c in cps if c.role == role]
        leader = max(mine, key=lambda c: c.share_of_flow) if mine else None
        if leader and any(share_is(c) for c in mine if c.share_of_flow == leader.share_of_flow):
            continue  # the largest one (ties included)
        other = next((c for c in cps if share_is(c)), None)
        if other is None:
            continue
        out.append(
            VerificationIssue(
                code=IssueCode.CONTRADICTION,
                severity="error",
                claim_id=cid,
                message=finding(
                    "superlative" if leader else "superlative_none",
                    lang,
                    cid=cid,
                    phrase=m.group(0),
                    pct=token,
                    other=f"{finding('of_' + other.role, lang)} {other.counterparty_id}",
                    role=finding(role, lang),
                    top=leader.counterparty_id if leader else "",
                    top_pct=f"{leader.share_of_flow * 100:.0f}" if leader else "",
                ),
                fix_hint=finding("superlative_hint", lang, pct=token),
            )
        )
    return out


# Advice that does not fit the kind of subject: a business step for a private person (gemma4:12b suggested an SME
# loan to a household saver), staff or VAT for the self-employed, a household step for a company.
_BUSINESS = (
    r"\bмсб\b|малого (?:и среднего )?бизнес|малому бизнесу|для бизнеса|\bип\b|юрлиц|юридическ\w* лиц|контрагент|"
    r"поставщик|\bндс\b|\bусн\b|выручк|оборотн\w* (?:средств|капитал)|факторинг|эквайринг|зарплатн\w* проект|"
    r"расч[её]тн\w* сч[её]т|сотрудник|\bsmes?\b|small business|for (?:your )?business|\bvat\b|revenue|"
    r"working capital|supplier|counterpart|factoring|acquiring|payroll|employees|business account"
)
_STAFF_VAT = r"\bндс\b|сотрудник|зарплатн|юрлиц|\bvat\b|employees|payroll|staff"
_HOUSEHOLD = (
    r"налогов\w* вычет|семейн\w* бюджет|личн\w* бюджет|пенсионн\w* (?:накоплен|сч[её]т)|\bиис\b|"
    r"household budget|personal budget|family budget|tax deduction|pension savings"
)
_NOT_FOR = {
    "individual": (re.compile(_BUSINESS), "kind_business"),
    "self_employed": (re.compile(_STAFF_VAT), "kind_staff"),
    "company": (re.compile(_HOUSEHOLD), "kind_household"),
}


def unfit_actions(content: InsightContent, ctx: InsightContext) -> list[VerificationIssue]:
    """An action whose own text gives advice this kind of subject cannot follow (rationales may explain, so only
    the step itself is checked)."""
    rule = _NOT_FOR.get(ctx.profile.kind.value)
    if rule is None:
        return []
    pattern, key = rule
    out = []
    for a in content.actions:
        m = pattern.search(normalize(a.text))
        if m:
            out.append(
                VerificationIssue(
                    code=IssueCode.CONTRADICTION,
                    severity="error",
                    claim_id=a.action_id,
                    message=finding(key, ctx.lang, cid=a.action_id, phrase=m.group(0)),
                    fix_hint=finding("kind_hint", ctx.lang),
                )
            )
    return out


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
                    message=finding("features", lang, bad=bad),
                    fix_hint=finding("features_hint", lang),
                )
            )

    for c in content.why_it_matters:
        if c.origin is Origin.HOUSE_VIEW:
            issues.append(
                VerificationIssue(
                    code=IssueCode.RETELLING,
                    severity="warning",
                    claim_id=c.claim_id,
                    message=finding("why_house", lang, cid=c.claim_id),
                    fix_hint=finding("why_house_hint", lang),
                )
            )

    for e in [*content.risks, *content.opportunities]:
        if _NO_EFFECT.search(normalize(e.text)):
            issues.append(
                VerificationIssue(
                    code=IssueCode.SHALLOW,
                    severity="error",
                    claim_id=e.claim_id,
                    message=finding("no_effect", lang, cid=e.claim_id),
                    fix_hint=finding("no_effect_hint", lang),
                )
            )

    texts = [(HEADLINE, content.headline), *((c.claim_id, c.text) for c in content.all_claims())]
    texts += [(a.action_id, f"{a.text} {a.rationale or ''}") for a in content.actions]
    for cid, text in texts:
        issues += counterparty_superlatives(text, cid, ctx)
    issues += unfit_actions(content, ctx)

    context = ctx.render()
    if extra := invented_numbers([content.headline], context):
        issues.append(
            VerificationIssue(
                code=IssueCode.NUMBER_NOT_IN_SOURCES,
                severity="error",
                claim_id=HEADLINE,
                message=finding("headline_numbers", lang, nums=sorted(extra)),
                fix_hint=finding("headline_hint", lang),
            )
        )
    for c in content.all_claims():
        if extra := invented_numbers([c.text], context):
            issues.append(
                VerificationIssue(
                    code=IssueCode.NUMBER_NOT_IN_SOURCES,
                    severity="error",
                    claim_id=c.claim_id,
                    message=finding("numbers", lang, nums=sorted(extra)),
                    fix_hint=finding("numbers_hint", lang),
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
                    message=finding("cap", lang, level=cap.level.value, reason=cap.reason)
                    + (finding("cap_claims", lang, ids=over) if over else ""),
                    fix_hint=finding("cap_hint", lang, level=cap.level.value),
                )
            )

    share = script_share(visible_text(content), lang)
    if share < LANGUAGE_MIN:
        issues.append(
            VerificationIssue(
                code=IssueCode.WRONG_LANGUAGE,
                severity="warning",
                message=finding("language", lang, share=share),
                fix_hint=finding("language_hint", lang),
            )
        )

    if not content.house_view and ctx.theses:
        issues.append(
            VerificationIssue(
                code=IssueCode.HOUSE_VIEW_WITHOUT_RESEARCH,
                severity="warning",
                message=finding("house_empty", lang),
                fix_hint=finding("house_hint", lang),
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
