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
    SourceType,
    VerificationIssue,
    violations,
)
from ..textutil import (
    invented_numbers,
    lexical_support,
    normalize,
    numbers,
    script_share,
    stems,
    trim_sentence,
)
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
    "structural": {
        "ru": "{cid}: замечание «{code}» снято — код проверяет это сам и ошибки не нашёл",
        "en": "{cid}: finding “{code}” lifted — code checks this itself and found no problem",
    },
    "overruled": {
        "ru": " [снято кодом: структура в порядке]",
        "en": " [lifted by code: the structure is fine]",
    },
    "why_house": {
        "ru": "{cid}: в «почему важно» — позиция института; она уже в блоке house_view, здесь нужен вывод для субъекта",
        "en": "{cid}: “why it matters” repeats the house view, which belongs in house_view; say what it means here",
    },
    "why_house_hint": {
        "ru": "origin=ai: что эта позиция значит для субъекта, через его признаки",
        "en": "origin=ai: what the position means for the subject, through its features",
    },
    "softened": {
        "ru": " [понижено кодом: сама проверка называет это формулировкой, а не ошибкой]",
        "en": " [downgraded by code: the review itself calls this wording, not an error]",
    },
    "softened_note": {
        "ru": "{cid}: «{code}» понижено до предупреждения — {why}",
        "en": "{cid}: “{code}” downgraded to a warning — {why}",
    },
    "why_wording": {"ru": "замечание о формулировке", "en": "a remark on wording"},
    "why_assumption": {
        "ru": "это допущение — станет гипотезой",
        "en": "an assumption — becomes a hypothesis",
    },
    "why_unattributed": {
        "ru": "утверждение не приписано институту",
        "en": "the statement is not attributed to the institution",
    },
    "why_cited": {"ru": "ссылка на аналитику есть", "en": "a research source is cited"},
    "no_effect": {
        "ru": "{cid}: это не риск и не возможность — утверждение говорит, что эффекта нет",
        "en": "{cid}: neither a risk nor an opportunity — the statement says there is no effect",
    },
    "no_effect_hint": {
        "ru": "удали утверждение или перенеси мысль в why_it_matters",
        "en": "delete it or move the thought to why_it_matters",
    },
}
# A risk or an opportunity that says there is none (“…but this is not a direct risk”).
_NO_EFFECT = re.compile(
    r"не является (?:прямым |существенным |реальным )?риском|прямого риска нет|риска нет|не несет риска"
    r"|is not a (?:direct |real |significant )?risk|not a direct risk|poses no (?:direct )?risk|no direct risk",
    re.IGNORECASE,
)


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

    for c in content.why_it_matters:
        if c.origin is Origin.HOUSE_VIEW:
            issues.append(
                VerificationIssue(
                    code=IssueCode.RETELLING,
                    severity="warning",
                    claim_id=c.claim_id,
                    message=_m("why_house", lang, cid=c.claim_id),
                    fix_hint=_m("why_house_hint", lang),
                )
            )

    for e in [*content.risks, *content.opportunities]:
        if _NO_EFFECT.search(normalize(e.text)):
            issues.append(
                VerificationIssue(
                    code=IssueCode.SHALLOW,
                    severity="error",
                    claim_id=e.claim_id,
                    message=_m("no_effect", lang, cid=e.claim_id),
                    fix_hint=_m("no_effect_hint", lang),
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
    """Last line of defence: remove statements with unresolved errors and the actions left without support.

    An action that relied on several statements stays while at least one of them stays, citing only those.
    """

    def keep(items):
        return [c for c in items if c.claim_id not in claim_ids]

    actions = [
        a.model_copy(update={"claim_ids": [c for c in a.claim_ids if c not in claim_ids]})
        for a in content.actions
        if a.action_id not in claim_ids and not (a.claim_ids and set(a.claim_ids) <= claim_ids)
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
    """Fallback headline: the first verified conclusion about the subject, cut at a clause boundary if long.

    A model's own conclusion (ai, hypothesis) is preferred: a house view repeated in “why it matters” describes the
    market, not the reader.
    """
    own = [c for c in content.why_it_matters if c.origin in (Origin.AI, Origin.HYPOTHESIS)]
    blocks = own or content.why_it_matters or content.what_happened
    if not blocks:
        return trim_sentence(content.headline, limit)
    return trim_sentence(blocks[0].text, limit)


REQUIRED_BLOCKS = ("what_happened", "why_it_matters")


def keep_required(
    pruned: InsightContent,
    draft: InsightContent,
    hard: set[str],
    lang: str = "ru",
    dropped: set[str] | frozenset[str] = frozenset(),
) -> tuple[InsightContent, list[str]]:
    """Required blocks must not become empty because of the second model's doubts.

    If the review removed all of “what happened” or “why it matters”, statements with only model findings come back
    with lower confidence and a caveat. `hard` are rule-check errors (unknown source, invented number): never restored.
    Actions removed only because they relied on a restored statement come back too; `dropped` are the ids flagged by
    the review — an action flagged itself stays removed.
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
    if notes:
        present = {c.claim_id for c in pruned.all_claims()}
        kept = {a.action_id for a in pruned.actions}
        actions = [
            a
            for a in draft.actions
            if a.action_id in kept
            or (a.action_id not in dropped and a.claim_ids and set(a.claim_ids) <= present)
        ]
        if len(actions) > len(pruned.actions):
            pruned = pruned.model_copy(update={"actions": actions})
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


# What code checks exactly: a second-model error of this kind stands only if the rule check found it too.
_STRUCTURAL = {
    IssueCode.UNKNOWN_SOURCE,
    IssueCode.MISSING_SOURCE,
    IssueCode.NUMBER_NOT_IN_SOURCES,  # “a number absent from the context” is exactly what the number check tests
    IssueCode.DUPLICATE_ID,
    IssueCode.SCHEMA_ERROR,
    IssueCode.WRONG_LANGUAGE,
}


def overrule_structural(
    issues: list[VerificationIssue], rule: list[VerificationIssue], lang: str = "ru"
) -> tuple[list[VerificationIssue], list[str]]:
    """Small reviewers often claim a statement has no sources, or a number absent from the context, when the rule
    check finds neither. Such findings become warnings."""
    found = {(i.code, i.claim_id) for i in rule}
    out, notes = [], []
    for i in issues:
        if i.severity == "error" and i.code in _STRUCTURAL and (i.code, i.claim_id) not in found:
            out.append(
                i.model_copy(update={"severity": "warning", "message": i.message + _m("overruled", lang)})
            )
            notes.append(_m("structural", lang, cid=i.claim_id or "—", code=i.code.value))
            continue
        out.append(i)
    return out, notes


# The review's own words that it found a wording problem, not wrong information (quoted fragments are ignored).
_QUOTED = re.compile(r'"[^"]*"|«[^»]*»|“[^”]*”')
_WORDING = re.compile(
    r"не противоречит|не является противоречием|это не противоречие|слишком (?:категорич|сильн|уверен|общ)"
    r"|излишне (?:категорич|уверен)|чрезмерно (?:категорич|уверен)|смягч|более осторожн|осторожнее"
    r"|(?:является|это|лишь)\s+(?:\w+\s+){0,2}(?:интерпретаци|обобщени)|перефразир|не прямой цитат"
    r"|(?:необходимо|следует|нужно) уточнить|свершивш\w* факт|реш[её]нн\w* дел|установленн\w* факт"
    r"|подан\w* как (?:факт|свершивш|реш[её]нн)"
    r"|presented as (?:a )?(?:fact|settled|done)|not (?:yet )?(?:a |an )?(?:settled|established|accomplished) fact"
    r"|done deal|not yet law"
    r"|does not contradict|doesn't contradict|is not a contradiction|not a contradiction"
    r"|too (?:strong|categorical|certain|confident|broad)|overly (?:strong|categorical|certain|confident)"
    r"|overstat|soften|more cautious|(?:is|an?|only)\s+(?:\w+\s+){0,2}(?:interpretation|generali[sz]ation)"
    r"|paraphras|should be clarified|needs? clarification",
    re.IGNORECASE,
)
_ASSUMPTION = re.compile(
    r"предположени|допущени|не может быть выведено как факт|нет данных, подтверждающих"
    r"|не подтвержда\w* (?:данными|профилем)|в профиле (?:\w+\s+){0,3}нет (?:данных|информации)"
    r"|an assumption|is assumed|speculat|not supported by (?:the )?(?:data|profile)|no data (?:to )?(?:support|confirm)"
    r"|the profile (?:does not|has no)",
    re.IGNORECASE,
)
_NO_SOURCE = re.compile(
    r"нет (?:явной )?ссылк|нет ссылок|отсутству\w* ссылк|без ссылк|не указан\w* источник"
    r"|no (?:source|citation|reference)|does not cite|missing (?:source|citation)",
    re.IGNORECASE,
)
_RECONCILED = {IssueCode.CONTRADICTION, IssueCode.HOUSE_VIEW_WITHOUT_RESEARCH}


def _names_institution(text: str, ctx: InsightContext) -> bool:
    inst, low = ctx.institution, normalize(text)
    stem = normalize(inst.name_ru)[:-1] if len(inst.name_ru) > 4 else normalize(inst.name_ru)
    return stem in low or any(normalize(n) in low for n in (inst.name, inst.name_gen) if n)


def reconcile(
    issues: list[VerificationIssue], content: InsightContent, ctx: InsightContext
) -> tuple[list[VerificationIssue], list[str], dict[str, str]]:
    """Make the reviewer's severity agree with its own explanation (small reviewers often do not).

    An error on contradiction or the house view becomes a warning when
    - the message says the statement is an assumption — at finalisation it is relabelled as a hypothesis;
    - the message itself says it is a wording problem (“does not contradict, but too categorical”, “soften”,
      “generalisation”) — the statement gets a caveat at finalisation;
    - it is a house-view finding on a statement that does not credit the institution (a labelling issue);
    - it is a house-view finding about a missing source while the statement cites the institution's research.
    Numbers are never reconciled: invented numbers are rule-check errors. Returns the issues, trace notes and
    {claim_id: "wording" | "assumption"} for the statements to soften.
    """
    lang = ctx.lang
    claims = {c.claim_id: c for c in content.all_claims()}
    research = {s.source_id for s in ctx.sources() if s.type is SourceType.RESEARCH}
    context = ctx.render()
    with_numbers = {cid for cid, c in claims.items() if invented_numbers([c.text], context)} | (
        {HEADLINE} if invented_numbers([content.headline], context) else set()
    )
    out, notes, kinds = [], [], {}
    for i in issues:
        cid = i.claim_id or ""
        c = claims.get(cid)
        why, kind = None, None
        if (
            i.severity == "error"
            and i.code in _RECONCILED
            and (c is not None or cid == HEADLINE)
            and cid not in with_numbers
        ):
            text = _QUOTED.sub(" ", f"{i.message} {i.fix_hint or ''}")
            if _ASSUMPTION.search(text):
                why, kind = "why_assumption", ("wording" if cid == HEADLINE else "assumption")
            elif _WORDING.search(text):
                why, kind = "why_wording", "wording"
            elif i.code is IssueCode.HOUSE_VIEW_WITHOUT_RESEARCH and c is not None:
                if c.origin is not Origin.HOUSE_VIEW and not _names_institution(c.text, ctx):
                    why = "why_unattributed"
                elif _NO_SOURCE.search(text) and set(c.source_ids) & research:
                    why = "why_cited"
        if why is None:
            out.append(i)
            continue
        out.append(i.model_copy(update={"severity": "warning", "message": i.message + _m("softened", lang)}))
        notes.append(_m("softened_note", lang, cid=cid, code=i.code.value, why=_m(why, lang)))
        if kind and kinds.get(cid) != "assumption":
            kinds[cid] = kind
    return out, notes, kinds


def for_revision(issues: list[VerificationIssue], kinds: dict[str, str]) -> list[VerificationIssue]:
    """What the revision sees: every error, plus the reconciled remarks — when a round runs anyway, the model
    softens the wording itself instead of code adding a caveat."""
    return [
        i for i in issues if i.severity == "error" or (i.code in _RECONCILED and (i.claim_id or "") in kinds)
    ]


def soften(
    content: InsightContent, kinds: dict[str, str], lang: str = "ru"
) -> tuple[InsightContent, list[str]]:
    """Apply what `reconcile` found: a caveat and lower confidence; an assumption becomes a hypothesis."""
    done: list[str] = []

    def fix(items: list) -> list:
        out = []
        for c in items:
            kind = kinds.get(c.claim_id)
            if kind == "assumption" and c.origin in (Origin.AI, Origin.FACT):
                c = c.model_copy(
                    update={
                        "origin": Origin.HYPOTHESIS,
                        "caveat": c.caveat or t("note.assumption", lang),
                        "confidence": min(c.confidence, 0.5),
                    }
                )
                done.append(c.claim_id)
            elif kind:
                c = c.model_copy(
                    update={
                        "caveat": c.caveat or t("note.overstated", lang),
                        "confidence": min(c.confidence, 0.6),
                    }
                )
                done.append(c.claim_id)
            out.append(c)
        return out

    update = {
        b: fix(getattr(content, b))
        for b in ("what_happened", "why_it_matters", "risks", "opportunities", "house_view")
    }
    if not done:
        return content, []
    return content.model_copy(update=update), [t("note.softened", lang, ids=done)]


DUPLICATE_MIN = 0.6


def dedupe_house_view(content: InsightContent, lang: str = "ru") -> tuple[InsightContent, list[str]]:
    """A house view repeated in “why it matters” is removed there, if the block keeps another statement."""
    dup = [
        w.claim_id
        for w in content.why_it_matters
        if w.origin is Origin.HOUSE_VIEW
        and any(lexical_support(w.text, b.text) >= DUPLICATE_MIN for b in content.house_view)
    ]
    if not dup or len(dup) >= len(content.why_it_matters):
        return content, []
    return drop_claims(content, set(dup)), [t("note.duplicate", lang, ids=dup)]


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


# ---------------------------------------------------------------------------------------- attribution repair

_PREFIX = re.compile(r"^\s*(риск|возможность|risk|opportunity)\s*[:—-]\s*", re.IGNORECASE)
_ATTR = {
    "number_source": {
        "ru": "{cid}: добавлен источник числа — {ids}",
        "en": "{cid}: added the source of its number — {ids}",
    },
    "mapped": {
        "ru": "{cid}: ключи признаков приведены к списку — {old} → {new}",
        "en": "{cid}: feature keys mapped to the list — {old} → {new}",
    },
    "features": {
        "ru": "{cid}: признаки определены кодом по тексту — {keys}",
        "en": "{cid}: features inferred by code from the text — {keys}",
    },
    "sources": {
        "ru": "{cid}: источник определён кодом по тексту — {ids}",
        "en": "{cid}: source inferred by code from the text — {ids}",
    },
    "fallback_why": {
        "ru": "блок why_it_matters пуст — добавлена связь события с субъектом, рассчитанная кодом",
        "en": "why_it_matters was empty — added the event–subject link computed by code",
    },
    "fallback_what": {
        "ru": "блок what_happened пуст — добавлен факт из новости",
        "en": "what_happened was empty — added a fact from the news",
    },
    "fallback_caveat": {
        "ru": "Сформулировано кодом по признакам субъекта: модель не дала подтверждённого вывода.",
        "en": "Worded by code from the subject's features: the model gave no verified conclusion.",
    },
}
FEATURE_MIN_SCORE = 2  # shared word stems, a shared number or a concept word counts double
SOURCE_MIN_SUPPORT = 0.5
_TAX = ("налог", "ндс", "усн", "упрощ", "патент", "ндфл", "нпд", "vat", "tax", "simplified")
# Words that name a feature without sharing a stem with its description (“НДС” for a tax regime). Prefixes of
# lowercase words; a feature key matches a concept key exactly or as its prefix (counterparties.supplier.*).
CONCEPTS: dict[str, tuple[str, ...]] = {
    "tax_regime": _TAX,
    "taxes": _TAX,
    "scale.revenue": ("выручк", "оборот", "revenue", "turnover", "sales"),
    "scale.employees": ("сотрудн", "персонал", "штат", "employee", "staff", "headcount"),
    "products.loans": ("кредит", "заем", "займ", "loan", "borrow"),
    "products.acquiring": ("эквайр", "acquir"),
    "products.deposits": ("депозит", "вклад", "deposit"),
    "products.leasing": ("лизинг", "leasing", "lease"),
    "products.fx_account": ("валютн", "currency account"),
    "products.payroll": ("зарплатн", "payroll"),
    "foreign_trade": ("экспорт", "импорт", "export", "import"),
    "counterparties.supplier": ("поставщ", "закуп", "supplier", "purchas"),
    "counterparties.customer": ("покупател", "клиент", "customer", "buyer"),
    "policy_rate": ("ставк", "кредит", "borrow"),
    "credit_market": ("кредит", "ипотек", "loan", "mortgag"),
    "inflation": ("инфляц", "подорож", "inflation"),
    "consumer_demand": ("спрос", "потребител", "трат", "demand", "spending"),
    "labor_market": ("зарплат", "персонал", "wage", "salar", "hiring"),
    "sme_support": ("поддержк", "субсид", "льгот", "support", "subsid", "grant"),
}


def _concepts(key: str) -> tuple[str, ...]:
    for k, words in CONCEPTS.items():
        if key == k or key.startswith(k + "."):
            return words
    return ()


def _feature_score(text: str, description: str, key: str = "") -> int:
    nums = {n for n in numbers(text) if not (n.is_integer() and n <= 12)}
    score = len(stems(text) & stems(description)) + 2 * len(nums & numbers(description))
    words = re.findall(r"[а-яa-z]+", normalize(text))
    if any(w.startswith(c) for c in _concepts(key) for w in words):
        score += 2
    return score


def infer_features(text: str, features: list[tuple[str, str]], limit: int = 3) -> list[str]:
    """Feature keys whose description the statement visibly refers to (shared words, numbers, concept words).

    Keys within one point of the best score are kept, at most `limit`.
    """
    scored = [(_feature_score(text, desc, key), key) for key, desc in features if key != "products.absent"]
    best = sorted((x for x in scored if x[0] >= FEATURE_MIN_SCORE), key=lambda x: -x[0])
    if not best:
        return []
    top = best[0][0]
    return [key for score, key in best[:limit] if score >= top - 1]


def _source_texts(ctx: InsightContext) -> dict[str, str]:
    texts = {}
    for a in ctx.event.articles:
        texts[f"src_{a.article_id}"] = f"{a.title}. {a.text or ''} {ctx.event.summary}"
    for th in ctx.theses:
        texts[f"src_{th.thesis_id}"] = f"{th.statement} {th.quote}"
    for c in ctx.research:
        texts[f"src_{c.chunk_id}"] = c.text
    for r in ctx.related:
        for a in r.articles:
            texts[f"src_{a.article_id}"] = f"{a.title}. {r.summary}"
    return texts


# Feature keys small models use instead of the listed ones.
FEATURE_ALIASES = {
    "sector": "industry", "industry_code": "industry", "okved": "industry", "nace": "industry",
    "activity": "industry", "business": "industry", "revenue": "scale.revenue", "turnover": "scale.revenue",
    "sales": "scale.revenue", "employees": "scale.employees", "staff": "scale.employees",
    "headcount": "scale.employees", "segment": "scale.segment", "size": "scale.segment", "tax": "tax_regime",
    "taxation": "tax_regime", "tax_system": "tax_regime", "location": "region", "city": "region",
    "import": "foreign_trade", "imports": "foreign_trade", "export": "foreign_trade", "exports": "foreign_trade",
    "trade": "foreign_trade",
}  # fmt: skip
SOURCE_WITH_PROFILE = 0.25  # an inference about the subject also rests on the source it partly retells


def resolve_feature(key: str, features: list[tuple[str, str]]) -> str | None:
    """A listed key for `key`: itself if known, an alias (sector → industry), or the one key ending in it
    (loans → products.loans, cp_dev_101 → counterparties.customer.cp_dev_101)."""
    if known_feature(key, features):
        return key
    k = key.strip().lower().replace(" ", "_")
    alias = FEATURE_ALIASES.get(k)
    if alias and known_feature(alias, features):
        return alias
    ends = [fk for fk, _ in features if fk.endswith("." + k)]
    return ends[0] if len(ends) == 1 else None


def attribute(content: InsightContent, ctx: InsightContext) -> tuple[InsightContent, list[str]]:
    """Repair what small models often leave out, before the checks run.

    - a statement without `features` (or with only unknown keys) gets the keys whose descriptions it visibly
      refers to — shared word stems or numbers; nothing is attached if nothing matches;
    - unknown feature keys are mapped to listed ones (sector → industry, loans → products.loans) or dropped;
    - a statement without sources gets the source that contains all its numbers, else the source it is nearly a
      retelling of (lexical support ≥ 0.5); a statement about the subject also cites the subject data, together
      with the source it partly retells (≥ 0.25);
    - “Risk:” / “Возможность:” prefixes are removed from the text.

    Every change is reported for the trace; nothing is invented — only links to text that is already in context.
    """
    feats, lang = ctx.features(), ctx.lang
    profile_id = "src_profile"
    texts = {**_source_texts(ctx), profile_id: " ".join([ctx.profile.summary, *(d for _, d in feats)])}
    notes: list[str] = []

    def fix(c, need_features: bool):
        update: dict = {}
        text = _PREFIX.sub("", c.text)
        if text != c.text and text:
            update["text"] = text[:1].upper() + text[1:]
        features = c.features
        if need_features and features and not all(known_feature(k, feats) for k in features):
            resolved = list(dict.fromkeys(r for k in features if (r := resolve_feature(k, feats))))
            if resolved != features:
                notes.append(_ATTR["mapped"][lang].format(cid=c.claim_id, old=features, new=resolved))
                update["features"] = features = resolved
        missing = need_features and not any(known_feature(k, feats) for k in features)
        if missing and (keys := infer_features(c.text, feats)):
            update["features"] = features = keys
            notes.append(_ATTR["features"][lang].format(cid=c.claim_id, keys=keys))
        scored = sorted(((lexical_support(c.text, t), sid) for sid, t in texts.items()), reverse=True)
        has_numbers = bool(invented_numbers([c.text], ""))
        if not c.source_ids:
            by_numbers = [
                sid for _, sid in scored if has_numbers and not invented_numbers([c.text], texts[sid])
            ]
            if by_numbers:
                # A number found in several sources (a 14% forecast and a 14% payment share) cites both.
                ids = by_numbers[:2]
                update["source_ids"] = ids + ([profile_id] if features and profile_id not in ids else [])
            elif (
                scored
                and scored[0][0] >= SOURCE_MIN_SUPPORT
                and not invented_numbers([c.text], texts[scored[0][1]])
            ):
                update["source_ids"] = [scored[0][1]]
            elif features:
                near = [
                    sid
                    for score, sid in scored[:1]
                    if score >= SOURCE_WITH_PROFILE
                    and sid != profile_id
                    and not invented_numbers([c.text], texts[sid])
                ]
                update["source_ids"] = [*near, profile_id]
            if update.get("source_ids"):
                notes.append(_ATTR["sources"][lang].format(cid=c.claim_id, ids=update["source_ids"]))
        elif has_numbers:
            # A number the cited sources do not contain is cited from where it is: “rates stay at 14%” citing only
            # the subject data gets the forecast that says 14%.
            cited = " ".join(texts.get(sid, "") for sid in c.source_ids)
            if invented_numbers([c.text], cited):
                more = [
                    sid
                    for _, sid in scored
                    if sid not in c.source_ids and not invented_numbers([c.text], f"{cited} {texts[sid]}")
                ]
                if more:
                    update["source_ids"] = [*c.source_ids, more[0]]
                    notes.append(_ATTR["number_source"][lang].format(cid=c.claim_id, ids=more[:1]))
        return c.model_copy(update=update) if update else c

    new = content.model_copy(
        update={
            "what_happened": [fix(c, False) for c in content.what_happened],
            "why_it_matters": [fix(c, True) for c in content.why_it_matters],
            "risks": [fix(c, True) for c in content.risks],
            "opportunities": [fix(c, True) for c in content.opportunities],
        }
    )
    return new, notes


def fallback_blocks(content: InsightContent, ctx: InsightContext) -> tuple[InsightContent, list[str]]:
    """Required blocks that are still empty after pruning get a deterministic statement instead of failing the run:
    “what happened” — the first key fact (or the title) of the news; “why it matters” — the strongest link between
    the event and the subject computed by `impact.event_links`, with a caveat."""
    from ..schemas import SubjectClaim
    from .impact import event_links

    lang, notes, update = ctx.lang, [], {}
    article = ctx.event.articles[0]
    if not content.what_happened:
        text = ctx.event.key_facts[0].text if ctx.event.key_facts else ctx.event.title
        update["what_happened"] = [
            Claim(
                claim_id="c_fact",
                text=text,
                origin=Origin.FACT,
                source_ids=[f"src_{article.article_id}"],
                confidence=0.9,
            )
        ]
        notes.append(_ATTR["fallback_what"][lang])
    if not content.why_it_matters:
        links = sorted(event_links(ctx.profile, ctx.event, lang, ctx.impact_policy), key=lambda x: -x.level)
        feats = ctx.features()
        link = next((x for x in links if x.feature and known_feature(x.feature, feats)), None)
        if link:
            text = link.text[:1].upper() + link.text[1:] + "."
            update["why_it_matters"] = [
                SubjectClaim(
                    claim_id="c_link",
                    text=text,
                    origin=Origin.AI,
                    source_ids=["src_profile", f"src_{article.article_id}"],
                    features=[link.feature],
                    confidence=0.5,
                    caveat=_ATTR["fallback_caveat"][lang],
                )
            ]
            notes.append(_ATTR["fallback_why"][lang])
    return (content.model_copy(update=update) if update else content), notes
