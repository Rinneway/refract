"""Weighing the second model's findings against what code can see.

A small reviewer raises false alarms: an error on a statement that is nearly verbatim from its source, a structural
error the rule check did not find, a “contradiction” its own message calls a wording issue. Code lifts those to
warnings; numbers absent from the context are never lifted.
"""

from __future__ import annotations

import re

from ...schemas import InsightContent, IssueCode, Origin, SourceType, VerificationIssue
from ...textutil import invented_numbers, lexical_support, normalize
from ..context import InsightContext
from .messages import finding
from .rules import HEADLINE

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
                        update={"severity": "warning", "message": i.message + finding("lifted", ctx.lang)}
                    )
                )
                notes.append(finding("confirmed", ctx.lang, cid=c.claim_id, code=i.code.value))
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
                i.model_copy(
                    update={"severity": "warning", "message": i.message + finding("overruled", lang)}
                )
            )
            notes.append(finding("structural", lang, cid=i.claim_id or "—", code=i.code.value))
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
    r"|no (?:\w+ )?(?:source|citation|reference)|does not cite|missing (?:source|citation)",
    re.IGNORECASE,
)


_RECONCILED = {IssueCode.CONTRADICTION, IssueCode.HOUSE_VIEW_WITHOUT_RESEARCH}


def names_institution(text: str, ctx: InsightContext) -> bool:
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
                if c.origin is not Origin.HOUSE_VIEW and not names_institution(c.text, ctx):
                    why = "why_unattributed"
                elif _NO_SOURCE.search(text) and set(c.source_ids) & research:
                    why = "why_cited"
        if why is None:
            out.append(i)
            continue
        out.append(
            i.model_copy(update={"severity": "warning", "message": i.message + finding("softened", lang)})
        )
        notes.append(finding("softened_note", lang, cid=cid, code=i.code.value, why=finding(why, lang)))
        if kind and kinds.get(cid) != "assumption":
            kinds[cid] = kind
    return out, notes, kinds


def for_revision(issues: list[VerificationIssue], kinds: dict[str, str]) -> list[VerificationIssue]:
    """What the revision sees: every error, plus the reconciled remarks — when a round runs anyway, the model
    softens the wording itself instead of code adding a caveat."""
    return [
        i for i in issues if i.severity == "error" or (i.code in _RECONCILED and (i.claim_id or "") in kinds)
    ]
