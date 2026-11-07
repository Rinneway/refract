"""Arithmetic on a proportional rule from the sources.

Research often states a rule of thumb: “every 10% rise in freight cuts the margin by about 1.5 percentage points”.
A small model applies it to the news and gets the arithmetic wrong — in a local run, “a 15% jump cuts the margin by
about 1.5 pp” (it is about 2.25). The number check passes such a sentence: 15% is in the news and 1.5 in the
research. This check finds the rules in the context and recomputes every statement that applies one.

Only the plain case is judged, so that a correct sentence is never flagged: a sentence that shares a word with the
rule (freight, margin) and has one change in percent other than the rule's own and one result in percentage points.
Anything else is left to the reviewer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...i18n import norm_lang
from ...schemas import InsightContent, IssueCode, VerificationIssue
from .messages import finding

_NUM = r"(\d+(?:[.,]\d+)?)"
_PP = r"(?:p\.?p\.?|percentage points?|п\.\s?п\.|процентн\w*\s+пункт\w*)"
_PCT = r"\s?(?:%|percent\b|процент\w*)"
# “every 10% rise … by about 1.5 percentage points”, “каждые 10% роста … на 1,5 п.п.”
_RULE = re.compile(
    rf"(?:every|each|каждые|каждый|каждое|на каждые)\s+(?:additional\s+|дополнительные\s+)?{_NUM}{_PCT}"
    rf"[^.;:!?\n]{{0,90}}?{_NUM}\s*{_PP}",
    re.IGNORECASE,
)
_PERCENT = re.compile(rf"(?<![\w.,]){_NUM}{_PCT}", re.IGNORECASE)
_POINTS = re.compile(rf"(?<![\w.,]){_NUM}\s*{_PP}", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])")
TOLERANCE = 0.15  # “about” in the rule and rounding in the statement


_WORD = re.compile(r"[^\W\d_]{5,}")
_COMMON = {
    "every",
    "about",
    "percent",
    "percentage",
    "points",
    "каждые",
    "каждый",
    "около",
    "примерно",
    "процентных",
    "пункта",
    "пункты",
}


def _stems(text: str) -> frozenset[str]:
    """Four-letter stems of the longer words: “маржу” and “маржа”, “freight” and “freights” meet."""
    return frozenset(w.lower()[:4] for w in _WORD.findall(text) if w.lower() not in _COMMON)


# words for the move itself: “a 6% rise in prices” is not about freight
_MOVES = {"рост", "сниж", "повы", "увел", "паде", "умен", "jump", "incr", "fall", "decl", "grow", "redu"}


@dataclass(frozen=True)
class Rule:
    change: float  # percent
    effect: float  # percentage points
    # what the rule is about: a sentence applying it shares a word with it
    stems: frozenset[str] = frozenset()
    # what changes: the words right after the change (“rise in freight”); the news must report that change
    subject: frozenset[str] = frozenset()


def _f(s: str) -> float:
    return float(s.replace(",", "."))


def _fmt(x: float, lang: str) -> str:
    text = f"{x:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") if norm_lang(lang) == "ru" else text


def rules_in(text: str) -> list[Rule]:
    out = []
    for m in _RULE.finditer(text or ""):
        change, effect = _f(m.group(1)), _f(m.group(2))
        if change > 0 and effect > 0:
            after = " ".join(m.group(0)[m.end(1) - m.start() :].split()[:4])
            out.append(Rule(change, effect, _stems(m.group(0)), _stems(after) - _MOVES))
    return list(dict.fromkeys(out))


def misapplied(text: str, rules: list[Rule]) -> tuple[Rule, float, float, float] | None:
    """(rule, change, stated effect, expected effect) of the first sentence that applies a rule wrongly."""
    for sentence in _SENTENCE.split(text or ""):
        points = [_f(m.group(1)) for m in _POINTS.finditer(sentence)]
        if not points:
            continue
        for rule in rules:
            if rule.stems and not rule.stems & _stems(sentence):
                continue  # another rule's topic
            percents = [(m.start(), _f(m.group(1))) for m in _PERCENT.finditer(sentence)]
            changes = {v for _, v in percents} - {rule.change}
            stated = set(points) - {rule.effect} or set(points)
            if stated == {rule.effect}:
                # the rule's own effect: wrong only if said of the news' change, not when the rule is quoted beside
                # it (“…erodes your margin; every 10% rise cuts it by about 1.5 pp”), i.e. before the rule's change
                own = [pos for pos, v in percents if v == rule.change]
                first_point = next(_POINTS.finditer(sentence)).start()
                if own and own[0] < first_point:
                    continue
            if len(changes) != 1 or len(stated) != 1:
                continue  # not the plain case
            change, value = changes.pop(), stated.pop()
            expected = rule.effect * change / rule.change
            if abs(value - expected) > TOLERANCE * expected:
                return rule, change, value, expected
    return None


def event_changes(event_text: str, rule: Rule) -> set[float]:
    """Changes in percent the news reports on the rule's topic (“a 15% jump in freight rates”)."""
    out: set[float] = set()
    for sentence in _SENTENCE.split(event_text or ""):
        if rule.subject and rule.subject & _stems(sentence):
            out |= {_f(m.group(1)) for m in _PERCENT.finditer(sentence)}
    return out - {rule.change}


def unscaled(text: str, rules: list[Rule], event_text: str) -> tuple[Rule, float, float, float] | None:
    """A sentence on the rule's topic that gives the rule's own effect with no change at all (“higher freight cuts
    your margin by ~1.5 pp”) while the news reports one other change (15%): the rule was applied without scaling.
    Found in the second local run, after `misapplied` caught the version of the sentence that named the 15%."""
    for sentence in _SENTENCE.split(text or ""):
        points = {_f(m.group(1)) for m in _POINTS.finditer(sentence)}
        if not points or _PERCENT.search(sentence):
            continue
        for rule in rules:
            if not (rule.stems & _stems(sentence)) or points != {rule.effect}:
                continue
            changes = event_changes(event_text, rule)
            if len(changes) != 1:
                continue
            change = changes.pop()
            expected = rule.effect * change / rule.change
            if abs(rule.effect - expected) > TOLERANCE * expected:
                return rule, change, rule.effect, expected
    return None


def arithmetic_issues(
    content: InsightContent, context: str, lang: str, event_text: str = ""
) -> list[VerificationIssue]:
    rules = rules_in(context)
    if not rules:
        return []
    lang = norm_lang(lang)
    out = []
    quoted = {c.claim_id for c in content.house_view}  # the institution's rule quoted as it is
    for c in content.all_claims():
        found = misapplied(c.text, rules) or (
            None if c.claim_id in quoted else unscaled(c.text, rules, event_text)
        )
        if found is None:
            continue
        rule, change, value, expected = found
        out.append(
            VerificationIssue(
                code=IssueCode.ARITHMETIC,
                severity="error",
                claim_id=c.claim_id,
                message=finding(
                    "arithmetic",
                    lang,
                    cid=c.claim_id,
                    x=_fmt(rule.change, lang),
                    y=_fmt(rule.effect, lang),
                    p=_fmt(change, lang),
                    e=_fmt(expected, lang),
                    v=_fmt(value, lang),
                ),
                fix_hint=finding("arithmetic_hint", lang),
            )
        )
    return out
