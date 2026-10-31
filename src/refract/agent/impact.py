"""How strongly an event is linked to the subject, and the impact ceiling — computed by code, not by the model.

LLMs tend to give one piece of news the same impact level for every reader. The ceiling fixes that: each link
between the event and the subject gets a strength, and the strongest link bounds the level the model may assign.

- strong (allows high): the event names the subject's main activity (two-digit NACE/OKVED division), a country it
  trades with, the activity of a counterparty with more than 30% of the payment flow, or a factor the subject is
  sensitive to with strength ≥ 0.8;
- moderate (at most medium): a counterparty with 10–30% of the flow, sensitivity 0.5–0.8, the subject's region,
  strong sensitivity to a secondary factor of the event;
- weak (low): only the section letter matches (section G covers both a fuel wholesaler and a clothing shop),
  weak sensitivity, a counterparty matched only by section.

`high` additionally requires a strong event (`NewsEvent.importance` ≥ 0.6). All thresholds live in
`ImpactPolicy`, so they can be varied in sensitivity experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..i18n import norm_lang
from ..schemas import ImpactLevel, NewsEvent, Profile
from ..schemas.vocab import country_code, division, section_of

STRONG, MODERATE, WEAK = 3, 2, 1
CAP = {STRONG: ImpactLevel.HIGH, MODERATE: ImpactLevel.MEDIUM, WEAK: ImpactLevel.LOW}
ORDER = {ImpactLevel.LOW: 0, ImpactLevel.MEDIUM: 1, ImpactLevel.HIGH: 2}

_LEVEL = {"ru": {3: "сильная", 2: "умеренная", 1: "слабая"}, "en": {3: "strong", 2: "moderate", 1: "weak"}}
_DIRECTION = {
    "ru": {"negative": "негативно", "positive": "позитивно", "mixed": "неоднозначно", "unclear": "неясно"},
    "en": {"negative": "negative", "positive": "positive", "mixed": "mixed", "unclear": "unclear"},
}
_TEXT = {
    "exposure": {
        "ru": "фактор {factor}{note}: чувствительность {direction}, сила {strength} — {reason}",
        "en": "factor {factor}{note}: sensitivity {direction}, strength {strength} — {reason}",
    },
    "secondary": {"ru": " (второстепенный фактор события)", "en": " (secondary factor of the event)"},
    "own_division": {
        "ru": "событие касается основного вида деятельности (NACE/ОКВЭД {code})",
        "en": "the event concerns the subject's main activity (NACE division {code})",
    },
    "own_section": {
        "ru": "совпал только раздел {section}; основной вид деятельности ({code}) событие прямо не называет",
        "en": "only section {section} matches; the event does not name the main activity ({code}) directly",
    },
    "cp_division": {
        "ru": "событие касается отрасли {who} {cid} (код {code}, {share})",
        "en": "the event concerns the industry of {who} {cid} (division {code}, {share})",
    },
    "cp_section": {
        "ru": "совпал только раздел {section} у {who} {cid} ({share})",
        "en": "only section {section} matches for {who} {cid} ({share})",
    },
    "country": {
        "ru": "событие касается стран внешней торговли: {countries}",
        "en": "the event concerns countries the subject trades with: {countries}",
    },
    "region": {
        "ru": "событие касается региона субъекта ({region})",
        "en": "the event concerns the subject's region ({region})",
    },
    "customer": {"ru": "покупателя", "en": "customer"},
    "supplier": {"ru": "поставщика", "en": "supplier"},
    "in": {"ru": "{share:.0%} входящих платежей", "en": "{share:.0%} of incoming payments"},
    "out": {"ru": "{share:.0%} исходящих платежей", "en": "{share:.0%} of outgoing payments"},
    "no_exposures": {
        "ru": "чувствительности не рассчитаны — без данных не выше medium",
        "en": "sensitivities are not computed — without data at most medium",
    },
    "none": {"ru": "прямых связей нет — общий макрофон", "en": "no direct links — general macro background"},
    "link": {"ru": "{level} связь: {text}", "en": "{level} link: {text}"},
    "weak_event": {
        "ru": "{reason}, но само событие слабое (сила {importance})",
        "en": "{reason}, but the event itself is weak (strength {importance})",
    },
}


def _num(x: float, lang: str) -> str:
    text = f"{x:.1f}"
    return text.replace(".", ",") if norm_lang(lang) == "ru" else text


def _t(key: str, lang: str, **kw: object) -> str:
    return _TEXT[key][norm_lang(lang)].format(**kw)


@dataclass(frozen=True)
class ImpactPolicy:
    """Thresholds of the impact ceiling."""

    high_importance: float = 0.6  # high only for events that are strong by themselves
    low_importance: float = 0.3  # a very weak event never rises above low
    share_min: float = 0.1  # counterparties below 10% of the flow are not a link
    share_high: float = 0.3  # above 30% — a strong link
    strong_exposure: float = 0.8
    moderate_exposure: float = 0.5


DEFAULT_IMPACT = ImpactPolicy()


@dataclass(frozen=True)
class Link:
    """One link between the event and the subject: what matched and how strong the link is."""

    kind: str  # exposure | own_division | own_section | counterparty | country | region
    text: str
    level: int
    feature: str = ""  # the subject's feature key the link rests on

    def render(self, lang: str = "ru") -> str:
        noun = "связь" if norm_lang(lang) == "ru" else "link"
        return f"{_LEVEL[norm_lang(lang)][self.level]} {noun} — {self.text}"


@dataclass(frozen=True)
class ImpactCap:
    level: ImpactLevel
    reason: str


def event_divisions(e: NewsEvent) -> set[str]:
    """Divisions the event concerns: its industry codes and sector:<code> factors."""
    out = {d for d in map(division, e.industry_codes) if d}
    out |= {d for f in e.macro_factors if f.startswith("sector:") and (d := division(f[7:]))}
    return out


def canonical_factor(factor: str) -> str:
    """country:<name> → country:<ISO code>; fx codes upper-cased; other factors unchanged."""
    if factor.startswith("country:"):
        return f"country:{country_code(factor[8:])}"
    if factor.startswith("fx:"):
        return f"fx:{factor[3:].strip().upper()}"
    return factor


def event_countries(e: NewsEvent) -> set[str]:
    names = list(e.countries) + [f[8:] for f in e.macro_factors if f.startswith("country:")]
    return {c for c in map(country_code, names) if c}


def _exposure_level(strength: float, policy: ImpactPolicy) -> int:
    if strength >= policy.strong_exposure:
        return STRONG
    if strength >= policy.moderate_exposure:
        return MODERATE
    return WEAK


def _factor_hit(factor: str, factors: set[str], divisions: set[str]) -> bool:
    if canonical_factor(factor) in factors:
        return True
    return factor.startswith("sector:") and division(factor[7:]) in divisions


def event_links(
    p: Profile, e: NewsEvent, lang: str = "ru", policy: ImpactPolicy = DEFAULT_IMPACT
) -> list[Link]:
    """How the event touches the subject: matching factors, activities, countries, counterparties, region."""
    links: list[Link] = []
    factors = {canonical_factor(f) for f in e.macro_factors}
    divisions = event_divisions(e)
    sections = set(e.industry_sections) | {s for s in map(section_of, divisions) if s}

    # The first factor of an event is the main one (the classifier puts it first). A policy-rate decision may
    # mention the labour market, but for a subject with a payroll that is background, not a strong link.
    primary = canonical_factor(e.macro_factors[0]) if e.macro_factors else None
    for x in sorted(p.exposures, key=lambda x: -x.strength):
        if not _factor_hit(x.factor, factors, divisions):
            continue
        level, note = _exposure_level(x.strength, policy), ""
        if x.via == "group":
            level = max(WEAK, level - 1)
        if len(e.macro_factors) > 1 and canonical_factor(x.factor) != primary:
            level, note = max(WEAK, level - 1), _t("secondary", lang)
        text = _t(
            "exposure",
            lang,
            factor=x.factor,
            note=note,
            direction=_DIRECTION[norm_lang(lang)][x.direction.value],
            strength=_num(x.strength, lang),
            reason=x.reason,
        )
        links.append(Link("exposure", text, level, x.factor))

    if p.industry:
        own_div, own_sec = division(p.industry.code), p.industry.section or section_of(p.industry.code)
        if own_div and own_div in divisions:
            links.append(Link("own_division", _t("own_division", lang, code=own_div), STRONG, "industry"))
        elif own_sec and own_sec in sections:
            links.append(
                Link(
                    "own_section",
                    _t("own_section", lang, section=own_sec, code=p.industry.code),
                    WEAK,
                    "industry",
                )
            )

    for cp in p.counterparties:
        if not cp.industry or cp.share_of_flow < policy.share_min:
            continue
        who = _t("customer" if cp.role == "customer" else "supplier", lang)
        share = _t("in" if cp.role == "customer" else "out", lang, share=cp.share_of_flow)
        cp_div = division(cp.industry.code)
        cp_sec = cp.industry.section or section_of(cp.industry.code)
        strong = cp.share_of_flow > policy.share_high
        if cp_div and cp_div in divisions:
            text = _t("cp_division", lang, who=who, cid=cp.counterparty_id, code=cp_div, share=share)
            links.append(
                Link(
                    "counterparty",
                    text,
                    STRONG if strong else MODERATE,
                    f"counterparties.{cp.role}.{cp.counterparty_id}",
                )
            )
        elif cp_sec and cp_sec in sections:
            text = _t("cp_section", lang, section=cp_sec, who=who, cid=cp.counterparty_id, share=share)
            links.append(
                Link(
                    "counterparty",
                    text,
                    MODERATE if strong else WEAK,
                    f"counterparties.{cp.role}.{cp.counterparty_id}",
                )
            )

    countries = event_countries(e)
    trade = [c for c in p.foreign_trade.countries if country_code(c) in countries]
    if trade:
        links.append(
            Link("country", _t("country", lang, countries=", ".join(trade)), STRONG, "foreign_trade")
        )

    regions = [r.lower() for r in e.regions] + [f[7:].lower() for f in factors if f.startswith("region:")]
    if p.region and any(p.region.lower() in r or r in p.region.lower() for r in regions if r):
        links.append(Link("region", _t("region", lang, region=p.region), MODERATE, "region"))
    return links


def impact_cap(
    p: Profile,
    e: NewsEvent,
    links: list[Link] | None = None,
    lang: str = "ru",
    policy: ImpactPolicy = DEFAULT_IMPACT,
) -> ImpactCap:
    """The highest impact level allowed for this subject × event pair."""
    links = event_links(p, e, lang, policy) if links is None else links
    best = max((x.level for x in links), default=0)
    if not p.exposures and best < STRONG:
        return ImpactCap(ImpactLevel.MEDIUM, _t("no_exposures", lang))
    if best == 0:
        return ImpactCap(ImpactLevel.LOW, _t("none", lang))
    level = CAP[best]
    top = next(x for x in links if x.level == best)
    reason = _t("link", lang, level=_LEVEL[norm_lang(lang)][best], text=top.text[:140])
    if level == ImpactLevel.HIGH and e.importance < policy.high_importance:
        level, reason = (
            ImpactLevel.MEDIUM,
            _t("weak_event", lang, reason=reason, importance=_num(e.importance, lang)),
        )
    elif level == ImpactLevel.MEDIUM and e.importance < policy.low_importance:
        level, reason = (
            ImpactLevel.LOW,
            _t("weak_event", lang, reason=reason, importance=_num(e.importance, lang)),
        )
    return ImpactCap(level, reason)


def above(level: ImpactLevel | str, cap: ImpactLevel) -> bool:
    return ORDER[ImpactLevel(level)] > ORDER[cap]
