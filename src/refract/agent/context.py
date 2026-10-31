"""The context the model sees: subject features, the event, the house view, sources.

Subject features get keys (`products.loans`, `counterparties.customer.<id>`…). The model cites them in
`Claim.features`, and the rule checks verify the keys are real — the model cannot invent a feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Institution
from ..i18n import norm_lang, t
from ..schemas import KnowledgeChunk, NewsEvent, Profile, SourceRef, SourceType, Thesis
from ..textutil import looks_like_injection, untrusted
from .actions import Candidate, candidate_actions
from .impact import DEFAULT_IMPACT, ImpactCap, ImpactPolicy, event_links, impact_cap
from .policy import DEFAULT_POLICY, Policy

PRODUCT_LABELS = {
    "current_account": {"ru": "расчётный счёт", "en": "current account"},
    "loans": {"ru": "кредиты", "en": "loans"},
    "mortgage": {"ru": "ипотека", "en": "mortgage"},
    "leasing": {"ru": "лизинг", "en": "leasing"},
    "deposits": {"ru": "депозиты", "en": "deposits"},
    "payroll": {"ru": "зарплатный проект", "en": "payroll services"},
    "factoring": {"ru": "факторинг", "en": "factoring"},
    "guarantees": {"ru": "банковские гарантии", "en": "bank guarantees"},
    "acquiring": {"ru": "эквайринг", "en": "card acquiring"},
    "fx_account": {"ru": "валютный счёт", "en": "foreign-currency account"},
    "investments": {"ru": "инвестиции", "en": "investments"},
}
KIND_LABELS = {
    "company": {"ru": "компания", "en": "company"},
    "sole_proprietor": {"ru": "индивидуальный предприниматель", "en": "sole proprietor"},
    "self_employed": {"ru": "самозанятый", "en": "self-employed person"},
    "individual": {"ru": "частное лицо", "en": "private individual"},
}
SEGMENT_LABELS = {
    "large": {"ru": "крупный бизнес", "en": "large business"},
    "medium": {"ru": "средний бизнес", "en": "medium-sized business"},
    "small": {"ru": "малый бизнес", "en": "small business"},
    "micro": {"ru": "микробизнес", "en": "micro business"},
    "household": {"ru": "домохозяйство", "en": "household"},
    "unknown": {"ru": "масштаб неизвестен", "en": "scale unknown"},
}
DIRECTION_LABELS = {
    "ru": {"negative": "негативно", "positive": "позитивно", "mixed": "неоднозначно", "unclear": "неясно"},
    "en": {"negative": "negative", "positive": "positive", "mixed": "mixed", "unclear": "unclear"},
}
VIA_LABELS = {
    "ru": {"own": "напрямую", "counterparty": "через контрагентов", "group": "через группу"},
    "en": {"own": "directly", "counterparty": "via counterparties", "group": "via the group"},
}
TRADE_DIRECTION = {
    "ru": {
        "import": "импорт",
        "export": "экспорт",
        "both": "импорт и экспорт",
        "unknown": "направление неизвестно",
    },
    "en": {
        "import": "imports",
        "export": "exports",
        "both": "imports and exports",
        "unknown": "direction unknown",
    },
}


def money(value: float | None, currency: str, lang: str) -> str:
    """12_600_000_000 RUB → «12,6 млрд руб.» / “RUB 12.6bn”; 2_100_000 → «2,1 млн руб.» / “RUB 2.1m”."""
    lang = norm_lang(lang)
    if value is None:
        return "нет данных" if lang == "ru" else "no data"

    def num(x: float) -> str:
        text = f"{x:.1f}" if x < 100 else f"{x:.0f}"
        text = text.removesuffix(".0")
        return text.replace(".", ",") if lang == "ru" else text

    unit = ("руб." if currency == "RUB" else currency) if lang == "ru" else currency
    if value >= 1e9:
        return f"{num(value / 1e9)} млрд {unit}" if lang == "ru" else f"{unit} {num(value / 1e9)}bn"
    if value >= 1e6:
        return f"{num(value / 1e6)} млн {unit}" if lang == "ru" else f"{unit} {num(value / 1e6)}m"
    if lang == "ru":
        return f"{value:,.0f}".replace(",", " ") + f" {unit}"
    return f"{unit} {value:,.0f}"


def profile_features(
    p: Profile, lang: str = "ru", institution: Institution | None = None
) -> list[tuple[str, str]]:
    """Subject features as (key, description) pairs."""
    lang = norm_lang(lang)
    inst = institution or Institution()
    ru = lang == "ru"
    f: list[tuple[str, str]] = [("kind", KIND_LABELS[p.kind.value][lang])]
    if p.industry:
        scheme = (
            "ОКВЭД"
            if p.industry.scheme == "okved" and ru
            else ("NACE" if p.industry.scheme == "nace" else "")
        )
        f.append(("industry", f"{scheme} {p.industry.code} — {p.industry.name}".strip()))
    f.append(("scale.segment", SEGMENT_LABELS[p.scale.segment][lang]))
    if p.country or p.region:
        place = ", ".join(x for x in (p.region, p.city) if x) or ""
        country = f"{p.country}" if p.country else ""
        f.append(("region", ", ".join(x for x in (place, country) if x)))
    cur = p.scale.currency
    if p.scale.revenue is not None:
        label = "выручка за год" if ru else "annual revenue"
        if p.kind.value == "individual":
            label = "доход за год" if ru else "annual income"
        f.append(("scale.revenue", f"{label}: {money(p.scale.revenue, cur, lang)}"))
    if p.scale.group_revenue is not None:
        f.append(
            (
                "scale.group_revenue",
                f"{'выручка группы' if ru else 'group revenue'}: {money(p.scale.group_revenue, cur, lang)}",
            )
        )
    if p.scale.employees:
        f.append(("scale.employees", f"{'численность' if ru else 'headcount'}: {p.scale.employees}"))
    if p.industry_position and p.industry_position.revenue_percentile is not None:
        ip = p.industry_position
        f.append(
            (
                "industry_position",
                f"по выручке выше {ip.revenue_percentile:.0f}% субъектов с тем же кодом в выборке ({ip.peers_in_sample})"
                if ru
                else f"revenue above {ip.revenue_percentile:.0f}% of peers with the same code in the sample ({ip.peers_in_sample})",
            )
        )
    if p.tax_regime:
        f.append(("tax_regime", f"{'налоговый режим' if ru else 'tax regime'}: {p.tax_regime}"))
    held = [name for name in PRODUCT_LABELS if getattr(p.products, name)]
    for name in held:
        label = PRODUCT_LABELS[name][lang]
        f.append(
            (
                f"products.{name}",
                f"есть {label} (продукт {inst.genitive(lang)})"
                if ru
                else f"has {label} with {inst.display(lang)}",
            )
        )
    absent = [PRODUCT_LABELS[n][lang] for n in PRODUCT_LABELS if n not in held]
    if absent:
        f.append(
            (
                "products.absent",
                f"продуктов {inst.genitive(lang)} нет: {', '.join(absent)}; о продуктах в других организациях данных нет"
                if ru
                else f"not held with {inst.display(lang)}: {', '.join(absent)}; products elsewhere are unknown",
            )
        )
    ft = p.foreign_trade
    if ft.active:
        countries = ", ".join(ft.countries) or ("страна не указана" if ru else "country unknown")
        f.append(
            (
                "foreign_trade",
                f"внешнеторговые операции ({TRADE_DIRECTION[lang][ft.direction]}): {countries}"
                if ru
                else f"foreign trade ({TRADE_DIRECTION[lang][ft.direction]}): {countries}",
            )
        )
    elif p.kind.value in ("company", "sole_proprietor"):
        f.append(("foreign_trade", "внешнеторговых операций нет" if ru else "no foreign trade"))
    if p.group_members:
        n = len(p.group_members) + 1
        f.append(
            (
                "group",
                f"входит в группу компаний ({n} юрлиц)"
                if ru
                else f"part of a group of companies ({n} entities)",
            )
        )
    b2c = bool(
        p.industry and p.industry.code.startswith(("47", "56"))
    )  # retail and food service: card payments
    for cp in p.counterparties:
        if ru:
            who = "покупатель" if cp.role == "customer" else "поставщик"
            if b2c and cp.role == "customer" and cp.industry is None and cp.share_of_flow >= 0.3:
                who = "канал поступлений (вероятно, эквайринг или маркетплейс, а не отдельный покупатель)"
            flow = "входящих" if cp.role == "customer" else "исходящих"
            activity = f"{cp.industry.code} {cp.industry.name}" if cp.industry else "отрасль неизвестна"
            text = f"{who}: {activity}{', ' + cp.region if cp.region else ''}; {cp.share_of_flow:.0%} {flow} платежей"
        else:
            who = "customer" if cp.role == "customer" else "supplier"
            if b2c and cp.role == "customer" and cp.industry is None and cp.share_of_flow >= 0.3:
                who = "payment channel (probably card acquiring or a marketplace, not a single customer)"
            flow = "incoming" if cp.role == "customer" else "outgoing"
            activity = f"{cp.industry.code} {cp.industry.name}" if cp.industry else "industry unknown"
            text = f"{who}: {activity}{', ' + cp.region if cp.region else ''}; {cp.share_of_flow:.0%} of {flow} payments"
        f.append((f"counterparties.{cp.role}.{cp.counterparty_id}", text))
    for a in p.affiliations:
        if a.kind == "shared_person":
            f.append(
                (
                    f"affiliations.{a.profile_id}",
                    f"возможная связь (гипотеза, уверенность {a.confidence:.0%}): {a.evidence}"
                    if ru
                    else f"possible link (hypothesis, confidence {a.confidence:.0%}): {a.evidence}",
                )
            )
    for x in p.exposures:
        how, via = DIRECTION_LABELS[lang][x.direction.value], VIA_LABELS[lang][x.via]
        strength = f"{x.strength:.1f}".replace(".", ",") if ru else f"{x.strength:.1f}"
        f.append(
            (
                x.factor,
                f"чувствительность: {how}, {via}, сила {strength} — {x.reason}"
                if ru
                else f"sensitivity: {how}, {via}, strength {strength} — {x.reason}",
            )
        )
    for key, value in p.user_provided.items():
        f.append(
            (
                f"user.{key}",
                f"{'со слов пользователя' if ru else 'as stated by the user'}: {untrusted(str(value))}",
            )
        )
    return f


def known_feature(key: str, features: list[tuple[str, str]]) -> bool:
    """A key is real if it equals a known key or refines one (products.loans.rate)."""
    return any(key == k or key.startswith(k + ".") for k, _ in features)


def profile_source(lang: str) -> SourceRef:
    return SourceRef(source_id="src_profile", type=SourceType.PROFILE, title=t("ctx.source.profile", lang))


def payments_source(lang: str) -> SourceRef:
    return SourceRef(source_id="src_payments", type=SourceType.PAYMENTS, title=t("ctx.source.payments", lang))


def thesis_source(th: Thesis, institution: Institution, lang: str) -> SourceRef:
    return SourceRef(
        source_id=f"src_{th.thesis_id}",
        type=SourceType.RESEARCH,
        title=th.doc_title,
        publisher=institution.display(lang),
        page=th.page,
        excerpt=th.quote[:1000],
    )


def chunk_source(c: KnowledgeChunk, institution: Institution, lang: str) -> SourceRef:
    return SourceRef(
        source_id=f"src_{c.chunk_id}",
        type=SourceType.RESEARCH,
        title=c.doc_title,
        publisher=institution.display(lang),
        page=c.page,
        excerpt=c.text[:1000],
    )


def event_sources(e: NewsEvent) -> list[SourceRef]:
    return [
        SourceRef(
            source_id=f"src_{a.article_id}",
            type=SourceType.NEWS,
            title=a.title,
            url=a.url,
            publisher=a.publisher,
            published_at=a.published_at,
        )
        for a in e.articles
    ]


@dataclass
class InsightContext:
    """Everything the model sees while writing the insight, plus the source registry for checking citations."""

    profile: Profile
    event: NewsEvent
    theses: list[Thesis] = field(default_factory=list)
    role: str = "cfo"
    research: list[KnowledgeChunk] = field(default_factory=list)
    related: list[NewsEvent] = field(default_factory=list)
    lang: str = "ru"
    institution: Institution = field(default_factory=Institution)
    policy: Policy = field(default_factory=lambda: DEFAULT_POLICY)
    impact_policy: ImpactPolicy = field(default_factory=lambda: DEFAULT_IMPACT)
    use_cap: bool = True
    use_actions: bool = True
    offer_products: bool = True

    def sources(self) -> list[SourceRef]:
        src = event_sources(self.event)
        src += [thesis_source(th, self.institution, self.lang) for th in self.theses]
        src += [chunk_source(c, self.institution, self.lang) for c in self.research]
        for e in self.related:
            src += event_sources(e)
        src.append(profile_source(self.lang))
        if self.profile.counterparties:
            src.append(payments_source(self.lang))
        unique: dict[str, SourceRef] = {}
        for s in src:
            unique.setdefault(s.source_id, s)
        return list(unique.values())

    def features(self) -> list[tuple[str, str]]:
        return profile_features(self.profile, self.lang, self.institution)

    def cap(self) -> ImpactCap:
        return impact_cap(self.profile, self.event, lang=self.lang, policy=self.impact_policy)

    def candidates(self) -> list[Candidate]:
        if not self.use_actions:
            return []
        return candidate_actions(self.profile, self.event, self.policy, self.lang)

    def inst(self, genitive: bool = False) -> dict[str, str]:
        return {
            "institution": self.institution.display(self.lang),
            "institution_gen": self.institution.genitive(self.lang),
        }

    def render(self) -> str:
        p, e, lang = self.profile, self.event, self.lang
        lines = [f"## {t('ctx.subject', lang)}", p.summary, "", t("ctx.features", lang)]
        lines += [f"- {k} — {v}" for k, v in self.features()]
        if p.data_quality.notes or p.data_quality.missing:
            lines += ["", t("ctx.limits", lang) + "; ".join(p.data_quality.notes + p.data_quality.missing)]

        lines += [
            "",
            "## " + t("ctx.event", lang, date=f"{e.published_at:%Y-%m-%d}", type=e.event_type.value),
            untrusted(f"{e.title}\n{e.summary}"),
        ]
        links = event_links(p, e, lang, self.impact_policy)
        lines += ["", f"## {t('ctx.links', lang)}"]
        if links:
            lines += [f"- {x.render(lang)}" for x in links]
        elif not p.exposures:
            lines.append(t("ctx.no_exposures", lang))
        else:
            lines.append(t("ctx.no_links", lang))
        if self.use_cap:
            cap = impact_cap(p, e, links, lang, self.impact_policy)
            lines.append(t("ctx.cap", lang, level=cap.level.value, reason=cap.reason))
        else:
            lines.append(t("ctx.no_cap", lang))

        lines += ["", f"## {t('ctx.event_text', lang)}"]
        texts = [e.title, e.summary]
        for a in e.articles:
            lines.append(f"- [src_{a.article_id}] {a.publisher}:")
            lines.append(untrusted(f"{a.title}\n{(a.text or '')[:3000]}".strip(), f"src_{a.article_id}"))
            texts += [a.title, a.text or ""]
        for kf in e.key_facts:
            lines.append(f"- {t('ctx.fact', lang)} [src_{kf.article_id}]: {untrusted(kf.text)}")
            texts.append(kf.text)
        if any(looks_like_injection(x) for x in texts):
            lines.append(t("ctx.injection", lang))

        lines += ["", "## " + t("ctx.house_view", lang, **self.inst())]
        if self.theses:
            for th in self.theses:
                when = f"{th.doc_date:%Y-%m-%d}" if th.doc_date else t("ctx.date_unknown", lang)
                stale = t("ctx.stale", lang) if th.is_stale else ""
                lines.append(
                    f"- [src_{th.thesis_id}] «{th.doc_title}», {when}, {t('ctx.page', lang)} {th.page or '?'}{stale}: {th.statement}"
                )
                lines.append(f"  {t('ctx.quote', lang)}: «{th.quote}»")
        else:
            lines.append(t("ctx.no_theses", lang))
        for c in self.research:
            when = f"{c.doc_date:%Y-%m-%d}" if c.doc_date else t("ctx.date_unknown", lang)
            lines.append(
                f"- {t('ctx.chunk', lang)} [src_{c.chunk_id}] «{c.doc_title}», {when}, {t('ctx.page', lang)} {c.page or '?'}:"
            )
            lines.append(f"  «{c.text[:900]}»")

        if self.related:
            lines += ["", f"## {t('ctx.related', lang)}"]
            for r in self.related:
                for a in r.articles:
                    lines.append(f"- [src_{a.article_id}] {a.publisher}, {a.published_at:%Y-%m-%d}:")
                    lines.append(untrusted(f"{a.title}\n{r.summary[:300]}", f"src_{a.article_id}"))

        lines += ["", f"## {t('ctx.actions', lang)}"]
        if not self.use_actions:
            lines.append(t("ctx.no_policy", lang))
        elif candidates := self.candidates():
            lines += [f"- {c.render(lang, self.offer_products)}" for c in candidates]
        else:
            lines.append(t("ctx.no_actions", lang))

        lines += ["", f"## {t('ctx.sources', lang)}"]
        lines += [f"- {s.source_id} — {s.type.value} — {s.title}" for s in self.sources()]
        return "\n".join(lines)
