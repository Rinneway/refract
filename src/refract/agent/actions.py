"""Next actions by the action policy: code decides which rules fire, the model adapts the wording, code checks it.

Principles: a product only by a fired rule and only if the subject does not hold it; at most one `apply_product`
per insight; with high negative impact `contact_manager` comes first; no products for tax events; an empty list
is topped up from the fired rules that need no judgement (no condition only the model can check), and an empty
list without such rules is replaced by the fallback action from the policy.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..i18n import norm_lang, t
from ..schemas import (
    Action,
    ActionKind,
    Direction,
    EventType,
    ImpactLevel,
    InsightContent,
    NewsEvent,
    Profile,
)
from ..schemas.vocab import country_code, division
from .impact import event_countries, event_divisions
from .policy import DEFAULT_POLICY, ActionRule, Policy

MAX_ACTIONS = 3
# Fewer actions than this are topped up from fired rules. A local run showed the cost of too few: checks cut the
# model's unsupported or off-policy actions, often down to one generic step, and both LLM judges scored “actions”
# lowest for the agent while a single call, which nothing checks, kept two or three.
MIN_ACTIONS = 2
# An action to contact the manager must say so; otherwise it is the reader's own step.
_MANAGER = re.compile(
    r"менеджер|банк|перезвон|обратн\w* звон|manager|bank|call ?back|relationship", re.IGNORECASE
)
_ANALYSIS = re.compile(
    r"анализ|оцен|рассчит|пересчит|сравн|пересмотр|провер|review|analy[sz]|assess|estimat|calculat|compar|check"
    r"|forecast|model",
    re.IGNORECASE,
)
_COMMODITIES = {"oil", "gas", "metals", "coal", "agri", "gold", "energy_prices"}
_DIRECTION = {
    "ru": {"negative": "для субъекта событие негативно", "positive": "для субъекта событие позитивно"},
    "en": {
        "negative": "the event is negative for the subject",
        "positive": "the event is positive for the subject",
    },
}
_MOVE = {
    "ru": {"up": "фактор растёт (например, ставку повысили)", "down": "фактор снижается"},
    "en": {"up": "the factor rises (e.g. the rate was raised)", "down": "the factor falls"},
}
_RENDER = {
    "ru": ("продукт", "без продукта", "только если", "Почему"),
    "en": ("product", "no product", "only if", "Why"),
}


@dataclass(frozen=True)
class Candidate:
    """A fired rule plus a condition that code cannot check and the model must judge."""

    rule: ActionRule
    condition: str | None

    def render(self, lang: str = "ru", offer_products: bool = True) -> str:
        product_w, none_w, only_w, why_w = _RENDER[norm_lang(lang)]
        product = self.rule.product(lang) if offer_products else None
        prod = f"; {product_w}: «{product}»" if product else f"; {none_w}"
        cond = f"; {only_w} {self.condition}" if self.condition else ""
        return f"[{self.rule.id}] {self.rule.kind}: «{self.rule.text(lang)}»{prod}{cond}. {why_w}: {self.rule.rationale(lang)}"


def _factor_match(pattern: str, e: NewsEvent) -> bool:
    factors = set(e.macro_factors)
    if not pattern.endswith(":*"):
        return pattern in factors
    prefix = pattern[:-1]
    if any(f.startswith(prefix) for f in factors):
        return True
    if prefix == "commodity:":
        return bool(factors & _COMMODITIES) or bool(e.commodities)
    if prefix == "sector:":
        return bool(event_divisions(e) or e.industry_sections)
    if prefix == "country:":
        return bool(e.countries)
    return False


def _feature(feature: str, p: Profile, e: NewsEvent) -> bool:
    if feature.startswith("products."):
        return bool(getattr(p.products, feature.split(".", 1)[1], False))
    if feature == "foreign_trade.active":
        return p.foreign_trade.active
    if feature == "foreign_trade.countries":
        return bool({country_code(c) for c in p.foreign_trade.countries} & event_countries(e))
    if feature == "tax_regime":
        return bool(p.tax_regime)
    return False


def _big_customer_touched(p: Profile, e: NewsEvent, min_share: float) -> bool:
    """A large customer whose activity the event names directly. A section match is a weak link — no product."""
    divisions = event_divisions(e)
    return any(
        cp.role == "customer"
        and cp.share_of_flow >= min_share
        and cp.industry
        and division(cp.industry.code) in divisions
        for cp in p.counterparties
    )


def candidate_actions(
    p: Profile, e: NewsEvent, policy: Policy = DEFAULT_POLICY, lang: str = "ru"
) -> list[Candidate]:
    """Rules whose conditions hold for this subject and these event tags."""
    lang = norm_lang(lang)
    out: list[Candidate] = []
    for rule in policy.rules():
        w = rule.when
        if "subject_kinds" in w and p.kind.value not in w["subject_kinds"]:
            continue
        if "event_types" in w and e.event_type.value not in w["event_types"]:
            continue
        if "factors" in w and not any(_factor_match(f, e) for f in w["factors"]):
            continue
        if not all(_feature(f, p, e) for f in w.get("has", [])):
            continue
        if any(_feature(f, p, e) for f in w.get("has_not", [])):
            continue
        if w.get("via") == "counterparty" and not _big_customer_touched(p, e, float(w.get("min_share", 0.3))):
            continue
        conditions = []
        if w.get("direction_for_subject") in _DIRECTION[lang]:
            conditions.append(_DIRECTION[lang][w["direction_for_subject"]])
        if w.get("event_move") in _MOVE[lang]:
            conditions.append(_MOVE[lang][w["event_move"]])
        if note := rule.note(lang):
            conditions.append(note)
        out.append(Candidate(rule, "; ".join(conditions) or None))
    return out


def _norm(text: str | None) -> str:
    return " ".join((text or "").lower().replace("ё", "е").split())


def _rule_for(product: str, candidates: list[Candidate]) -> Candidate | None:
    """The fired rule a product name belongs to, in either language."""
    want = _norm(product)
    for c in candidates:
        for have in map(_norm, c.rule.product_names.values()):
            if have and (want == have or want in have or have in want):
                return c
    return None


_WORD = re.compile(r"[^\W\d_]{5,}")
# generic verbs say nothing about what an action is about
_GENERIC = {
    "пров",
    "обсу",
    "оцен",
    "свяж",
    "запр",
    "менед",
    "check",
    "discu",
    "revi",
    "asse",
    "conta",
    "manag",
    "банка",
}


def _topic(text: str) -> set[str]:
    """Four-letter stems of an action's longer words: “Пересмотрите личный бюджет” meets “Пересмотреть личный…”."""
    return {w.lower()[:4] for w in _WORD.findall(text or "")} - {g[:4] for g in _GENERIC}


def _covered(text: str, kept: list[Action]) -> bool:
    """A kept action already says it: two content words in common."""
    want = _topic(text)
    return any(len(want & _topic(a.text)) >= 2 for a in kept)


def _top_up(
    kept: list[Action],
    candidates: list[Candidate],
    content: InsightContent,
    e: NewsEvent,
    lang: str,
    direction: str,
    offer_products: bool,
    product_used: bool,
) -> list[tuple[Action, str]]:
    """Actions from fired rules, up to `MIN_ACTIONS`, for rules code can decide alone: the subject's direction is
    known, there is no condition left for the model (a rate that “stays high”), and the product, if any, may be
    offered. The rule's own wording and rationale are used; the action relies on the insight's first risk or reason."""
    out: list[tuple[Action, str]] = []
    have_text = {_norm(a.text) for a in kept}
    have_products = {_norm(a.product) for a in kept if a.product}
    basis = [c.claim_id for c in (content.risks or content.why_it_matters)[:1]]
    # “p” for policy: an id of the model's own (a1, a2…) may belong to an action cut earlier, and the trace must not
    # name two different actions alike
    used = {a.action_id for a in kept}
    for c in candidates:
        if len(kept) + len(out) >= MIN_ACTIONS:
            break
        w = c.rule.when
        need = w.get("direction_for_subject")
        if (need and need != direction) or w.get("event_move") or w.get("note"):
            continue
        product = c.rule.product(lang)
        if product and (
            not offer_products or e.event_type == EventType.TAX or _norm(product) in have_products
        ):
            continue
        kind = ActionKind(c.rule.kind)
        if kind == ActionKind.APPLY_PRODUCT and (product_used or not product):
            continue
        text = c.rule.text(lang)
        if _norm(text) in have_text or _covered(text, kept):
            continue
        aid = next(f"p{i}" for i in range(1, 10) if f"p{i}" not in used)
        used.add(aid)
        product_used = product_used or kind == ActionKind.APPLY_PRODUCT
        out.append(
            (
                Action(
                    action_id=aid,
                    text=text,
                    kind=kind,
                    rationale=c.rule.rationale(lang),
                    product=product,
                    claim_ids=basis,
                ),
                c.rule.id,
            )
        )
    return out


def enforce_actions(
    content: InsightContent,
    candidates: list[Candidate],
    e: NewsEvent,
    *,
    policy: Policy = DEFAULT_POLICY,
    lang: str = "ru",
    offer_products: bool = True,
    institution_names: Iterable[str] = (),
) -> tuple[InsightContent, list[str]]:
    """Bring the draft's actions in line with the policy. Returns the fixed draft and notes for the trace.

    Besides the `product` field, the text is checked: an action that names a product the fired rules do not offer
    (“apply for factoring”) is removed, one that names an offered product gets it; “contact the manager” without a
    product and without a manager, bank or callback in the text becomes the reader's own analysis or operational
    step.
    """
    en = norm_lang(lang) == "en"
    notes: list[str] = []
    kept: list[Action] = []
    product_used = False
    direction = Direction(content.overall_direction).value
    patterns = policy.product_patterns()
    names = [n.lower() for n in institution_names if n]
    for a in content.actions:
        text = _norm(a.text)
        if not a.product and (named := [k for k, rx in patterns.items() if rx.search(text)]):
            allowed = offer_products and e.event_type != EventType.TAX
            c = next((c for c in candidates if c.rule.product_key in named), None) if allowed else None
            need = c.rule.when.get("direction_for_subject") if c else None
            if c is None or (need and need != direction):
                notes.append(
                    f"{a.action_id}: removed — its text offers a product the fired rules do not ({named[0]})"
                    if en
                    else f"{a.action_id}: убрано — в тексте продукт, которого нет в сработавших правилах ({named[0]})"
                )
                continue
            a = a.model_copy(update={"product": c.rule.product(lang)})
            notes.append(
                f"{a.action_id}: the product named in the text is set — {a.product}"
                if en
                else f"{a.action_id}: продукт из текста указан явно — {a.product}"
            )
        if (
            a.kind in (ActionKind.CONTACT_MANAGER, ActionKind.REQUEST_CALLBACK)
            and not a.product  # a product is offered through the manager anyway
            and not _MANAGER.search(text)
            and not any(n in text for n in names)
        ):
            kind = ActionKind.ANALYSIS if _ANALYSIS.search(text) else ActionKind.OPERATIONAL
            notes.append(
                f"{a.action_id}: {a.kind.value} → {kind.value} — the text does not involve the manager"
                if en
                else f"{a.action_id}: {a.kind.value} → {kind.value} — в тексте нет обращения к менеджеру"
            )
            a = a.model_copy(update={"kind": kind})
        if a.product:
            c = _rule_for(a.product, candidates)
            need = c.rule.when.get("direction_for_subject") if c else None
            if e.event_type == EventType.TAX or not offer_products:
                if not offer_products:
                    why = "products are switched off" if en else "продукты отключены"
                else:
                    why = (
                        "no products for tax events" if en else "для налоговых событий продукт не предлагаем"
                    )
                notes.append(
                    f"{a.action_id}: product “{a.product}” removed — {why}"
                    if en
                    else f"{a.action_id}: продукт «{a.product}» убран — {why}"
                )
                a = a.model_copy(update={"product": None})
            elif c is None or (need and need != direction):
                notes.append(
                    f"{a.action_id}: removed — product “{a.product}” does not follow from fired rules"
                    if en
                    else f"{a.action_id}: убрано — продукт «{a.product}» не следует из сработавших правил"
                )
                continue
            else:
                a = a.model_copy(update={"product": c.rule.product(lang)})
        if a.kind == ActionKind.APPLY_PRODUCT:
            if not a.product:
                a = a.model_copy(update={"kind": ActionKind.CONTACT_MANAGER})
            elif product_used:
                notes.append(
                    f"{a.action_id}: removed — the insight already has a product application"
                    if en
                    else f"{a.action_id}: убрано — на карточке уже есть заявка на продукт"
                )
                continue
            else:
                product_used = True
        kept.append(a)

    if len(kept) < MIN_ACTIONS:
        added = _top_up(kept, candidates, content, e, lang, direction, offer_products, product_used)
        for a, rule_id in added:
            notes.append(
                f"{a.action_id}: added from the fired rule {rule_id} — too few actions survived the checks"
                if en
                else f"{a.action_id}: добавлено по сработавшему правилу {rule_id} — после проверок осталось мало действий"
            )
        kept += [a for a, _ in added]

    if ImpactLevel(content.overall_impact) == ImpactLevel.HIGH and direction == "negative":
        first = next((i for i, a in enumerate(kept) if a.kind == ActionKind.CONTACT_MANAGER), None)
        if first is None:
            kept.insert(
                0,
                Action(
                    action_id="a0",
                    text=t("action.contact_manager.text", lang),
                    kind=ActionKind.CONTACT_MANAGER,
                    rationale=t("action.contact_manager.rationale", lang),
                    claim_ids=[r.claim_id for r in content.risks[:1]],
                ),
            )
            notes.append(
                "contact_manager added first: impact is high and negative"
                if en
                else "первым действием добавлена связь с менеджером: влияние высокое и негативное"
            )
        elif first:
            kept.insert(0, kept.pop(first))

    if not kept:
        fb = policy.fallback(lang)
        kept = [
            Action(action_id="a1", text=fb["text"], kind=ActionKind(fb["kind"]), rationale=fb["rationale"])
        ]
        notes.append(
            "no actions left — fallback action added"
            if en
            else "действий не осталось — добавлено запасное действие"
        )
    if len(kept) > MAX_ACTIONS:
        notes.append(
            f"more than {MAX_ACTIONS} actions — extra removed"
            if en
            else f"действий больше {MAX_ACTIONS} — лишние убраны"
        )
        kept = kept[:MAX_ACTIONS]
    if notes or kept != content.actions:
        content = content.model_copy(update={"actions": kept})
    return content, notes
