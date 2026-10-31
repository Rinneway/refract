"""Next actions by the action policy: code decides which rules fire, the model adapts the wording, code checks it.

Principles: a product only by a fired rule and only if the subject does not hold it; at most one `apply_product`
per insight; with high negative impact `contact_manager` comes first; no products for tax events; an empty list
is replaced by the fallback action from the policy.
"""

from __future__ import annotations

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


def enforce_actions(
    content: InsightContent,
    candidates: list[Candidate],
    e: NewsEvent,
    *,
    policy: Policy = DEFAULT_POLICY,
    lang: str = "ru",
    offer_products: bool = True,
) -> tuple[InsightContent, list[str]]:
    """Bring the draft's actions in line with the policy. Returns the fixed draft and notes for the trace."""
    en = norm_lang(lang) == "en"
    notes: list[str] = []
    kept: list[Action] = []
    product_used = False
    direction = Direction(content.overall_direction).value
    for a in content.actions:
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
