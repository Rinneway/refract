"""Attribution repair: small models often leave `features` and `source_ids` empty or wrong.

Code fills them from what the statement visibly refers to — shared word stems, numbers and concept words with the
feature descriptions; a source is the one that holds the statement's numbers or that it nearly retells.
"""

from __future__ import annotations

import re

from ...schemas import InsightContent, Origin, SourceType
from ...textutil import invented_numbers, lexical_support, normalize, numbers, stems
from ..context import InsightContext, known_feature
from .messages import NOTES
from .review import names_institution

# ---------------------------------------------------------------------------------------- attribution repair

_PREFIX = re.compile(r"^\s*(риск|возможность|risk|opportunity)\s*[:—-]\s*", re.IGNORECASE)


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


def _evidence_numbers(text: str) -> set[float]:
    """Numbers that identify a source: not small counts and not years (a forecast “for 2026” is everywhere)."""
    return {n for n in numbers(text) if not (n.is_integer() and (n <= 12 or 1900 <= n <= 2100))}


HOUSE_MIN_SUPPORT = (
    0.6  # without shared numbers, only a near-retelling: a loose match would launder a misattribution
)


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
    - a house view without a research source cites the thesis or excerpt sharing its numbers (two, or the only one;
      across languages; a number from the news does not block the match), else
      the one it nearly retells (≥ 0.6); a statement crediting the institution looks for its source in research first;
    - “Risk:” / “Возможность:” prefixes are removed from the text.

    Every change is reported for the trace; nothing is invented — only links to text that is already in context.
    """
    feats, lang = ctx.features(), ctx.lang
    profile_id = "src_profile"
    texts = {**_source_texts(ctx), profile_id: " ".join([ctx.profile.summary, *(d for _, d in feats)])}
    notes: list[str] = []
    research = {sr.source_id for sr in ctx.sources() if sr.type is SourceType.RESEARCH} & set(texts)

    def fix(c, need_features: bool):
        update: dict = {}
        text = _PREFIX.sub("", c.text)
        if text != c.text and text:
            update["text"] = text[:1].upper() + text[1:]
        features = c.features
        if need_features and features and not all(known_feature(k, feats) for k in features):
            resolved = list(dict.fromkeys(r for k in features if (r := resolve_feature(k, feats))))
            if resolved != features:
                notes.append(NOTES["mapped"][lang].format(cid=c.claim_id, old=features, new=resolved))
                update["features"] = features = resolved
        missing = need_features and not any(known_feature(k, feats) for k in features)
        if missing and (keys := infer_features(c.text, feats)):
            update["features"] = features = keys
            notes.append(NOTES["features"][lang].format(cid=c.claim_id, keys=keys))
        # A house view, or a statement crediting the institution, rests on its research: look there first.
        house = c.origin is Origin.HOUSE_VIEW
        pool = research if (house or names_institution(c.text, ctx)) else set(texts)
        scored = sorted(
            ((lexical_support(c.text, t), sid) for sid, t in texts.items() if sid in pool), reverse=True
        )
        if not house and not scored:
            scored = sorted(((lexical_support(c.text, t), sid) for sid, t in texts.items()), reverse=True)
        has_numbers = bool(invented_numbers([c.text], ""))
        if house and not set(c.source_ids) & research:
            # “house view without research” is a rule error: cite the thesis sharing the statement's numbers (this
            # works across languages; a number taken from the news does not block it), else the one it retells.
            own = _evidence_numbers(c.text)
            need = min(2, len(own))
            by_numbers = sorted(
                ((len(own & numbers(texts[sid])), score, sid) for score, sid in scored), reverse=True
            )
            pick = [sid for shared, _, sid in by_numbers[:1] if need and shared >= need] or [
                sid for score, sid in scored[:1] if score >= HOUSE_MIN_SUPPORT
            ]
            if pick:
                update["source_ids"] = [*c.source_ids, *pick]
                notes.append(NOTES["house_source"][lang].format(cid=c.claim_id, ids=pick))
        elif not c.source_ids:
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
                notes.append(NOTES["sources"][lang].format(cid=c.claim_id, ids=update["source_ids"]))
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
                    notes.append(NOTES["number_source"][lang].format(cid=c.claim_id, ids=more[:1]))
        return c.model_copy(update=update) if update else c

    new = content.model_copy(
        update={
            "what_happened": [fix(c, False) for c in content.what_happened],
            "why_it_matters": [fix(c, True) for c in content.why_it_matters],
            "risks": [fix(c, True) for c in content.risks],
            "opportunities": [fix(c, True) for c in content.opportunities],
            "house_view": [fix(c, False) for c in content.house_view],
        }
    )
    return new, notes
