"""Edits of the draft between review and the final insight: pruning, restoring what a revision dropped without
a finding, required blocks, softening, de-duplication, the impact ceiling and the headline.
"""

from __future__ import annotations

import re

from pydantic import ValidationError

from ...i18n import norm_lang, t
from ...schemas import Claim, ImpactLevel, InsightContent, Origin, SubjectClaim
from ...textutil import lexical_support, trim_sentence
from ..context import InsightContext, known_feature
from ..impact import above, event_links, thesis_fits
from .messages import NOTES


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
        and thesis_fits(th, ctx.event) is not False
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


_BLOCKS = ("what_happened", "why_it_matters", "risks", "opportunities", "house_view")


PATCH_SAME = 0.8  # a revised statement this close to an old one is that statement under a new id


def merge_revision(
    old: InsightContent, new: InsightContent, flagged: set[str]
) -> tuple[InsightContent, list[str]]:
    """A revision fixes what the review flagged; it must not rewrite the rest.

    Small models often return a much shorter insight: unflagged risks, the house view or the actions disappear. A
    statement or action that was not flagged and is missing from the revision comes back as it was, unless the
    revision keeps it under another id (nearly the same text) or its id is now taken by another statement.
    """
    restored: list[str] = []
    update: dict = {}
    ids_new = {c.claim_id for c in new.all_claims()} | {a.action_id for a in new.actions}
    texts_new = [c.text for c in new.all_claims()]
    for block in _BLOCKS:
        have = list(getattr(new, block))
        for i, c in enumerate(getattr(old, block)):
            if c.claim_id in flagged or c.claim_id in ids_new:
                continue
            if any(c.text == t or lexical_support(c.text, t) >= PATCH_SAME for t in texts_new):
                continue
            have.insert(min(i, len(have)), c)
            restored.append(c.claim_id)
        update[block] = have
    present = {c.claim_id for b in _BLOCKS for c in update[b]}
    actions = list(new.actions)
    for i, a in enumerate(old.actions):
        if a.action_id in flagged or a.action_id in ids_new or not set(a.claim_ids) <= present:
            continue
        if any(a.text == b.text or lexical_support(a.text, b.text) >= PATCH_SAME for b in new.actions):
            continue
        actions.insert(min(i, len(actions)), a)
        restored.append(a.action_id)
    update["actions"] = actions
    if not new.scenarios and old.scenarios:
        update["scenarios"] = old.scenarios
    if not restored:
        return new, []
    return new.model_copy(update=update), restored


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


def name_counterparties(
    content: InsightContent, profile, lang: str = "ru"
) -> tuple[InsightContent, list[str]]:
    """Payment data are anonymised, so a counterparty has only a code (`cp_import_309`). A code means nothing to the
    reader: it is replaced by what is known — “the supplier with 14% of outgoing payments”. In Russian a word before
    the code keeps its case and the description goes in brackets: «контрагенту (поставщик, 14% исходящих платежей)»."""
    lang = norm_lang(lang)
    names: dict[str, tuple[str, str]] = {}  # code → (as a noun phrase, in brackets after a noun)
    for cp in getattr(profile, "counterparties", None) or []:
        role = "customer" if cp.role == "customer" else "supplier"
        share = f"{cp.share_of_flow:.0%}"
        names[cp.counterparty_id] = (
            t(f"cp.{role}", lang, share=share),
            t(f"cp.{role}.short", lang, share=share),
        )
    if not names:
        return content, []
    ids = "|".join(re.escape(i) for i in sorted(names, key=len, reverse=True))
    pattern = re.compile(
        rf"(?:\b(?P<en>(?:the\s+)?counterpart(?:y|ies))\s+|\b(?P<ru>контрагент\w*)\s+)?\b(?P<id>{ids})\b",
        re.IGNORECASE,
    )
    found: set[str] = set()

    def name(m: re.Match) -> str:
        code = m.group("id")
        found.add(code)
        full, short = names[code]
        if m.group("ru"):
            return f"{m.group('ru')} ({short})"
        return full

    def fix(value):
        if isinstance(value, str):
            return pattern.sub(name, value)
        if isinstance(value, list):
            return [fix(v) for v in value]
        if isinstance(value, dict):
            return {
                k: v if k == "features" or k.endswith(("_id", "_ids")) else fix(v) for k, v in value.items()
            }
        return value

    data = fix(content.model_dump())
    if not found:
        return content, []
    try:
        return InsightContent.model_validate(data), [t("note.counterparty_named", lang, ids=sorted(found))]
    except ValidationError:  # a replacement made a text too long: keep the draft as it was
        return content, []


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


def fallback_blocks(content: InsightContent, ctx: InsightContext) -> tuple[InsightContent, list[str]]:
    """Required blocks that are still empty after pruning get a deterministic statement instead of failing the run:
    “what happened” — the first key fact (or the title) of the news; “why it matters” — the strongest link between
    the event and the subject computed by `impact.event_links`, with a caveat."""
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
        notes.append(NOTES["fallback_what"][lang])
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
                    caveat=NOTES["fallback_caveat"][lang],
                )
            ]
            notes.append(NOTES["fallback_why"][lang])
    return (content.model_copy(update=update) if update else content), notes
