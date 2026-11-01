"""Checks of a draft and repairs of it — deterministic code around the model.

- `rules` — rule checks: sources, origins, feature keys, numbers absent from the context, the impact ceiling;
- `review` — the second model's findings weighed against sources (false alarms lifted to warnings);
- `attribution` — features and sources filled from what a statement visibly refers to;
- `repair` — pruning, restoring, required blocks, softening, de-duplication, the headline.

Semantic contradictions are checked by a second model (`prompts/verify.py`); the orchestrator merges both.
Everything is re-exported here, so callers use `checks.<name>`.
"""

from .attribution import (
    CONCEPTS,
    FEATURE_ALIASES,
    FEATURE_MIN_SCORE,
    HOUSE_MIN_SUPPORT,
    SOURCE_MIN_SUPPORT,
    SOURCE_WITH_PROFILE,
    attribute,
    infer_features,
    resolve_feature,
)
from .repair import (
    DUPLICATE_MIN,
    PATCH_SAME,
    REQUIRED_BLOCKS,
    clamp_impact,
    dedupe_house_view,
    drop_claims,
    fallback_blocks,
    house_view_from_theses,
    keep_required,
    merge_revision,
    short_headline,
    soften,
)
from .review import (
    INFERENCE,
    SUPPORT_MIN,
    cited_text,
    confirm_by_sources,
    for_revision,
    overrule_structural,
    reconcile,
)
from .rules import (
    HEADLINE,
    LANGUAGE_MIN,
    as_insight,
    counterparty_superlatives,
    errors,
    merge,
    rule_check,
    visible_text,
)

__all__ = [
    "CONCEPTS",
    "DUPLICATE_MIN",
    "FEATURE_ALIASES",
    "FEATURE_MIN_SCORE",
    "HEADLINE",
    "HOUSE_MIN_SUPPORT",
    "INFERENCE",
    "LANGUAGE_MIN",
    "PATCH_SAME",
    "REQUIRED_BLOCKS",
    "SOURCE_MIN_SUPPORT",
    "SOURCE_WITH_PROFILE",
    "SUPPORT_MIN",
    "as_insight",
    "attribute",
    "cited_text",
    "clamp_impact",
    "confirm_by_sources",
    "counterparty_superlatives",
    "dedupe_house_view",
    "drop_claims",
    "errors",
    "fallback_blocks",
    "for_revision",
    "house_view_from_theses",
    "infer_features",
    "keep_required",
    "merge",
    "merge_revision",
    "overrule_structural",
    "reconcile",
    "resolve_feature",
    "rule_check",
    "short_headline",
    "soften",
    "visible_text",
]
