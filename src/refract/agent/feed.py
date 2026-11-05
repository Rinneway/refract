"""A personal news feed: which of the stored events concern this subject, strongest first — computed by code.

The agent explains one event for one subject. Before that, a reader with a hundred news items needs to know which
few matter. The feed ranks events by the same typed links that bound the impact level (`impact.event_links`): an
event that names the subject's activity, a country it trades with, a large counterparty's industry or a factor it
is sensitive to comes first. Ranking needs no model; only news that has not been tagged yet is tagged once
(`tag_event`, the agent's own tagging step).

Relevance:
- `direct` — at least a moderate link (the impact ceiling is medium or high);
- `background` — only weak links (the same section letter, a weak sensitivity);
- `none` — nothing in common.

Within a level, events are ordered by the ceiling, the number of links, the event's own importance and recency.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

from ..config import Settings
from ..i18n import norm_lang
from ..llm.client import LLM, LLMError
from ..prompts import classify as classify_prompt
from ..schemas import NewsEvent, Profile
from ..schemas.feed import Feed, FeedItem, Relevance
from .impact import DEFAULT_IMPACT, MODERATE, ORDER, WEAK, ImpactPolicy, event_links, impact_cap


def relevance(strength: int) -> Relevance:
    """Relevance of an event from the strength of its strongest link (0–3)."""
    return "direct" if strength >= MODERATE else "background" if strength >= WEAK else "none"


def tag_event(
    llm: LLM, settings: Settings, event: NewsEvent, lang: str = "ru", step: str = "classify_news"
) -> NewsEvent | None:
    """Tag news that has no factors yet (the agent's tagging step, without a trace). None if the model failed."""
    r = settings.route("classify_news")
    try:
        res = llm.chat_json(
            r.model,
            classify_prompt.build_messages(event, lang),
            classify_prompt.NewsTags,
            max_tokens=r.max_tokens,
            temperature=r.temperature,
            step=step,
            timeout_s=r.timeout_s,
        )
    except LLMError:
        return None
    return classify_prompt.apply(event, res.value) if res.value else None


def item(
    profile: Profile, event: NewsEvent, lang: str = "ru", policy: ImpactPolicy | None = None
) -> FeedItem:
    policy = policy or DEFAULT_IMPACT
    links = event_links(profile, event, lang, policy)
    strength = max((x.level for x in links), default=0)
    cap = impact_cap(profile, event, links, lang, policy)
    links = sorted(links, key=lambda x: -x.level)
    return FeedItem(
        event_id=event.event_id,
        title=event.title,
        published_at=event.published_at,
        language=event.language,
        relevance=relevance(strength),
        strength=strength,
        impact_ceiling=cap.level,
        reason=cap.reason,
        links=[x.render(lang) for x in links],
        importance=event.importance,
        tagged=bool(event.macro_factors),
    )


def _key(it: FeedItem) -> tuple:
    published = it.published_at if it.published_at.tzinfo else it.published_at.replace(tzinfo=timezone.utc)
    return (-it.strength, -ORDER[it.impact_ceiling], -len(it.links), -it.importance, -published.timestamp())


def rank(
    profile: Profile,
    events: Iterable[NewsEvent],
    *,
    lang: str = "ru",
    policy: ImpactPolicy | None = None,
    top: int | None = 10,
    since: datetime | None = None,
    include_unrelated: bool = False,
    untagged: list[str] | None = None,
) -> Feed:
    """The feed for one subject. `since` keeps events published at or after it; `top=None` keeps all."""
    lang = norm_lang(lang)
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    items = []
    for e in events:
        published = e.published_at if e.published_at.tzinfo else e.published_at.replace(tzinfo=timezone.utc)
        if since is not None and published < since:
            continue
        it = item(profile, e, lang, policy)
        if include_unrelated or it.relevance != "none":
            items.append(it)
    items.sort(key=_key)
    return Feed(
        profile_id=profile.profile_id,
        language=lang,
        created_at=datetime.now(timezone.utc),
        items=items[:top] if top is not None else items,
        considered=len(items),
        untagged=list(untagged or []),
    )
