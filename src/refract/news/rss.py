"""A small RSS/Atom collector: feeds → deduplicated `NewsEvent`s. Standard library only.

    refract news fetch --feeds feeds.json --data examples/demo

`feeds.json` is a list of `{"name": "…", "url": "…", "language": "ru"}`. Only titles, links, dates and the
feed's own summaries are stored — not the full article text, which usually belongs to the publisher. Articles about
the same thing (similar titles within 48 hours) are merged into one event. Events have no factors yet; the agent
tags them on first use.
"""

from __future__ import annotations

import hashlib
import html
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from ..knowledge.search import tokens
from ..schemas import NewsArticle, NewsEvent
from ..textutil import read_json

USER_AGENT = "Mozilla/5.0 (compatible; refract-news/0.1; +https://github.com/Rinneway/refract)"
_TAG = re.compile(r"<[^>]+>")
_ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class Feed:
    name: str
    url: str
    language: str = "ru"


def load_feeds(path: str | Path) -> list[Feed]:
    return [Feed(**f) for f in read_json(path)]


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", text or ""))).strip()


def _date(value: str | None) -> datetime:
    if value:
        try:
            d = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            try:
                d = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                d = None
        if d is not None:
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def parse_feed(xml_text: str, feed: Feed) -> list[tuple[NewsArticle, str]]:
    """(article, summary) pairs from an RSS 2.0 or Atom document."""
    root = ET.fromstring(xml_text)
    out: list[tuple[NewsArticle, str]] = []
    items = root.iter("item")
    atom = list(root.iter(f"{_ATOM}entry"))
    for it in items:
        title, link = _clean(it.findtext("title")), (it.findtext("link") or "").strip()
        if title and link:
            out.append(
                (
                    _article(feed, title, link, _date(it.findtext("pubDate"))),
                    _clean(it.findtext("description")),
                )
            )
    for it in atom:
        link_el = it.find(f"{_ATOM}link")
        link = (link_el.get("href") if link_el is not None else "") or ""
        title = _clean(it.findtext(f"{_ATOM}title"))
        when = _date(it.findtext(f"{_ATOM}published") or it.findtext(f"{_ATOM}updated"))
        summary = _clean(it.findtext(f"{_ATOM}summary") or it.findtext(f"{_ATOM}content"))
        if title and link:
            out.append((_article(feed, title, link, when), summary))
    return out


def _article(feed: Feed, title: str, link: str, when: datetime) -> NewsArticle:
    aid = hashlib.sha1(link.encode("utf-8")).hexdigest()[:12]
    return NewsArticle(
        article_id=f"art_{aid}",
        url=link,
        publisher=feed.name,
        title=title,
        published_at=when,
        language=feed.language,
    )


def fetch(feed: Feed, timeout: float = 15) -> list[tuple[NewsArticle, str]]:
    req = urllib.request.Request(feed.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return parse_feed(resp.read().decode("utf-8", errors="replace"), feed)


def _similar(a: str, b: str) -> float:
    x, y = set(tokens(a)), set(tokens(b))
    return len(x & y) / len(x | y) if x | y else 0.0


def group_events(
    items: list[tuple[NewsArticle, str]], threshold: float = 0.5, window_h: int = 48
) -> list[NewsEvent]:
    """Merge articles with similar titles published within `window_h` hours into one event."""
    items = sorted(items, key=lambda x: x[0].published_at)
    groups: list[list[tuple[NewsArticle, str]]] = []
    for art, summary in items:
        for g in groups:
            head = g[0][0]
            if (
                abs(art.published_at - head.published_at) <= timedelta(hours=window_h)
                and _similar(art.title, head.title) >= threshold
            ):
                if all(a.url != art.url for a, _ in g):
                    g.append((art, summary))
                break
        else:
            groups.append([(art, summary)])
    events = []
    for g in groups:
        head, summary = g[0]
        eid = hashlib.sha1("|".join(sorted(a.url for a, _ in g)).encode("utf-8")).hexdigest()[:12]
        events.append(
            NewsEvent(
                event_id=f"ev_{eid}",
                title=head.title,
                summary=summary or head.title,
                published_at=head.published_at,
                articles=[a for a, _ in g],
                language=head.language or "ru",
            )
        )
    return events


def collect(feeds: list[Feed], *, limit_per_feed: int = 30) -> tuple[list[NewsEvent], dict[str, str]]:
    """Fetch every feed; returns events and a per-feed status ("ok: N" or the error)."""
    items, status = [], {}
    for f in feeds:
        try:
            got = fetch(f)[:limit_per_feed]
            items += got
            status[f.name] = f"ok: {len(got)}"
        except Exception as e:  # noqa: BLE001 — one broken feed must not stop the others
            status[f.name] = f"error: {e}"
    return group_events(items), status
