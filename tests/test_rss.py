from refract.news.rss import Feed, group_events, parse_feed

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Central bank keeps the rate at 14%</title><link>https://a.example/1</link>
<pubDate>Fri, 23 Oct 2026 13:30:00 +0300</pubDate><description>&lt;p&gt;The board kept the rate.&lt;/p&gt;</description></item>
<item><title>Central bank keeps key rate at 14%, signals pause</title><link>https://b.example/2</link>
<pubDate>Fri, 23 Oct 2026 14:00:00 +0300</pubDate></item>
<item><title>Wheat prices climb</title><link>https://a.example/3</link><pubDate>Mon, 26 Oct 2026 08:00:00 +0000</pubDate></item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Dollar strengthens</title><link href="https://c.example/4"/><published>2026-10-15T13:45:00Z</published><summary>US CPI 3.4%</summary></entry>
</feed>"""


def test_parse_and_group() -> None:
    items = parse_feed(RSS, Feed("Wire", "u", "en")) + parse_feed(ATOM, Feed("Atom", "u", "en"))
    assert len(items) == 4 and items[0][1] == "The board kept the rate."
    events = group_events(items)
    assert len(events) == 3
    rate = next(e for e in events if "14%" in e.title)
    assert len(rate.articles) == 2 and rate.macro_factors == []
