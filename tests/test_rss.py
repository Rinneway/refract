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


CP1251 = """<?xml version="1.0" encoding="windows-1251"?>
<rss xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>
<item><title>Ставка сохранена на уровне 14%</title><link>http://news.test/rate</link>
<pubDate>Mon, 03 Nov 2026 10:00:00 +0300</pubDate><content:encoded><![CDATA[<p>Банк России сохранил ставку.</p>]]></content:encoded></item>
<item><title>Юань подешевел</title><link>http://news.test/yuan</link><pubDate>Mon, 03 Nov 2026 11:00:00 +0300</pubDate></item>
</channel></rss>""".encode("cp1251")
ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Rate held</title>
<link rel="replies" href="http://news.test/rate#comments"/><link rel="alternate" href="http://news.test/rate-en"/>
<updated>2026-11-03T10:00:00Z</updated><summary>The central bank held its rate.</summary></entry></feed>"""
HEADER_ONLY = "<rss><channel><item><title>Новость в заголовке HTTP</title><link>http://news.test/h</link></item></channel></rss>"


def test_feeds_in_other_encodings_and_atom_links(tmp_path, capsys) -> None:
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from refract.cli import main

    pages = {
        "/cp1251.xml": (CP1251, "application/rss+xml"),
        "/atom.xml": (ATOM, "application/atom+xml"),
        "/header.xml": (HEADER_ONLY.encode("cp1251"), "application/rss+xml; charset=windows-1251"),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body, ctype = pages[self.path]
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        feeds = tmp_path / "feeds.json"
        feeds.write_text(
            json.dumps([{"name": n, "url": base + n, "language": "ru"} for n in pages]), encoding="utf-8"
        )
        data = tmp_path / "data"
        assert main(["news", "fetch", "--feeds", str(feeds), "--data", str(data)]) == 0
        (path,) = (data / "events").glob("*.json")
        events = {e["articles"][0]["url"]: e for e in json.loads(path.read_text(encoding="utf-8"))}
        assert events["http://news.test/rate"]["title"] == "Ставка сохранена на уровне 14%"  # not mojibake
        assert events["http://news.test/rate"]["summary"] == "Банк России сохранил ставку."  # content:encoded
        assert "http://news.test/rate-en" in events  # the article, not its comments
        assert events["http://news.test/h"]["title"] == "Новость в заголовке HTTP"  # charset from the header
        capsys.readouterr()
        assert main(["news", "fetch", "--feeds", str(feeds), "--data", str(data)]) == 0  # nothing new
        assert "no new events" in capsys.readouterr().out and len(list((data / "events").glob("*.json"))) == 1
    finally:
        server.shutdown()
