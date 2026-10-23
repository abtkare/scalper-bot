"""News window: parsing, tagging, translation, de-duplication, no backlog alerts, calendar + event window."""
import json
import os
import tempfile
import time

import news

RSS = b"""<?xml version="1.0"?><rss><channel><title>x</title>
<item><title>Federal Reserve issues FOMC statement</title><link>https://f/1</link><guid>g1</guid>
<pubDate>Wed, 21 Oct 2026 18:00:00 GMT</pubDate></item>
<item><title>Speech by Governor X on banking</title><link>https://f/2</link><guid>g2</guid>
<pubDate>Tue, 20 Oct 2026 14:00:00 GMT</pubDate></item></channel></rss>"""
ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Oil &amp; gas output</title>
<link href="https://e/1"/><id>e1</id><updated>2026-10-22T10:00:00Z</updated></entry></feed>"""


def test_parse_partial_and_classify():
    full = json.dumps([{"id": "3", "content": "a"}, {"id": "2", "content": "b, ]"}, {"id": "1", "content": "c"}])
    assert [o["id"] for o in news.parse_partial(full[: full.index('"1"') + 3])] == ["3", "2"]
    assert news.parse_partial("garbage") == []
    assert news.classify("New TARIFFS on China!") == "trade"
    assert news.classify("Powell must cut rates") == "fed"
    assert news.classify("Happy birthday") == "other"
    assert news.clean("<p>Hi  <b>there</b> &amp; you</p>") == "Hi there & you"


def test_feeds():
    it = news.parse_feed(RSS, "fed", "الفيدرالي", "cb", 1)
    assert [i["imp"] for i in it] == [2, 1] and it[0]["ts"] == "2026-10-21T18:00:00Z"
    a = news.parse_feed(ATOM, "eia", "الطاقة", "gov", 0)
    assert a[0]["url"] == "https://e/1" and a[0]["text"] == "Oil & gas output" and a[0]["cat"] == "war"


def test_calendar_and_event_window():
    t = time.time()
    iso = lambda s: time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(s))
    cal = news.parse_calendar([
        {"title": "CPI m/m", "country": "USD", "date": iso(t + 300), "impact": "High", "forecast": "0.3%", "previous": "0.2%"},
        {"title": "Bank Holiday", "country": "USD", "date": iso(t), "impact": "Holiday"},
        {"title": "AUD thing", "country": "AUD", "date": iso(t), "impact": "High"}])
    assert len(cal) == 1
    sent = []
    w = news.NewsWatcher(sent.append, path=os.path.join(tempfile.mkdtemp(), "n.json"), feeds=[],
                         fetch=lambda: [], calendar=lambda: cal, translator=lambda s: "ع:" + s)
    w.check(t)
    assert w.event_window({"USD"}, t) == "USD CPI m/m"
    assert w.event_window({"EUR"}, t) is None
    assert w.event_window({"USD"}, t + 3600) is None
    assert len(sent) == 1 and "ع:CPI m/m" in sent[0]          # 15-min warning, translated
    w.check(t + 60)
    assert len(sent) == 1                                      # only once


def test_watcher_flow():
    feed = [[{"id": "trump:2", "src": "trump", "src_ar": "ترامب", "group": "trump", "ts": "2026-10-22T10:00:00Z",
              "text": "tariff", "url": "u", "cat": "trade", "imp": 2}]]
    rss = [RSS]
    sent, path = [], os.path.join(tempfile.mkdtemp(), "n.json")
    mk = lambda: news.NewsWatcher(sent.append, fetch=lambda: feed[0], path=path,
                                  feeds=[("fed", "الفيدرالي", "cb", "http://x", 600, 1)], http=lambda u: rss[0],
                                  calendar=lambda: [], translator=lambda s: "ترجمة " + s)
    w = mk()
    assert w.check(1000.0) == [] and sent == []               # first run seeds only
    assert all(i.get("ar", "").startswith("ترجمة") for i in w.items) and len(w.items) == 3
    feed[0] = [{"id": "trump:3", "src": "trump", "src_ar": "ترامب", "group": "trump", "ts": "2026-10-22T11:00:00Z",
                "text": "Fed now", "url": "u", "cat": "fed", "imp": 2}] + feed[0]
    rss[0] = RSS.replace(b"<item><title>Federal", b"<item><title>Bank keeps interest rates unchanged</title>"
                         b"<guid>g9</guid><pubDate>Thu, 22 Oct 2026 12:00:00 GMT</pubDate></item><item><title>Federal")
    new = w.check(5000.0)
    assert [i["id"].split(":")[0] for i in new] == ["trump", "fed"] and len(sent) == 2
    assert "ترجمة Fed now" in sent[0]
    assert w.check(9000.0) == [] and len(sent) == 2           # no duplicates
    w2 = mk()                                                  # restart: still no re-alert
    assert w2.check(20000.0) == [] and len(sent) == 2
    w3 = news.NewsWatcher(sent.append, fetch=lambda: (_ for _ in ()).throw(OSError("down")), path=path,
                          feeds=[], calendar=lambda: [], translator=lambda s: s)
    assert w3.check() == [] and w3.snapshot()["health"]["ok"] is False


def test_translation_failure_keeps_english():
    def bad(s):
        raise OSError("429")
    w = news.NewsWatcher(lambda m: None, fetch=lambda: [{"id": "trump:1", "src": "trump", "src_ar": "ت", "group": "trump",
                         "ts": "2026-10-22T10:00:00Z", "text": "hello", "url": "", "cat": "other", "imp": 0}],
                         path=os.path.join(tempfile.mkdtemp(), "n.json"), feeds=[], calendar=lambda: [], translator=bad)
    w.check()
    assert w.items[0]["text"] == "hello" and not w.items[0].get("ar")


if __name__ == "__main__":
    for f in (test_parse_partial_and_classify, test_feeds, test_calendar_and_event_window, test_watcher_flow,
              test_translation_failure_keeps_english):
        f()
    print("news tests OK")
