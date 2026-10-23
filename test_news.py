"""News window: partial-JSON parsing, tagging, de-duplication, no backlog alerts, never raises."""
import json
import os
import tempfile

import news


def test_parse_partial():
    full = json.dumps([{"id": "3", "content": "a"}, {"id": "2", "content": "b, ]"}, {"id": "1", "content": "c"}])
    cut = full[: full.index('"1"') + 3]
    got = news.parse_partial(cut)
    assert [o["id"] for o in got] == ["3", "2"]
    assert news.parse_partial("garbage") == []


def test_classify():
    assert news.classify("New TARIFFS on China!") == "trade"
    assert news.classify("Powell must cut rates") == "fed"
    assert news.classify("Happy birthday") == "other"
    assert news.clean("<p>Hi  <b>there</b></p>") == "Hi there"


def test_watcher_flow():
    feed = [[{"id": "2", "ts": "t", "text": "tariff", "url": "u", "cat": "trade"},
             {"id": "1", "ts": "t", "text": "x", "url": "u", "cat": "other"}]]
    sent = []
    path = os.path.join(tempfile.mkdtemp(), "n.json")
    w = news.NewsWatcher(sent.append, fetch=lambda: feed[0], path=path)
    assert w.check() == [] and sent == []                     # first run seeds only
    feed[0] = [{"id": "4", "ts": "t", "text": "golf", "url": "u", "cat": "other"},
               {"id": "3", "ts": "t", "text": "Fed", "url": "u", "cat": "fed"}] + feed[0]
    new = w.check()
    assert [p["id"] for p in new] == ["3", "4"] and len(sent) == 1 and "الفيدرالي" in sent[0]
    assert w.check() == [] and len(sent) == 1                 # no duplicates
    w2 = news.NewsWatcher(sent.append, fetch=lambda: feed[0], path=path)   # restart: still no re-alert
    assert w2.check() == [] and len(sent) == 1
    w3 = news.NewsWatcher(sent.append, fetch=lambda: (_ for _ in ()).throw(OSError("down")), path=path)
    assert w3.check() == [] and w3.snapshot()["health"]["ok"] is False


if __name__ == "__main__":
    for f in (test_parse_partial, test_classify, test_watcher_flow):
        f()
    print("news tests OK")
