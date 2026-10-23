"""News window (v9): Trump's Truth Social posts from CNN's free public archive - CONTEXT ONLY.

  * Source: https://ix.cnn.io/data/truth-social/truth_archive.json (no key, newest post first).
    The file is ~20 MB, so we download only its first ~24 KB with an HTTP Range request every 5 minutes
    (~7 MB/day instead of ~6 GB/day). Light on the server.
  * Each new post is tagged by keyword (tariffs/China, Fed/rates, war/oil, markets, dollar/economy).
    Market-relevant posts are pushed to the alerts channel (ntfy -a / Telegram); all appear on the dashboard.
  * It NEVER opens, closes or blocks a trade. Research (RESEARCH.md section 10): over the last year these
    posts did not move QQQ / gold / EUR-USD more than normal in the next hour, and the archive itself lags
    several minutes - so they are information for you, not a trading signal.
  * Switch off with NEWS_WATCH=0. Another source later = another `fetch_*` function returning the same dicts.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Callable, List, Optional

import requests

log = logging.getLogger("news")

CNN_URL = os.getenv("NEWS_URL", "https://ix.cnn.io/data/truth-social/truth_archive.json")
RANGE_BYTES = 24575
STORE = "news.json"

CATS = [  # (key, Arabic label, regex) - first match wins
    ("trade", "رسوم/تجارة/الصين", r"tariff|china|chinese|\bxi\b|trade deal|trade war|\bimports?\b|\bexports?\b"),
    ("fed", "الفيدرالي/الفائدة", r"\bfed\b|powell|interest rate|rate cut|federal reserve|\brates\b"),
    ("war", "حرب/نفط/جيوسياسة", r"iran|israel|russia|ukraine|\boil\b|\bwar\b|missile|strike|venezuela"),
    ("markets", "الأسواق", r"stock market|\bdow\b|nasdaq|s&p|record high|\bstocks\b"),
    ("econ", "الدولار/الاقتصاد", r"dollar|inflation|\bjobs\b|economy|\bgdp\b|recession|gold|crypto|bitcoin"),
]
RELEVANT = {"trade", "fed", "war", "markets", "econ"}
_TAG = re.compile(r"<[^>]+>")


def parse_partial(text: str) -> List[dict]:
    """Complete JSON objects from the START of a JSON array that was cut off anywhere (Range download)."""
    dec = json.JSONDecoder()
    i = text.find("[")
    if i < 0:
        return []
    i += 1
    out = []
    n = len(text)
    while i < n:
        while i < n and text[i] in " \r\n\t,":
            i += 1
        if i >= n or text[i] == "]":
            break
        try:
            obj, i = dec.raw_decode(text, i)
        except ValueError:
            break                                   # the cut-off last object
        if isinstance(obj, dict):
            out.append(obj)
    return out


def classify(text: str) -> str:
    t = (text or "").lower()
    for key, _, rx in CATS:
        if re.search(rx, t):
            return key
    return "other"


def label(cat: str) -> str:
    return next((lab for k, lab, _ in CATS if k == cat), "أخرى")


def clean(text: str, limit: int = 600) -> str:
    t = _TAG.sub(" ", text or "")
    return re.sub(r"\s+", " ", t).strip()[:limit]


def fetch_cnn(session: Optional[requests.Session] = None, timeout: float = 20.0) -> List[dict]:
    s = session or requests
    r = s.get(CNN_URL, headers={"Range": f"bytes=0-{RANGE_BYTES}"}, timeout=timeout)
    r.raise_for_status()
    posts = []
    for o in parse_partial(r.content.decode("utf-8", errors="ignore")):
        pid = str(o.get("id") or "")
        if not pid:
            continue
        txt = clean(o.get("content", ""))
        if not txt and o.get("media"):
            txt = "[صورة/فيديو بلا نص]"
        posts.append({"id": pid, "ts": o.get("created_at", ""), "text": txt, "url": o.get("url", ""),
                      "cat": classify(txt)})
    return posts


class NewsWatcher:
    def __init__(self, notify: Callable[[str], None], poll_sec: float = 300.0, keep: int = 40,
                 fetch: Callable[[], List[dict]] = None, path: str = STORE):
        self.notify, self.poll, self.keep, self.path = notify, max(60.0, poll_sec), keep, path
        self.fetch = fetch or fetch_cnn
        self.items: List[dict] = []
        self.seen: set = set()
        self.health = {"ok": None, "last_check": None, "error": "", "new_today": 0}
        self._day = ""
        self._load()

    # ---- persistence (survives restarts so old posts are not re-alerted)
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.items = d.get("items", [])[: self.keep]
            self.seen = set(d.get("seen", []))
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"items": self.items, "seen": sorted(self.seen)[-500:]}, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as e:
            log.warning("news: could not save (%s)", e)

    def check(self, now: Optional[float] = None) -> List[dict]:
        """One poll. Returns the NEW posts (oldest first). First run only seeds - no backlog alerts."""
        now = now or time.time()
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if day != self._day:
            self._day, self.health["new_today"] = day, 0
        try:
            posts = self.fetch()
        except Exception as e:
            self.health.update(ok=False, last_check=now, error=f"{type(e).__name__}: {e}"[:200])
            return []
        self.health.update(ok=True, last_check=now, error="")
        first_run = not self.seen
        new = [p for p in posts if p["id"] not in self.seen]
        for p in posts:
            self.seen.add(p["id"])
        if first_run:
            self.items = posts[: self.keep]
            self._save()
            return []
        if new:
            self.items = (new + [i for i in self.items if i["id"] not in {p["id"] for p in new}])[: self.keep]
            self.health["new_today"] += len(new)
            self._save()
            for p in reversed(new):                 # oldest first
                if p["cat"] in RELEVANT:
                    self.notify(f"📰 ترامب ({label(p['cat'])}): {p['text'][:300]} — {p['url']}")
        return list(reversed(new))

    def snapshot(self) -> dict:
        return {"health": dict(self.health), "items": [dict(i, label=label(i["cat"])) for i in self.items[:20]]}

    def start(self, stopping: threading.Event) -> None:
        def loop():
            while not stopping.is_set():
                self.check()
                stopping.wait(self.poll)
        threading.Thread(target=loop, name="news", daemon=True).start()
