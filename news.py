"""News window (v10): the sources that move gold, oil, the dollar and indices - each item translated to Arabic.

Sources (all free, no API keys, light polling with conditional GET / Range requests):
  decision makers : Trump (Truth Social via CNN's public archive), White House, Federal Reserve (statements +
                    speeches), ECB, Bank of England, Bank of Japan, US Energy Information Administration (EIA)
  market wires    : Investing.com RSS - forex, commodities, economy, economic indicators
  calendar        : ForexFactory weekly calendar (CPI, jobs, rate decisions...) with impact, forecast, previous

Every new item gets an Arabic translation (Google Translate public endpoint, cached; on failure the item is
shown in English and retried). Important items (rate decisions, central-bank statements, market-relevant Trump
posts, high-impact events 15 min before release) are pushed to the alerts channel (ntfy -a / Telegram).

Research (RESEARCH.md section 13): Trump posts showed no tradable average effect -> news NEVER opens or closes
trades. The only trading link is the optional event brake (`event_window`, EVENT_BRAKE in bot.py): no NEW entries
around high-impact releases. Default "shadow" = it only reports what it would have blocked.
"""
from __future__ import annotations

import email.utils
import hashlib
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

log = logging.getLogger("news")

CNN_URL = os.getenv("NEWS_URL", "https://ix.cnn.io/data/truth-social/truth_archive.json")
CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
TR_URL = "https://translate.googleapis.com/translate_a/single"
RANGE_BYTES = 24575
STORE = "news.json"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) scalper-bot news window"}

# id, Arabic name, group, url, poll seconds, base importance (0 info / 1 notable / 2 alert)
FEEDS = [
    ("fed", "الاحتياطي الفيدرالي", "cb", "https://www.federalreserve.gov/feeds/press_all.xml", 600, 1),
    ("fed_sp", "خطابات الفيدرالي", "cb", "https://www.federalreserve.gov/feeds/speeches.xml", 1800, 1),
    ("ecb", "المركزي الأوروبي", "cb", "https://www.ecb.europa.eu/rss/press.html", 900, 1),
    ("boe", "بنك إنجلترا", "cb", "https://www.bankofengland.co.uk/rss/news", 1800, 0),
    ("boj", "بنك اليابان", "cb", "https://www.boj.or.jp/en/rss/whatsnew.xml", 1800, 0),
    ("wh", "البيت الأبيض", "gov", "https://www.whitehouse.gov/news/feed/", 1800, 1),
    ("eia", "إدارة معلومات الطاقة", "gov", "https://www.eia.gov/rss/todayinenergy.xml", 3600, 0),
    ("inv_fx", "Investing — العملات", "mkt", "https://www.investing.com/rss/news_1.rss", 600, 0),
    ("inv_cmd", "Investing — السلع", "mkt", "https://www.investing.com/rss/news_11.rss", 600, 0),
    ("inv_eco", "Investing — الاقتصاد", "mkt", "https://www.investing.com/rss/news_14.rss", 900, 0),
    ("inv_ind", "Investing — المؤشرات", "mkt", "https://www.investing.com/rss/news_95.rss", 900, 0),
]
GROUPS = {"trump": "ترامب", "cb": "البنوك المركزية", "gov": "الحكومة والطاقة", "mkt": "الأسواق"}

CATS = [  # (key, Arabic label, regex) - first match wins
    ("trade", "رسوم/تجارة/الصين", r"tariff|china|chinese|\bxi\b|trade deal|trade war|\bimports?\b|\bexports?\b"),
    ("fed", "الفائدة/البنوك المركزية", r"\bfed\b|powell|interest rate|rate cut|rate hike|federal reserve|\brates\b|fomc|monetary policy"),
    ("war", "حرب/نفط/جيوسياسة", r"iran|israel|russia|ukraine|\boil\b|\bopec|\bwar\b|missile|strike|venezuela|houthi|sanction"),
    ("markets", "الأسواق", r"stock market|\bdow\b|nasdaq|s&p|record high|\bstocks\b|treasury yields?|bond"),
    ("econ", "الدولار/الاقتصاد", r"dollar|inflation|\bcpi\b|\bjobs\b|payroll|unemployment|economy|\bgdp\b|recession|gold|crypto|bitcoin|\bpmi\b"),
]
RELEVANT = {"trade", "fed", "war", "markets", "econ"}
# central-bank / government headlines that deserve a phone alert
HOT = re.compile(r"monetary policy (decision|statement)|fomc statement|federal reserve issues fomc|policy rate|"
                 r"interest rates?\b.*(decision|unchanged|raise|lower|cut)|bank rate|minutes of the federal open|"
                 r"executive order|tariff|proclamation", re.I)
CAL_CCY = {"USD", "EUR", "GBP", "JPY", "CNY", "All"}
_TAG = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------- helpers
def parse_partial(text: str) -> List[dict]:
    """Complete JSON objects from the START of a JSON array that was cut off anywhere (Range download)."""
    dec = json.JSONDecoder()
    i = text.find("[")
    if i < 0:
        return []
    i += 1
    out, n = [], len(text)
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
    for a, b in (("&amp;", "&"), ("&#39;", "'"), ("&quot;", '"'), ("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()[:limit]


def iso(ts) -> str:
    """Any RSS/Atom/ISO date -> 'YYYY-MM-DDTHH:MM:SSZ' (UTC). Unknown -> now."""
    dt = None
    if ts:
        try:
            dt = email.utils.parsedate_to_datetime(ts)
        except (TypeError, ValueError):
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except ValueError:
                dt = None
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def translate(text: str, session=None, timeout: float = 15.0) -> str:
    """English -> Arabic via Google Translate's public endpoint. Raises on failure (caller retries later)."""
    if not text:
        return ""
    s = session or requests
    parts = []
    for i in range(0, len(text), 1500):
        r = s.get(TR_URL, params={"client": "gtx", "sl": "auto", "tl": "ar", "dt": "t", "q": text[i:i + 1500]},
                  headers=UA, timeout=timeout)
        r.raise_for_status()
        parts.append("".join(seg[0] for seg in (r.json()[0] or []) if seg and seg[0]))
    return "".join(parts).strip()


MM_URL = "https://api.mymemory.translated.net/get"


def translate_mymemory(text: str, session=None, timeout: float = 15.0) -> str:
    """Fallback translator (free, ~5000 chars/day per IP)."""
    s = session or requests
    r = s.get(MM_URL, params={"q": text[:480], "langpair": "en|ar"}, headers=UA, timeout=timeout)
    r.raise_for_status()
    out = (r.json().get("responseData") or {}).get("translatedText") or ""
    if not out or "MYMEMORY WARNING" in out.upper():
        raise RuntimeError("mymemory quota")
    return out.strip()


class Translator:
    """Google first (paced: one call per 2 s); after a 429/error it rests 15 min and MyMemory is used meanwhile."""

    def __init__(self, session=None, gap: float = 2.0):
        self.s, self.gap = session or requests.Session(), gap
        self.last = 0.0
        self.google_off_until = 0.0

    def __call__(self, text: str) -> str:
        now = time.time()
        if now >= self.google_off_until:
            wait = self.last + self.gap - now
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                return translate(text, self.s)
            except Exception:
                self.google_off_until = time.time() + 15 * 60
        return translate_mymemory(text, self.s)


# ---------------------------------------------------------------- sources
def fetch_cnn(session=None, timeout: float = 20.0) -> List[dict]:
    s = session or requests
    r = s.get(CNN_URL, headers={"Range": f"bytes=0-{RANGE_BYTES}", **UA}, timeout=timeout)
    r.raise_for_status()
    posts = []
    for o in parse_partial(r.content.decode("utf-8", errors="ignore")):
        pid = str(o.get("id") or "")
        if not pid:
            continue
        txt = clean(o.get("content", ""))
        if not txt and o.get("media"):
            txt = "[صورة/فيديو بلا نص]"
        cat = classify(txt)
        posts.append({"id": "trump:" + pid, "src": "trump", "src_ar": "ترامب — Truth Social", "group": "trump",
                      "ts": iso(o.get("created_at")), "text": txt, "url": o.get("url", ""), "cat": cat,
                      "imp": 2 if cat in RELEVANT else 0})
    return posts


def parse_feed(xml_bytes: bytes, fid: str, name: str, group: str, base_imp: int) -> List[dict]:
    root = ET.fromstring(xml_bytes)
    items = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        f = {c.tag.split("}")[-1]: c for c in el}
        title = clean((f["title"].text if "title" in f else "") or "", 400)
        if not title:
            continue
        link = ""
        if "link" in f:
            link = (f["link"].text or f["link"].get("href") or "").strip()
        date = next((f[k].text for k in ("pubDate", "published", "updated", "date") if k in f and f[k].text), "")
        guid = (f["guid"].text if "guid" in f and f["guid"].text else link or title)
        cat = classify(title)
        imp = 2 if HOT.search(title) else base_imp
        items.append({"id": f"{fid}:" + hashlib.sha1(guid.encode()).hexdigest()[:16], "src": fid, "src_ar": name,
                      "group": group, "ts": iso(date), "text": title, "url": link, "cat": cat, "imp": imp})
    items.sort(key=lambda x: x["ts"], reverse=True)
    return items


def parse_calendar(js: list) -> List[dict]:
    out = []
    for e in js or []:
        if e.get("country") not in CAL_CCY or e.get("impact") not in ("High", "Medium"):
            continue
        out.append({"id": "cal:" + hashlib.sha1(f"{e.get('title')}{e.get('country')}{e.get('date')}".encode()).hexdigest()[:16],
                    "title": clean(e.get("title", ""), 200), "ccy": e.get("country"), "ts": iso(e.get("date")),
                    "impact": e.get("impact"), "forecast": e.get("forecast", ""), "previous": e.get("previous", ""),
                    "actual": e.get("actual", "")})
    out.sort(key=lambda x: x["ts"])
    return out


def epoch(ts_iso: str) -> float:
    return datetime.strptime(ts_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


# ---------------------------------------------------------------- watcher
class NewsWatcher:
    def __init__(self, notify: Callable[[str], None], poll_sec: float = 300.0, keep: int = 90,
                 fetch: Callable[[], List[dict]] = None, path: str = STORE, feeds=None,
                 http: Optional[Callable] = None, translator: Optional[Callable[[str], str]] = None,
                 calendar: Optional[Callable[[], List[dict]]] = None):
        self.notify, self.keep, self.path = notify, keep, path
        self.poll = max(60.0, poll_sec)
        self.fetch_trump = fetch or fetch_cnn
        self.feeds = FEEDS if feeds is None else feeds
        self.http = http or self._http_get
        self.tr = translator or Translator()
        self.cal_fetch = calendar or self._fetch_calendar
        self.items: List[dict] = []
        self.seen: set = set()
        self.cal: List[dict] = []
        self.cal_alerted: set = set()
        self.cal_tr: Dict[str, str] = {}
        self.health: Dict[str, dict] = {}
        self.next_due: Dict[str, float] = {}
        self.etags: Dict[str, dict] = {}
        self.new_today, self._day = 0, ""
        self.lock = threading.Lock()
        self.sess = requests.Session()
        self._load()

    # ---- persistence (survives restarts so old items are not re-alerted)
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            items = d.get("items", [])
            for i in items:                               # v9 files: plain ids, no source fields
                if ":" not in i["id"]:
                    i.update(id="trump:" + i["id"], src="trump", src_ar="ترامب — Truth Social", group="trump",
                             imp=2 if i.get("cat") in RELEVANT else 0)
            self.items = items[: self.keep]
            self.seen = {s if ":" in s else "trump:" + s for s in d.get("seen", [])}
            self.cal_alerted = set(d.get("cal_alerted", []))
            self.cal_tr = d.get("cal_tr", {})
        except (OSError, ValueError, KeyError):
            pass

    def _save(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"items": self.items, "seen": sorted(self.seen)[-3000:],
                           "cal_alerted": sorted(self.cal_alerted)[-200:],
                           "cal_tr": dict(list(self.cal_tr.items())[-300:])}, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as e:
            log.warning("news: could not save (%s)", e)

    def _http_get(self, url: str) -> Optional[bytes]:
        """Conditional GET: unchanged feeds cost a few hundred bytes. None = not modified."""
        h = dict(UA)
        c = self.etags.get(url) or {}
        if c.get("etag"):
            h["If-None-Match"] = c["etag"]
        if c.get("lm"):
            h["If-Modified-Since"] = c["lm"]
        r = self.sess.get(url, headers=h, timeout=20)
        if r.status_code == 304:
            return None
        r.raise_for_status()
        self.etags[url] = {"etag": r.headers.get("ETag", ""), "lm": r.headers.get("Last-Modified", "")}
        return r.content

    def _fetch_calendar(self) -> List[dict]:
        r = self.sess.get(CAL_URL, headers=UA, timeout=20)
        r.raise_for_status()
        return parse_calendar(r.json())

    def _ok(self, src: str, now: float, err: str = "") -> None:
        self.health[src] = {"ok": not err, "t": now, "err": err[:160]}

    # ---- one source -> new items
    def _ingest(self, src: str, items: List[dict], now: float) -> List[dict]:
        first = not any(s.startswith(src + ":") for s in self.seen)
        new = [i for i in items if i["id"] not in self.seen]
        self.seen.update(i["id"] for i in items)
        if first:
            new = new[:8]                       # seed: show the latest few, no backlog alerts
            for i in new:
                i["seed"] = True
        return new

    def _translate_missing(self, budget: int = 20) -> int:
        start = budget
        for i in list(self.items):
            if budget <= 0:
                break
            if i.get("ar") or not i.get("text") or i["text"].startswith("["):
                continue
            try:
                i["ar"] = self.tr(i["text"])
            except Exception as e:
                log.info("news: translation failed (%s) - retry next cycle", type(e).__name__)
                return start - budget
            budget -= 1
        for e in self.cal:
            if budget <= 0:
                break
            if e["title"] in self.cal_tr:
                continue
            try:
                self.cal_tr[e["title"]] = self.tr(e["title"])
            except Exception:
                return start - budget
            budget -= 1
        return start - budget

    def check(self, now: Optional[float] = None) -> List[dict]:
        """Polls every source that is due. Returns the NEW (non-seed) items, oldest first."""
        now = now or time.time()
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if day != self._day:
            self._day, self.new_today = day, 0
        fresh: List[dict] = []
        sources = [("trump", self.poll, None)] + [(f[0], f[4], f) for f in self.feeds] + [("calendar", 1800, None)]
        for src, every, feed in sources:
            if now < self.next_due.get(src, 0):
                continue
            self.next_due[src] = now + every
            try:
                if src == "trump":
                    fresh += self._ingest(src, self.fetch_trump(), now)
                elif src == "calendar":
                    self.cal = self.cal_fetch()
                else:
                    raw = self.http(feed[3])
                    if raw is not None:
                        fresh += self._ingest(src, parse_feed(raw, feed[0], feed[1], feed[2], feed[5]), now)
                self._ok(src, now)
            except Exception as e:
                self._ok(src, now, f"{type(e).__name__}: {e}")
        with self.lock:
            if fresh:
                ids = {i["id"] for i in fresh}
                self.items = sorted(fresh + [i for i in self.items if i["id"] not in ids],
                                    key=lambda x: x["ts"], reverse=True)[: self.keep]
        translated = self._translate_missing()          # outside the lock: may take ~40 s (paced)
        real = [i for i in fresh if not i.get("seed")]
        self.new_today += len(real)
        for i in sorted(real, key=lambda x: x["ts"]):
            if i.get("imp", 0) >= 2:
                cur = next((x for x in self.items if x["id"] == i["id"]), i)
                ar = cur.get("ar") or ""
                self.notify(f"📰 {i['src_ar']}: {ar or i['text']}"[:420] + (f"\n{i['text'][:200]}" if ar else "")
                            + (f"\n{i['url']}" if i.get("url") else ""))
        self._calendar_alerts(now)
        if fresh or translated:
            self._save()
        return sorted(real, key=lambda x: x["ts"])

    def _calendar_alerts(self, now: float) -> None:
        for e in self.cal:
            if e["impact"] != "High" or e["id"] in self.cal_alerted:
                continue
            t = epoch(e["ts"])
            if 0 <= t - now <= 15 * 60:
                self.cal_alerted.add(e["id"])
                bg = datetime.fromtimestamp(t, ZoneInfo("Asia/Baghdad")).strftime("%H:%M")
                ar = self.cal_tr.get(e["title"], "")
                self.notify(f"📅 خلال {int((t - now) // 60)} دقيقة: {ar or e['title']} ({e['ccy']}) — "
                            f"المتوقع {e['forecast'] or '–'} · السابق {e['previous'] or '–'} · تأثير عالٍ ({bg} بغداد)")
                self._save()

    # ---- used by the bot's event brake
    def event_window(self, currencies, now: Optional[float] = None, before: float = 10 * 60,
                     after: float = 20 * 60) -> Optional[str]:
        """Title of a HIGH-impact release for one of `currencies` within [t-before, t+after], else None."""
        now = now or time.time()
        cs = set(currencies) | {"All"}
        for e in self.cal:
            if e["impact"] == "High" and e["ccy"] in cs:
                t = epoch(e["ts"])
                if t - before <= now <= t + after:
                    return f"{e['ccy']} {e['title']}"
        return None

    def snapshot(self, now: Optional[float] = None) -> dict:
        now = now or time.time()
        with self.lock:
            items = [dict(i, label=label(i.get("cat", "other"))) for i in self.items[:60]]
        cal = [dict(e, ar=self.cal_tr.get(e["title"], ""), past=epoch(e["ts"]) < now - 3600)
               for e in self.cal if epoch(e["ts"]) > now - 6 * 3600][:25]
        bad = {k: v["err"] for k, v in self.health.items() if not v["ok"]}
        return {"health": {"ok": not bad if self.health else None, "bad": bad, "new_today": self.new_today,
                           "sources": len(self.health)},
                "groups": GROUPS, "items": items, "calendar": cal}

    def start(self, stopping: threading.Event) -> None:
        def loop():
            while not stopping.is_set():
                try:
                    self.check()
                except Exception as e:                 # never kill the thread
                    log.warning("news loop error: %s", e)
                stopping.wait(60)
        threading.Thread(target=loop, name="news", daemon=True).start()
