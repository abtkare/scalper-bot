"""Live price board (v10): gold, silver, platinum, copper, oil, dollar index and currencies on the dashboard.

Sources, from most to least live - no extra API keys, no Twelve Data credits used:
  1. the bot's OWN live quotes (Twelve Data / Binance, every 5-30 s) for symbols already on the watchlist
     (gold XAU/USD, EUR/USD, GBP/USD, USD/JPY, BTC, ETH ...)            -> "live"
  2. Yahoo Finance public chart data, ONE batched request every 15 s for everything else:
     spot currencies (~real time) and COMEX/NYMEX futures (exchange delay ~10 min) -> label shows the delay
Information only: nothing here is used for trading decisions.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional

import requests

log = logging.getLogger("prices")

# key, Arabic name, Yahoo symbol, bot ticker (live quote used when the bot watches it), decimals
BOARD = [
    ("XAU", "الذهب (أونصة)", "GC=F", "XAU/USD", 2),
    ("XAG", "الفضة", "SI=F", "XAG/USD", 3),
    ("XPT", "البلاتين", "PL=F", "XPT/USD", 1),
    ("HG", "النحاس (رطل)", "HG=F", "", 4),
    ("WTI", "نفط WTI", "CL=F", "", 2),
    ("BRENT", "نفط برنت", "BZ=F", "", 2),
    ("DXY", "مؤشر الدولار", "DX-Y.NYB", "", 3),
    ("EURUSD", "يورو/دولار", "EURUSD=X", "EUR/USD", 5),
    ("GBPUSD", "استرليني/دولار", "GBPUSD=X", "GBP/USD", 5),
    ("USDJPY", "دولار/ين", "JPY=X", "USD/JPY", 3),
    ("USDCHF", "دولار/فرنك", "CHF=X", "USD/CHF", 5),
    ("USDCAD", "دولار/كندي", "CAD=X", "USD/CAD", 5),
    ("AUDUSD", "أسترالي/دولار", "AUDUSD=X", "AUD/USD", 5),
    ("USDTRY", "دولار/ليرة تركية", "TRY=X", "", 3),
    ("USDIQD", "دولار/دينار (رسمي)", "IQD=X", "", 0),
    ("BTC", "بيتكوين", "BTC-USD", "BTC/USD", 0),
    ("US10Y", "عائد السندات 10 سنوات %", "^TNX", "", 3),
]
YAHOO = "https://query1.finance.yahoo.com/v7/finance/spark"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}


def parse_spark(js: dict) -> Dict[str, dict]:
    out = {}
    for r in (js.get("spark") or {}).get("result") or []:
        try:
            m = r["response"][0]["meta"]
            px, prev = float(m["regularMarketPrice"]), float(m.get("chartPreviousClose") or m.get("previousClose") or 0)
            out[r["symbol"]] = {"price": px, "prev": prev or None, "ts": float(m.get("regularMarketTime") or 0)}
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


def fetch_yahoo(symbols: List[str], timeout: float = 15.0) -> Dict[str, dict]:
    r = requests.get(YAHOO, params={"symbols": ",".join(symbols), "range": "1d", "interval": "5m"},
                     headers=UA, timeout=timeout)
    r.raise_for_status()
    return parse_spark(r.json())


class PriceBoard:
    def __init__(self, live_fn: Callable[[], Dict[str, tuple]], poll_sec: float = 15.0,
                 fetch: Callable[[List[str]], Dict[str, dict]] = None):
        """live_fn() -> {bot ticker: (price, epoch_ts)} from the bot's own quotes."""
        self.live_fn, self.poll = live_fn, max(5.0, poll_sec)
        self.fetch = fetch or fetch_yahoo
        self.y: Dict[str, dict] = {}
        self.err = ""
        self.last_ok = 0.0
        self.lock = threading.Lock()

    def refresh(self) -> None:
        try:
            data = self.fetch([b[2] for b in BOARD])
            if data:
                with self.lock:
                    self.y.update(data)
                self.err, self.last_ok = "", time.time()
        except Exception as e:
            self.err = f"{type(e).__name__}: {e}"[:160]

    def board(self, now: Optional[float] = None) -> dict:
        now = now or time.time()
        try:
            live = self.live_fn() or {}
        except Exception:
            live = {}
        with self.lock:
            y = dict(self.y)
        items = []
        for key, name, ysym, bot_t, dec in BOARD:
            yd = y.get(ysym) or {}
            prev = yd.get("prev")
            lv = live.get(bot_t) if bot_t else None
            if lv and lv[0] and now - lv[1] < 120:                         # the bot's own live quote
                price, ts, src = float(lv[0]), float(lv[1]), "live"
            elif yd:
                price, ts = yd["price"], yd["ts"]
                src = "live" if now - ts < 120 else "delayed"
            else:
                continue
            # day change from the Yahoo series itself (spot vs futures prices differ slightly, ratios don't)
            chg = round((yd["price"] / prev - 1) * 100, 2) if (yd and prev) else None
            items.append({"key": key, "name": name, "price": round(price, dec + 2), "dec": dec, "chg_pct": chg,
                          "age_s": int(max(0, now - ts)), "src": src})
        return {"items": items, "error": self.err, "updated": self.last_ok}

    def start(self, stopping: threading.Event) -> None:
        def loop():
            while not stopping.is_set():
                self.refresh()
                stopping.wait(self.poll)
        threading.Thread(target=loop, name="prices", daemon=True).start()
