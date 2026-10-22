"""Market data helpers: parsing/validation, bar cache, history download (CSV).

Usage:
    python data.py download QQQ 5min 10000
    python data.py download BTC/USD 5min 10000
    python data.py download BTC/USD 5min 10000 --provider binance   # BTCUSDT from Binance (free)

The live bot does NOT use `fetch()` (it uses providers.py with strict timeouts and a credit budget);
`fetch()`/`download()` are for history only.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from datetime import timedelta
from typing import Optional

import pandas as pd
import requests

from config import load_env

log = logging.getLogger("data")

BASE = "https://api.twelvedata.com/time_series"
BINANCE_BASE = os.getenv("BINANCE_BASE", "https://data-api.binance.vision")
COLS = ["open", "high", "low", "close"]

_SECRET = re.compile(r"(apikey=)[^&\s'\"]+|(bot\d+:)[\w-]+", re.I)


def redact(text: str) -> str:
    """Remove API keys / Telegram tokens from any text before it is logged or sent anywhere.
    (requests' exception messages contain the full URL including ?apikey=...)."""
    return _SECRET.sub(lambda m: (m.group(1) or m.group(2)) + "***", str(text))


# ------------------------------------------------------------------ parsing / validation
def to_frame(values: list) -> pd.DataFrame:
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in COLS + ["volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif c == "volume":
            df[c] = 0.0
    df = df.set_index("datetime").sort_index()
    return df[COLS + ["volume"]]


def klines_to_frame(rows: list) -> pd.DataFrame:
    """Binance kline rows: [openTime(ms), o, h, l, c, volume, closeTime, ...]. Index = UTC bar start."""
    df = pd.DataFrame([r[:6] for r in rows], columns=["t", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("t").astype("int64"), unit="ms")
    df.index.name = "datetime"
    for c in COLS + ["volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_index()


def clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Drop anything a strategy/engine must never see: NaN/inf, non-positive prices,
    inconsistent OHLC (high < low, close outside [low, high]), duplicated/unsorted timestamps.
    A NaN high/low would make every stop/target comparison silently False (missed exits)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=COLS + ["volume"])
    d = df.copy()
    d = d[~d.index.isna()]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    ohlc = d[COLS].astype(float)
    ok = ohlc.notna().all(axis=1) & (ohlc > 0).all(axis=1)
    ok &= (ohlc.max(axis=1) < float("inf"))
    ok &= d["high"] >= d[["open", "close", "low"]].max(axis=1) - 1e-12
    ok &= d["low"] <= d[["open", "close", "high"]].min(axis=1) + 1e-12
    dropped = int((~ok).sum())
    if dropped:
        log.warning("dropped %d invalid bar(s)", dropped)
    d = d[ok]
    if "volume" not in d.columns:
        d["volume"] = 0.0
    d["volume"] = d["volume"].fillna(0.0)
    return d


class BarStore:
    """Rolling cache of bars so each cycle only needs to download the last few rows.

    `merge()` returns "gap" when the new rows do not connect to the cache (missed bars / new
    session): the caller then does one full re-download instead of computing indicators on holes.
    """

    def __init__(self, interval_minutes: int, keep: int = 300):
        self.itv = timedelta(minutes=interval_minutes)
        self.keep = keep
        self.df = pd.DataFrame(columns=COLS + ["volume"])

    def merge(self, new: pd.DataFrame) -> str:
        if new is None or new.empty:
            return "empty"
        if self.df.empty:
            self.df = new.tail(self.keep)
            return "ok"
        full = len(new) >= self.keep * 0.5            # a full download is authoritative
        if not full and new.index[0] > self.df.index[-1] + self.itv * 1.5:
            return "gap"
        merged = pd.concat([self.df, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index().tail(self.keep)
        self.df = merged
        return "ok"

    def closed(self, now_local, settle_sec: float) -> pd.DataFrame:
        """Bars that are certainly finished. The last row counts only if its close time plus a small
        settle delay has passed; earlier rows are closed because a later bar exists."""
        d = self.df
        if d.empty:
            return d
        cut = now_local - self.itv - timedelta(seconds=settle_sec)
        return d if d.index[-1] <= cut else d.iloc[:-1]


# ------------------------------------------------------------------ history download (offline use)
def _get_json(url: str, params: dict, what: str, retries: int = 4):
    last = "unknown"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=(5, 30))
            js = r.json()
        except (requests.RequestException, ValueError) as e:
            last = redact(f"{type(e).__name__}: {e}")
            time.sleep(2 * (attempt + 1))
            continue
        if isinstance(js, dict) and js.get("status") == "error":
            if js.get("code") == 429:                      # rate limit: wait and retry
                time.sleep(20 * (attempt + 1))
                last = "rate limit"
                continue
            raise RuntimeError(f"{what}: {redact(js.get('message'))}")
        return js
    raise RuntimeError(f"{what}: failed after {retries} attempts ({last})")


def fetch(symbol: str, interval: str, outputsize: int = 500,
          end_date: Optional[str] = None, api_key: Optional[str] = None,
          retries: int = 4, timezone: Optional[str] = None) -> pd.DataFrame:
    api_key = api_key or os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise RuntimeError("TWELVE_DATA_API_KEY is not set (see .env.example)")
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize,
              "apikey": api_key, "format": "JSON"}
    if end_date:
        params["end_date"] = end_date
    if timezone:
        params["timezone"] = timezone
    js = _get_json(BASE, params, f"Twelve Data {symbol}", retries)
    if not js.get("values"):
        raise RuntimeError(f"Twelve Data returned no bars for {symbol}")
    return clean_bars(to_frame(js["values"]))


def binance_symbol(symbol: str) -> str:
    s = symbol.replace("/", "").upper()
    return s[:-3] + "USDT" if s.endswith("USD") and not s.endswith("USDT") else s


def fetch_binance(symbol: str, interval: str, limit: int = 1000, end_ms: Optional[int] = None) -> pd.DataFrame:
    itv = {"1min": "1m", "5min": "5m", "15min": "15m", "1h": "1h"}.get(interval, interval)
    params = {"symbol": binance_symbol(symbol), "interval": itv, "limit": min(limit, 1000)}
    if end_ms:
        params["endTime"] = end_ms
    rows = _get_json(f"{BINANCE_BASE}/api/v3/klines", params, f"Binance {symbol}")
    if not rows:
        return pd.DataFrame(columns=COLS + ["volume"])
    return clean_bars(klines_to_frame(rows))


def download(symbol: str, interval: str, bars: int, provider: str = "twelvedata") -> pd.DataFrame:
    """Page backwards until `bars` rows are collected."""
    frames, remaining, end_date = [], bars, None
    page = 1000 if provider == "binance" else 5000
    while remaining > 0:
        size = min(page, remaining)
        if provider == "binance":
            chunk = fetch_binance(symbol, interval, size, end_ms=end_date)
        else:
            chunk = fetch(symbol, interval, size, end_date=end_date)
        if chunk.empty:
            break
        frames.append(chunk)
        remaining -= len(chunk)
        if len(chunk) < size:
            break                                   # provider has no more history
        if provider == "binance":
            end_date = int((chunk.index[0] - pd.Timedelta(milliseconds=1)).timestamp() * 1000)
        else:
            end_date = (chunk.index[0] - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
            time.sleep(8)                           # stay under free-plan rate limits
    if not frames:
        raise RuntimeError(f"no data returned for {symbol}")
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


def csv_path(symbol: str, interval: str) -> str:
    return os.path.join("data", f"{symbol.replace('/', '-')}_{interval}.csv")


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime").sort_index()
    return clean_bars(df)


if __name__ == "__main__":
    load_env()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    prov = "twelvedata"
    if "--provider" in sys.argv:
        prov = sys.argv[sys.argv.index("--provider") + 1].lower()
        args = [a for a in args if a != prov]
    if len(args) >= 4 and args[0] == "download":
        sym, itv, n = args[1], args[2], int(args[3])
        os.makedirs("data", exist_ok=True)
        out = download(sym, itv, n, prov)
        out.to_csv(csv_path(sym, itv), index_label="datetime")
        print(f"saved {len(out)} bars: {out.index[0]} -> {out.index[-1]}  ({csv_path(sym, itv)})")
    else:
        print(__doc__)
