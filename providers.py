"""Live market-data providers: bars + real-time price, with hard timeouts, a credit budget and
tick sanity filtering.

Design rules (each one exists to prevent a loss or a hang):
  * every HTTP call has a short connect/read timeout  -> the bot can never be frozen by the network
  * every failure becomes a `ProviderError(kind=...)` with the API key REMOVED from the message
  * Twelve Data calls pass through a token bucket sized to your plan  -> no 429 storms
  * one keep-alive HTTP session per thread                             -> no TLS handshake per call
  * `QuoteFilter` rejects single bad ticks                             -> a glitch cannot trigger a stop
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Dict, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from config import Params, SymbolCfg
from data import BASE as TD_BARS_URL, BINANCE_BASE, binance_symbol, clean_bars, klines_to_frame, redact, to_frame

log = logging.getLogger("providers")
TD_BASE = TD_BARS_URL.rsplit("/", 1)[0]          # https://api.twelvedata.com


class ProviderError(Exception):
    """kind: rate | network | server | auth | data"""

    def __init__(self, kind: str, msg: str, retry_after: float = 0.0):
        super().__init__(f"{kind}: {msg}")
        self.kind, self.retry_after = kind, retry_after


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    last: float
    ts: float            # local epoch seconds when the reply was received
    src: str = ""

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


# ------------------------------------------------------------------ rate budget
class RateBudget:
    """Token bucket. Sustained rate = 75% of the plan limit and a small burst, so that even with
    the provider's calendar-minute counting the plan limit is never exceeded."""

    def __init__(self, per_min: float, clock=time.time):
        self.clock = clock
        self.rate = max(per_min * 0.75, 1.0) / 60.0
        self.cap = max(2.0, per_min * 0.25)
        self.tokens = self.cap
        self.t = clock()
        self.blocked_until = 0.0
        self.lock = threading.Lock()

    def try_acquire(self, n: float = 1.0) -> bool:
        with self.lock:
            now = self.clock()
            self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate)
            self.t = now
            if now < self.blocked_until or self.tokens < n:
                return False
            self.tokens -= n
            return True

    def penalize(self, seconds: float) -> None:
        with self.lock:
            self.blocked_until = max(self.blocked_until, self.clock() + seconds)
            self.tokens = 0.0


# ------------------------------------------------------------------ http
_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        ad = HTTPAdapter(pool_connections=2, pool_maxsize=4, max_retries=0)
        s.mount("https://", ad)
        s.mount("http://", ad)
        s.headers["Accept-Encoding"] = "gzip"
        _tls.s = s
    return s


class _Base:
    name = "base"
    budget: Optional[RateBudget] = None

    def __init__(self, p: Params):
        self.timeout = (p.http_connect_timeout, p.http_read_timeout)
        self.skew = 0.0              # local clock minus server clock, seconds (smoothed)
        self._skew_seen = False

    def _note_clock(self, r: requests.Response) -> None:
        d = r.headers.get("Date")
        if not d:
            return
        try:
            skew = time.time() - parsedate_to_datetime(d).timestamp()
        except Exception:
            return
        self.skew = skew if not self._skew_seen else 0.8 * self.skew + 0.2 * skew
        self._skew_seen = True

    def _request(self, url: str, params: dict):
        try:
            r = _session().get(url, params=params, timeout=self.timeout)
        except requests.RequestException as e:
            # `from None`: the chained exception's text contains the URL (and the API key)
            raise ProviderError("network", redact(f"{type(e).__name__}: {e}")) from None
        self._note_clock(r)
        try:
            js = r.json()
        except ValueError:
            raise ProviderError("server", f"non-JSON reply (HTTP {r.status_code})") from None
        return r, js

    def bars(self, cfg: SymbolCfg, n: int) -> pd.DataFrame:      # pragma: no cover - interface
        raise NotImplementedError

    def quote(self, cfg: SymbolCfg) -> Quote:                    # pragma: no cover - interface
        raise NotImplementedError


def _pos_float(x, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ProviderError("data", f"bad {what}: {x!r}") from None
    if not math.isfinite(v) or v <= 0:
        raise ProviderError("data", f"bad {what}: {x!r}")
    return v


# ------------------------------------------------------------------ Twelve Data
class TwelveData(_Base):
    name = "twelvedata"

    def __init__(self, p: Params, api_key: Optional[str] = None, budget: Optional[RateBudget] = None):
        super().__init__(p)
        self.key = api_key or os.getenv("TWELVE_DATA_API_KEY", "")
        self.budget = budget if budget is not None else RateBudget(p.td_credits_per_min)

    def _get(self, path: str, params: dict) -> dict:
        r, js = self._request(TD_BASE + path, dict(params, apikey=self.key, format="JSON"))
        if isinstance(js, dict) and js.get("status") == "error":
            code = int(js.get("code") or r.status_code or 0)
            msg = redact(js.get("message"))
            if code == 429:
                self.budget.penalize(15.0)
                raise ProviderError("rate", msg, retry_after=15.0)
            kind = "auth" if code in (401, 403) else "data" if code in (400, 404) else "server"
            raise ProviderError(kind, f"{code}: {msg}")
        if r.status_code >= 400:
            raise ProviderError("server", f"HTTP {r.status_code}")
        if not isinstance(js, dict):
            raise ProviderError("server", "unexpected reply shape")
        return js

    def bars(self, cfg: SymbolCfg, n: int) -> pd.DataFrame:
        js = self._get("/time_series", {"symbol": cfg.provider_symbol or cfg.symbol,
                                        "interval": cfg.interval, "outputsize": n, "timezone": cfg.tz})
        if not js.get("values"):
            raise ProviderError("data", f"no bars for {cfg.symbol}")
        df = clean_bars(to_frame(js["values"]))
        if df.empty:
            raise ProviderError("data", f"all bars invalid for {cfg.symbol}")
        return df

    def quote(self, cfg: SymbolCfg) -> Quote:
        js = self._get("/price", {"symbol": cfg.provider_symbol or cfg.symbol})
        px = _pos_float(js.get("price"), "price")
        return Quote(cfg.symbol, px, px, px, time.time(), self.name)   # /price has no bid/ask


# ------------------------------------------------------------------ Binance (public market data, no key)
class Binance(_Base):
    name = "binance"
    _ITV = {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "1h": "1h"}

    def __init__(self, p: Params):
        super().__init__(p)
        self.budget = None            # public limit is ~6000 weight/min; our polling is far below

    def _get(self, path: str, params: dict):
        r, js = self._request(BINANCE_BASE + path, params)
        if r.status_code in (418, 429):
            ra = float(r.headers.get("Retry-After", 10) or 10)
            raise ProviderError("rate", "binance rate limit", retry_after=ra)
        if r.status_code >= 400:
            msg = js.get("msg") if isinstance(js, dict) else ""
            raise ProviderError("server", f"HTTP {r.status_code} {redact(msg)}")
        return js

    def bars(self, cfg: SymbolCfg, n: int) -> pd.DataFrame:
        sym = cfg.provider_symbol or binance_symbol(cfg.symbol)
        itv = self._ITV.get(cfg.interval)
        if not itv:
            raise ProviderError("data", f"unsupported interval {cfg.interval}")
        rows = self._get("/api/v3/klines", {"symbol": sym, "interval": itv, "limit": min(n, 1000)})
        if not isinstance(rows, list) or not rows:
            raise ProviderError("data", f"no klines for {sym}")
        df = clean_bars(klines_to_frame(rows))
        if df.empty:
            raise ProviderError("data", f"all klines invalid for {sym}")
        return df

    def quote(self, cfg: SymbolCfg) -> Quote:
        sym = cfg.provider_symbol or binance_symbol(cfg.symbol)
        js = self._get("/api/v3/ticker/bookTicker", {"symbol": sym})
        if not isinstance(js, dict):
            raise ProviderError("data", "unexpected bookTicker reply")
        bid, ask = _pos_float(js.get("bidPrice"), "bid"), _pos_float(js.get("askPrice"), "ask")
        if bid > ask:                                   # crossed book = bad snapshot
            raise ProviderError("data", f"crossed book {bid}>{ask}")
        return Quote(cfg.symbol, bid, ask, (bid + ask) / 2.0, time.time(), self.name)


def make_providers(p: Params, symbols) -> Dict[str, _Base]:
    need = {s.provider for s in symbols} | {s.quote_src for s in symbols}
    out: Dict[str, _Base] = {}
    if "twelvedata" in need:
        out["twelvedata"] = TwelveData(p)
    if "binance" in need:
        out["binance"] = Binance(p)
    return out


# ------------------------------------------------------------------ tick sanity
class QuoteFilter:
    """Rejects a single wild tick. A jump larger than `max_jump_pct` is accepted only when the NEXT
    tick agrees with it (real gaps persist, glitches do not)."""

    def __init__(self, max_jump_pct: float):
        self.max = max_jump_pct
        self.last: Optional[float] = None
        self.last_ts = 0.0
        self.suspect: Optional[float] = None

    def accept(self, q: Quote) -> Optional[Quote]:
        vals = (q.bid, q.ask, q.last)
        if any((not math.isfinite(v)) or v <= 0 for v in vals):
            return None
        if q.bid > q.ask:                               # crossed: use the last trade for both sides
            q = Quote(q.symbol, q.last, q.last, q.last, q.ts, q.src)
        px = q.mid
        if self.last is None or q.ts - self.last_ts > 900:   # first tick / after a long pause (overnight gap)
            self._take(px, q.ts)
            return q
        if abs(px / self.last - 1.0) * 100.0 <= self.max:
            self._take(px, q.ts)
            return q
        if self.suspect is not None and abs(px / self.suspect - 1.0) * 100.0 <= self.max:
            self._take(px, q.ts)                        # confirmed by a second tick
            return q
        self.suspect = px
        return None

    def _take(self, px: float, ts: float) -> None:
        self.last, self.last_ts, self.suspect = px, ts, None
