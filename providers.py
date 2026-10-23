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
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import date, timedelta
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


# ------------------------------------------------------------------ persistent daily quota
def utc_day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def next_utc_midnight(ts: float) -> float:
    return (int(ts) // 86400 + 1) * 86400.0


def quota_health(count: float, cap: int) -> str:
    """Alert level only (monitoring) - it never changes what try_acquire allows.
    ok < 90% <= warn < 95% <= critical < 100% <= blocked   (800/day: 720 / 760 / 800)"""
    if not cap:
        return "ok"
    if count >= cap:
        return "blocked"
    if count * 100 >= cap * 95:
        return "critical"
    if count * 100 >= cap * 90:
        return "warn"
    return "ok"


class QuotaStore:
    """Daily credit counter in SQLite, so a restart never forgets what was already spent today.
    One row per (provider, UTC day); rows older than KEEP_DAYS days are deleted. Holds NO API keys.
    Any SQLite failure -> the caller keeps counting in RAM and the store tries again on the first call
    after RETRY_SEC (a passing error never degrades the process for good)."""
    KEEP_DAYS = 7
    RETRY_SEC = 60.0
    SCHEMA = ("CREATE TABLE IF NOT EXISTS api_quota (provider TEXT NOT NULL, day_utc TEXT NOT NULL, "
              "count INTEGER NOT NULL DEFAULT 0, last_hit TEXT NOT NULL, PRIMARY KEY (provider, day_utc))",
              "CREATE TABLE IF NOT EXISTS api_quota_alerts (provider TEXT NOT NULL, day_utc TEXT NOT NULL, "
              "level TEXT NOT NULL, at TEXT NOT NULL, PRIMARY KEY (provider, day_utc, level))")

    def __init__(self, path: str, provider: str = "twelvedata", clock=time.time):
        self.path, self.provider, self.clock = os.path.abspath(path), provider, clock
        self.conn: Optional[sqlite3.Connection] = None
        self.journal = "-"
        self.healthy = False
        self.retry_at = 0.0
        self.failures = 0
        self.last_error = ""
        self.lock = threading.Lock()
        with self.lock:
            self._connect()

    def _connect(self) -> bool:
        try:
            conn = sqlite3.connect(self.path, timeout=3.0, isolation_level=None, check_same_thread=False)
            try:
                conn.execute("PRAGMA busy_timeout=3000")
                try:
                    mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
                except sqlite3.Error:
                    mode = ""
                if mode != "wal":                                # file system without WAL support
                    mode = str(conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower()
                    log.warning("quota db: WAL not supported here - using journal_mode=%s", mode)
                conn.execute("PRAGMA synchronous=FULL")
                for sql in self.SCHEMA:
                    conn.execute(sql)
            except Exception:
                conn.close()
                raise
        except (sqlite3.Error, OSError) as e:
            self._fail(e)
            return False
        self.conn, self.journal = conn, mode
        self._ok()
        return True

    def _ok(self) -> None:
        if not self.healthy and self.failures:
            log.info("quota db recovered (%s)", self.path)
        self.healthy = True

    def _fail(self, e: Exception) -> None:
        self.healthy = False
        self.failures += 1
        self.last_error = f"{type(e).__name__}: {e}"[:160]
        self.retry_at = self.clock() + self.RETRY_SEC
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
        log.warning("quota db failed (%s) - counting in RAM, retry in %.0f s", self.last_error, self.RETRY_SEC)

    def _run(self, fn, write: bool):
        """-> (True, result) or (False, None). Writes are one BEGIN IMMEDIATE ... COMMIT transaction."""
        with self.lock:
            if self.conn is None and (self.clock() < self.retry_at or not self._connect()):
                return False, None
            try:
                if write:
                    self.conn.execute("BEGIN IMMEDIATE")
                res = fn(self.conn)
                if write:
                    self.conn.execute("COMMIT")
                self._ok()
                return True, res
            except (sqlite3.Error, OSError) as e:
                try:
                    if write and self.conn is not None:
                        self.conn.execute("ROLLBACK")
                except Exception:
                    pass
                self._fail(e)
                return False, None

    def _now_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.clock()))

    def get(self, day: str) -> Optional[int]:
        ok, r = self._run(lambda c: c.execute("SELECT count FROM api_quota WHERE provider=? AND day_utc=?",
                                              (self.provider, day)).fetchone(), write=False)
        return (int(r[0]) if r else 0) if ok else None

    def add(self, day: str, n: float = 1, floor: float = 0) -> Optional[int]:
        """count = max(count + n, floor) atomically; -> new count (None = SQLite unavailable).
        `floor` lets the RAM counter catch the file up after an outage, so the file is never behind."""
        def tx(c):
            c.execute("INSERT OR IGNORE INTO api_quota (provider, day_utc, count, last_hit) VALUES (?, ?, 0, ?)",
                      (self.provider, day, self._now_iso()))
            c.execute("UPDATE api_quota SET count = MAX(count + ?, ?), last_hit = ? WHERE provider=? AND day_utc=?",
                      (int(math.ceil(n)), int(math.ceil(floor)), self._now_iso(), self.provider, day))
            return c.execute("SELECT count FROM api_quota WHERE provider=? AND day_utc=?",
                             (self.provider, day)).fetchone()[0]
        ok, r = self._run(tx, write=True)
        return int(r) if ok else None

    def migrate_from_memory(self, day: str, mem_count: float) -> Optional[int]:
        """First start after the upgrade: seed the file with the RAM counter (never lowers a saved count)."""
        return self.add(day, 0, floor=mem_count)

    def cleanup(self, today: str) -> Optional[int]:
        """Delete rows older than KEEP_DAYS days (today included in the 7). -> rows deleted."""
        cutoff = (date.fromisoformat(today) - timedelta(days=self.KEEP_DAYS - 1)).isoformat()

        def tx(c):
            n = c.execute("DELETE FROM api_quota WHERE day_utc < ?", (cutoff,)).rowcount
            c.execute("DELETE FROM api_quota_alerts WHERE day_utc < ?", (cutoff,))
            return n
        ok, r = self._run(tx, write=True)
        return r if ok else None

    def mark_alert(self, day: str, level: str) -> Optional[bool]:
        """True the first time (provider, day, level) is seen - across restarts. None = SQLite unavailable."""
        ok, r = self._run(lambda c: c.execute(
            "INSERT OR IGNORE INTO api_quota_alerts (provider, day_utc, level, at) VALUES (?, ?, ?, ?)",
            (self.provider, day, level, self._now_iso())).rowcount, write=True)
        return bool(r) if ok else None

    def close(self) -> None:
        with self.lock:
            if self.conn is not None:
                self.conn.close()
                self.conn = None


# ------------------------------------------------------------------ rate budget
class RateBudget:
    """Token bucket. Sustained rate = 75% of the plan limit and a small burst, so that even with
    the provider's calendar-minute counting the plan limit is never exceeded.

    Optional daily cap (Twelve Data free plan: 800/day) with priority tiers, so that when credits run low
    the bot stops SCANNING first and keeps the credits for managing open positions:
        quote (open position)  -> may use the whole day
        bars  (auto-traded)    -> up to cap - 60
        scan / qual (watch-only streams, qualification) -> up to 70% of the cap

    With a `store` (QuotaStore) the daily count survives restarts: it is loaded at the start of each UTC day
    and every granted credit is written BEFORE the request is sent. The allow/deny decision is unchanged."""
    TIER_FRAC = {"quote": 1.0, "bars": None, "scan": 0.70, "qual": 0.70}

    def __init__(self, per_min: float, clock=time.time, per_day: int = 0, store: Optional[QuotaStore] = None):
        self.per_day = int(per_day or 0)
        self.used_day = 0
        self.day = None
        self.clock = clock
        self.rate = max(per_min * 0.75, 1.0) / 60.0
        self.cap = max(2.0, per_min * 0.25)
        self.tokens = self.cap
        self.t = clock()
        self.blocked_until = 0.0
        self.lock = threading.Lock()
        self.store = store
        self._alerted: set = set()
        if store is not None:
            with self.lock:
                self._roll(utc_day(clock()))

    def _roll(self, day: str) -> None:
        """New UTC day (or first use): start from the saved count of that day, 0 if none."""
        self.day, self.used_day = day, 0
        if self.store is not None:
            saved = self.store.get(day)
            if saved:
                self.used_day = saved
            self.store.cleanup(day)

    def _day_limit(self, tier: str) -> float:
        if not self.per_day:
            return float("inf")
        frac = self.TIER_FRAC.get(tier, 1.0)
        return self.per_day - 60 if frac is None else self.per_day * frac

    def try_acquire(self, n: float = 1.0, tier: str = "quote") -> bool:
        with self.lock:
            now = self.clock()
            day = time.strftime("%Y-%m-%d", time.gmtime(now))
            if day != self.day:
                self._roll(day)
            self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate)
            self.t = now
            if now < self.blocked_until or self.tokens < n:
                return False
            if self.used_day + n > self._day_limit(tier):
                return False
            self.tokens -= n
            self.used_day += n
            if self.store is not None:                   # persist BEFORE the caller sends the request
                saved = self.store.add(day, n, floor=self.used_day)
                if saved is not None and saved > self.used_day:
                    self.used_day = saved                # file was ahead (RAM lost a load) -> trust it
            return True

    def quota_status(self) -> dict:
        """Daily quota for diagnose / dashboard / alerts. `used`/`cap` keep their old meaning."""
        with self.lock:
            now = self.clock()
            if utc_day(now) != self.day:
                self._roll(utc_day(now))
            used, cap, st = int(math.ceil(self.used_day)), self.per_day, self.store
            out = {"provider": st.provider if st else "", "used": used, "cap": cap,
                   "soft": int(self._day_limit("scan")) if cap else 0, "hard": int(self._day_limit("bars")) if cap else 0,
                   "remaining": max(0, cap - used) if cap else None, "day": self.day,
                   "reset_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(next_utc_midnight(now))),
                   "health": quota_health(used, cap), "source": "sqlite" if st and st.healthy else "memory",
                   "file": st.path if st else None, "journal": st.journal if st else None}
            if st and not st.healthy:
                out["error"] = st.last_error
            return out

    def alert_once(self, level: str) -> bool:
        """True only the first time `level` is reached on this UTC day (remembered in the store across restarts)."""
        with self.lock:
            key = (self.day, level)
            if key in self._alerted:
                return False
            self._alerted.add(key)
            if self.store is not None and self.store.mark_alert(self.day, level) is False:
                return False                             # already sent before a restart today
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

    def __init__(self, p: Params, api_key: Optional[str] = None, budget: Optional[RateBudget] = None,
                 quota_path: Optional[str] = None):
        super().__init__(p)
        self.key = api_key or os.getenv("TWELVE_DATA_API_KEY", "")
        store = QuotaStore(quota_path, self.name) if (quota_path and budget is None) else None
        self.budget = budget if budget is not None else RateBudget(p.td_credits_per_min,
                                                                    per_day=p.td_credits_per_day, store=store)

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
        sym = binance_symbol(cfg.ticker)
        itv = self._ITV.get(cfg.interval)
        if not itv:
            raise ProviderError("data", f"unsupported interval {cfg.interval}")
        rows = self._get("/api/v3/klines", {"symbol": sym, "interval": itv, "limit": min(n, 1000)})
        if not isinstance(rows, list) or not rows:
            raise ProviderError("data", f"no klines for {sym}")
        while len(rows) < n:                            # long history (qualification): page backwards
            more = self._get("/api/v3/klines", {"symbol": sym, "interval": itv, "endTime": int(rows[0][0]) - 1,
                                                "limit": min(n - len(rows), 1000)})
            if not isinstance(more, list) or not more:
                break
            rows = more + rows
        df = clean_bars(klines_to_frame(rows))
        if df.empty:
            raise ProviderError("data", f"all klines invalid for {sym}")
        return df

    def quote(self, cfg: SymbolCfg) -> Quote:
        sym = binance_symbol(cfg.ticker)
        js = self._get("/api/v3/ticker/bookTicker", {"symbol": sym})
        if not isinstance(js, dict):
            raise ProviderError("data", "unexpected bookTicker reply")
        bid, ask = _pos_float(js.get("bidPrice"), "bid"), _pos_float(js.get("askPrice"), "ask")
        if bid > ask:                                   # crossed book = bad snapshot
            raise ProviderError("data", f"crossed book {bid}>{ask}")
        return Quote(cfg.symbol, bid, ask, (bid + ask) / 2.0, time.time(), self.name)


def make_providers(p: Params, symbols, quota_path: Optional[str] = None) -> Dict[str, _Base]:
    """quota_path: SQLite file for the persistent daily Twelve Data counter (None = RAM only, e.g. tests)."""
    need = {s.provider for s in symbols} | {s.quote_src for s in symbols}
    out: Dict[str, _Base] = {}
    if "twelvedata" in need:
        out["twelvedata"] = TwelveData(p, quota_path=quota_path)
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
