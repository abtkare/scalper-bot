"""Robustness tests: fast exits, crash recovery, bad data, rejected orders, calendar, rate limits.
Run: python test_robust.py      (no network, no API key; a simulated market + clock drive the real bot loop)
"""
import csv
import json
import logging
import math
import os
import shutil
import tempfile
from concurrent.futures import Future
from datetime import datetime

import numpy as np
import pandas as pd

import bot
import market_calendar as cal
from bot import Runtime, epoch_of, local_naive
from broker import Broker, Fill, PaperBroker
from config import Params, SymbolCfg, validate_config, ConfigError
from data import BarStore, clean_bars, redact
from infra import Notifier, StateStore
from providers import (Binance, ProviderError, Quote, QuoteFilter, RateBudget, TwelveData)

logging.getLogger().setLevel(logging.CRITICAL)
REAL_SIGNALS = bot.signals
PASSED = []


def ok(name):
    PASSED.append(name)
    print(f"  ok  {name}")


# ------------------------------------------------------------------ simulation harness
class Clock:
    def __init__(self, t):
        self.now = float(t)

    def __call__(self):
        return self.now


class SyncExecutor:
    """Runs jobs immediately: deterministic tests, same code path as the thread pool."""
    def submit(self, fn, *a):
        f = Future()
        try:
            f.set_result(fn(*a))
        except BaseException as e:
            f.set_exception(e)
        return f


class Rec(Notifier):
    def __init__(self):
        super().__init__("", "")
        self.msgs = []

    def send(self, msg, key=None, every=0.0, level=logging.INFO):
        self.msgs.append(msg)

    def has(self, text):
        return any(text in m for m in self.msgs)


class Market:
    """Fake data provider driven by price(t). Serves real-looking bars and quotes."""
    name = "fake"
    budget = None
    skew = 0.0

    def __init__(self, clock, price, tz="UTC", spread=0.0, publish_delay=None):
        self.clock, self.price, self.tz, self.spread = clock, price, tz, spread
        self.publish_delay = publish_delay
        self.cache = {}
        self.quote_calls = self.bar_calls = 0
        self.fail_quotes = self.fail_bars = False
        self.quote_override = {}          # {call_number: price}
        self.quote_blackout = None        # (t0, t1)

    def _ohlc(self, s, end):
        ts = np.arange(s, max(end, s + 1), 5.0)
        px = np.array([self.price(t) for t in ts])
        return px[0], px.max(), px.min(), px[-1]

    def bars(self, cfg, n):
        self.bar_calls += 1
        if self.fail_bars:
            raise ProviderError("network", "boom")
        now = self.clock.now
        cur = int(now // 300 * 300)
        rows, idx = [], []
        for s in range(cur - (n - 1) * 300, cur + 300, 300):
            closed = s + 300 <= now
            if self.publish_delay is not None:
                if not closed or s + 300 + self.publish_delay > now:
                    continue
            if closed:
                if s not in self.cache:
                    self.cache[s] = self._ohlc(s, s + 300)
                o, h, l, c = self.cache[s]
            else:
                o, h, l, c = self._ohlc(s, now)
            rows.append((o, h, l, c, 1.0))
            idx.append(local_naive(s, self.tz))
        return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="datetime"),
                            columns=["open", "high", "low", "close", "volume"])

    def quote(self, cfg):
        self.quote_calls += 1
        now = self.clock.now
        if self.fail_quotes or (self.quote_blackout and self.quote_blackout[0] <= now < self.quote_blackout[1]):
            raise ProviderError("network", "boom")
        px = self.quote_override.get(self.quote_calls, self.price(now))
        return Quote(cfg.symbol, px - self.spread / 2, px + self.spread / 2, px, now, "fake")


def fake_signals(bars, side=1, atr=1.0):
    times = {pd.Timestamp(b) for b in bars}

    def f(d, p, s):
        out = d.copy()
        out["atr"] = atr
        out["side"] = 0
        out.loc[out.index.isin(times), "side"] = side
        return out
    return f


T0 = epoch_of(datetime(2026, 1, 5, 10, 0, 0), "UTC")            # a Monday, 10:00:00 UTC
SIG_BAR = local_naive(T0, "UTC")                                 # the 10:00 bar -> entry at 10:05:00


def base(t):
    return 100.0 + 0.05 * math.sin(t / 37.0)


def ramp(start, rate):
    return lambda t: base(t) + max(0.0, t - start) * rate


class Env:
    def __init__(self, price, tz="UTC", cfg=None, p=None, broker=None, spread=0.0, start=T0,
                 publish_delay=None, sig_bars=(SIG_BAR,), sig_side=1, tmp=None, state=None, files=None):
        self.tmp = tmp or tempfile.mkdtemp(prefix="bt_")
        self.clock = Clock(start)
        self.mkt = Market(self.clock, price, tz=tz, spread=spread, publish_delay=publish_delay)
        self.cfg = cfg or SymbolCfg("TEST/USD", "crypto", "UTC", cost_bps=0.0, provider="fake")
        self.p = p or Params(initial_equity=1000, risk_pct=0.01, max_leverage=10, quote_poll_sec=2.0,
                             max_consec_losses=99, max_trades_per_day=99)
        self.n = Rec()
        self.store = state or StateStore(os.path.join(self.tmp, "state.json"))
        self.broker = broker or PaperBroker()
        self.files = files or (os.path.join(self.tmp, "STOP"), os.path.join(self.tmp, "FLATTEN"),
                               os.path.join(self.tmp, "trades.csv"))
        bot.signals = fake_signals(sig_bars, sig_side)
        self.new_runtime()

    def new_runtime(self):
        self.rt = Runtime(self.p, [self.cfg], {"fake": self.mkt}, self.broker, self.store, self.n,
                          SyncExecutor(), clock=self.clock, stop_file=self.files[0],
                          flatten_file=self.files[1], trades_file=self.files[2])
        return self.rt

    @property
    def ss(self):
        return self.rt.syms[0]

    def run(self, seconds, dt=0.2, until=None):
        end = self.clock.now + seconds
        while self.clock.now < end:
            self.rt.step()
            self.clock.now += dt
            if until and until():
                return True
        return until() if until else None

    def trades(self):
        if not os.path.exists(self.files[2]):
            return []
        with open(self.files[2], newline="") as f:
            return list(csv.DictReader(f))


# ------------------------------------------------------------------ 1) fast entry + fast target exit
def test_fast_entry_and_target():
    e = Env(ramp(T0 + 450, 0.1))
    opened = {}

    def watch():
        if e.ss.eng.pos and "t" not in opened:
            opened["t"] = e.clock.now
        return bool(e.trades())
    assert e.run(900, until=watch), "trade never closed"
    pos_open_delay = opened["t"] - (T0 + 300)
    assert 0 <= pos_open_delay <= 3.0, f"entry too slow: {pos_open_delay:.2f}s after the bar open"
    tr = e.trades()[0]
    assert tr["reason"] == "target" and tr["side"] == "long"
    fill = float(tr["entry"])
    assert abs(float(tr["exit"]) - (fill + 1.5)) < 1e-9
    cross = next(t for t in np.arange(T0 + 450, T0 + 900, 0.1) if ramp(T0 + 450, 0.1)(t) >= fill + 1.5)
    exit_delay = e.clock.now - cross
    assert exit_delay <= 3.0, f"target exit too slow: {exit_delay:.2f}s after price crossed"
    assert e.rt.acct.equity > 1000
    ok(f"entry {pos_open_delay:.1f}s after bar open, target exit {exit_delay:.1f}s after price crossed")


# ------------------------------------------------------------------ 2) stop exit (fast, bounded loss)
def test_fast_stop():
    e = Env(ramp(T0 + 450, -0.1))
    assert e.run(900, until=lambda: bool(e.trades()))
    tr = e.trades()[0]
    assert tr["reason"] == "stop"
    r = float(tr["R"])
    assert -1.4 < r <= -0.99, f"loss should be about -1R, got {r}"
    ok(f"stop exit at {r:.2f}R")


# ------------------------------------------------------------------ 3) restart with an open position
def test_restart_no_duplicate():
    e = Env(ramp(T0 + 450, 0.1))
    assert e.run(400, until=lambda: e.ss.eng.pos is not None)
    entry_price = e.ss.eng.pos.entry
    e.clock.now += 20
    e.new_runtime()                                              # "kill -9" + restart
    st = e.store.load()
    e.rt.restore(st)
    assert e.ss.eng.pos is not None and abs(e.ss.eng.pos.entry - entry_price) < 1e-12
    assert e.run(900, until=lambda: bool(e.trades()))
    tr = e.trades()
    assert len(tr) == 1 and tr[0]["reason"] == "target", tr
    e.run(400)                                                   # keep running: the signal bar must not re-fire
    assert len(e.trades()) == 1 and e.rt.acct.trades_today == 1
    ok("restart mid-position: state restored, exactly one trade, no duplicate entry")


# ------------------------------------------------------------------ 4) live-price blackout + bar backstop
def test_blackout_backstop():
    e = Env(ramp(T0 + 450, -0.1))
    e.mkt.quote_blackout = (T0 + 440, T0 + 700)                  # price feed dies while the stop is crossed
    assert e.run(900, until=lambda: bool(e.trades()))
    tr = e.trades()[0]
    assert tr["reason"] == "stop_bar", tr["reason"]
    assert abs(float(tr["exit"]) - (float(tr["entry"]) - 1.0)) < 1e-9
    assert e.n.has("NO LIVE PRICE")
    ok("feed blackout: alert raised, bar backstop closed the position at the stop")


# ------------------------------------------------------------------ 5) broker rejects exits -> position kept, retried
class FlakyBroker(PaperBroker):
    def __init__(self, fail_exits=3):
        self.fail_exits, self.exit_calls = fail_exits, 0

    def market_exit(self, *a, **k):
        self.exit_calls += 1
        if self.exit_calls <= self.fail_exits:
            return Fill(False, reason="simulated reject")
        return super().market_exit(*a, **k)


def test_rejected_exit_retried():
    b = FlakyBroker(3)
    e = Env(ramp(T0 + 450, 0.1), broker=b)
    assert e.run(900, until=lambda: bool(e.trades()))
    assert b.exit_calls == 4 and e.n.has("EXIT REJECTED")
    assert e.trades()[0]["reason"] == "target"
    ok("broker rejected 3 exits: position kept, retried, closed on the 4th try")


class ExplodingBroker(PaperBroker):
    def market_exit(self, *a, **k):
        raise RuntimeError("adapter bug")


def test_broker_exception_never_loses_position():
    e = Env(ramp(T0 + 450, 0.1), broker=ExplodingBroker())
    e.run(700)
    assert e.ss.eng.pos is not None and not e.trades()
    assert e.n.has("EXIT REJECTED")
    ok("broker exception: position still tracked (never silently dropped)")


# ------------------------------------------------------------------ 6) bad tick cannot trigger a stop
def test_spike_tick_ignored():
    e = Env(lambda t: base(t))
    assert e.run(400, until=lambda: e.ss.eng.pos is not None)
    calls = e.mkt.quote_calls
    e.mkt.quote_override = {calls + 3: 60.0}                     # ONE bad print, -40%
    e.run(30)
    assert e.ss.eng.pos is not None and not e.trades(), "a single bad tick closed the position"
    assert e.n.has("sanity filter")
    ok("single -40% bad tick rejected by the sanity filter (no false stop)")


# ------------------------------------------------------------------ 7) feed errors never crash / backoff
def test_feed_errors_backoff():
    e = Env(lambda t: base(t))
    e.mkt.fail_bars = True
    e.run(120)
    assert e.mkt.bar_calls < 60, f"no backoff: {e.mkt.bar_calls} calls in 120s"
    assert e.ss.errs["bar"] > 0 and e.n.has("bar error")
    e.mkt.fail_bars = False
    e.run(40)
    assert e.ss.errs["bar"] == 0 and e.ss.last_eval is not None
    ok(f"bar-feed outage: {e.mkt.bar_calls} calls (backoff), recovered automatically")


# ------------------------------------------------------------------ 8) late data / slippage guards
def test_late_bar_skipped():
    e = Env(lambda t: base(t), publish_delay=40.0)              # provider publishes the bar 40s late
    e.run(400)
    assert e.ss.eng.pos is None and e.n.has("too late")
    ok("bar published 40s late: entry skipped instead of chasing")


def test_slippage_guard():
    jump = lambda t: base(t) + (1.0 if t >= T0 + 300 else 0.0)  # +1.0 (=1 ATR) gap exactly at the open
    e = Env(jump)
    e.run(400)
    assert e.ss.eng.pos is None and e.n.has("entry skipped")
    ok("price gapped 1 ATR at the open: entry skipped")


# ------------------------------------------------------------------ 9) STOP / FLATTEN files
def test_stop_and_flatten_files():
    e = Env(lambda t: base(t))
    open(e.files[0], "w").close()
    e.run(400)
    assert e.ss.eng.pos is None
    os.remove(e.files[0])
    e2 = Env(lambda t: base(t))
    assert e2.run(400, until=lambda: e2.ss.eng.pos is not None)
    open(e2.files[1], "w").close()
    assert e2.run(20, until=lambda: bool(e2.trades()))
    assert e2.trades()[0]["reason"] == "manual_flatten"
    ok("STOP blocks entries; FLATTEN closes the open position")


# ------------------------------------------------------------------ 10) calendar
def test_calendar():
    d = lambda y, m, dd: datetime(y, m, dd).date()
    assert cal.session(d(2026, 12, 25)) is None and cal.session(d(2026, 7, 3)) is None
    assert cal.session(d(2026, 1, 3)) is None                            # Saturday
    s = cal.session(d(2026, 11, 27))
    assert s[1].hour == 13, "early close"
    assert cal.flatten_time(d(2026, 11, 27), 5) == datetime(2026, 11, 27, 12, 55)
    assert cal.flatten_time(d(2026, 9, 21), 5) == datetime(2026, 9, 21, 15, 55)
    assert cal.covers(d(2026, 9, 21)) and not cal.covers(d(2028, 1, 3))
    ok("NYSE calendar: holidays, weekend, early close 13:00 -> flatten 12:55")


STOCK = dict(kind="stock", tz="America/New_York", cost_bps=0.0, flatten_eod=True, provider="fake",
             windows=[("09:35", "11:00"), ("15:00", "15:50")])


def _stock_env(day, hh, mm, price=lambda t: 100.0):
    cfg = SymbolCfg("QQQ", **STOCK)
    start = epoch_of(datetime(*day, hh, mm, 0), "America/New_York")
    return Env(price, tz="America/New_York", cfg=cfg, start=start, sig_bars=[])


def _inject(e, entry_dt, side=1, entry=100.0):
    from engine import Position
    e.ss.eng.pos = Position(side=side, qty=10, entry=entry, stop=entry - 5, target=entry + 8,
                            entry_ts=pd.Timestamp(entry_dt), risk_amt=50.0, opened_epoch=e.clock.now - 60)


def test_early_close_eod_flatten():
    e = _stock_env((2026, 11, 27), 12, 50)                               # early-close day
    _inject(e, datetime(2026, 11, 27, 10, 0))
    assert e.run(600, until=lambda: bool(e.trades()))
    tr = e.trades()[0]
    hhmm = local_naive(e.clock.now, "America/New_York").strftime("%H:%M")
    assert tr["reason"] == "eod" and hhmm in ("12:55", "12:56"), (tr["reason"], hhmm)
    ok("early-close day (13:00): stock position flattened at 12:55, not carried overnight")


def test_overnight_flatten_at_open():
    # (a) bar data unavailable: the live-price rule closes the stale position at the first price after the open
    e = _stock_env((2026, 1, 6), 9, 20)
    e.mkt.fail_bars = True
    _inject(e, datetime(2026, 1, 5, 15, 50))                             # left over from yesterday
    e.run(900, until=lambda: bool(e.trades()))
    tr = e.trades()[0]
    lt = local_naive(e.clock.now, "America/New_York")
    assert tr["reason"] == "overnight" and lt.strftime("%H:%M") >= "09:30", (tr["reason"], lt)
    # (b) bar data available: catch-up replays the missed 15:55 bar and flattens there (as the bot would have)
    e = _stock_env((2026, 1, 6), 9, 20)
    _inject(e, datetime(2026, 1, 5, 15, 50))
    e.run(600, until=lambda: bool(e.trades()))
    assert e.trades()[0]["reason"] == "eod" and e.trades()[0]["exit_ts"].startswith("2026-01-05 15:50")   # the bar that ENDS at 15:55
    ok("stale overnight stock position: closed at first price after the open (or replayed at yesterday's bar ending 15:55)")


def test_holiday_no_calls():
    e = _stock_env((2026, 12, 25), 10, 0)
    e.run(300)
    assert e.mkt.bar_calls == 0 and e.mkt.quote_calls == 0
    ok("market holiday: zero API calls (no wasted credits)")


# ------------------------------------------------------------------ 11) state store
def test_state_store():
    d = tempfile.mkdtemp()
    s = StateStore(os.path.join(d, "state.json"))
    s.save({"equity": 1.0}); s.save({"equity": 2.0})
    assert s.load()["equity"] == 2.0
    open(s.path, "w").write("{ broken json")                            # corruption
    s2 = StateStore(s.path)
    assert s2.load()["equity"] == 1.0 and s2.notes                       # falls back to the backup
    assert any(f.startswith("state.json.corrupt-") for f in os.listdir(d))
    open(s2.path, "w").write("garbage"); open(s2.bak, "w").write("garbage")
    s3 = StateStore(s.path)
    assert s3.load() is None and len(s3.notes) == 2
    ok("state file: atomic save, corrupt file quarantined, backup restored")


def test_legacy_state_v1():
    e = Env(lambda t: base(t))
    old = {"equity": 1234.5, "day": "2026-01-05", "day_start_equity": 1200.0, "trades_today": 2,
           "consec_losses": 1, "last_eval": {"TEST/USD": "2026-01-05T09:55:00"},
           "positions": {"TEST/USD": {"side": 1, "qty": 3.0, "entry": 100.0, "stop": 99.0, "target": 101.5,
                                      "entry_ts": "2026-01-05T09:55:00", "risk_amt": 3.0}}}
    e.rt.restore(old)
    assert e.ss.eng.pos.qty == 3.0 and abs(e.rt.acct.equity - 1234.5) < 1e-9
    ok("state written by the previous bot version still loads")


# ------------------------------------------------------------------ 12) data validation
def test_clean_bars():
    idx = pd.date_range("2026-01-05", periods=6, freq="5min")
    df = pd.DataFrame({"open": [1, 1, 1, np.nan, 1, -1], "high": [2, 0.5, 2, 2, 2, 2],
                       "low": [0.5, 0.9, 0.5, 0.5, 0.5, 0.5], "close": [1.5, 1, 1.5, 1.5, 3.0, 1],
                       "volume": 1.0}, index=idx)
    out = clean_bars(df)
    assert list(out.index) == [idx[0], idx[2]], list(out.index)          # bad high, NaN, close>high, negative dropped
    dup = pd.concat([out, out])
    assert len(clean_bars(dup)) == 2
    ok("invalid bars (NaN, high<low, close>high, <=0, duplicates) are dropped")


def test_barstore():
    s = BarStore(5, keep=300)
    idx = pd.date_range("2026-01-05 09:00", periods=300, freq="5min")
    mk = lambda ix: pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=ix)
    assert s.merge(mk(idx)) == "ok"
    assert s.merge(mk(idx[-3:])) == "ok" and len(s.df) == 300            # overlap
    nxt = pd.date_range(idx[-1] + pd.Timedelta(minutes=5), periods=2, freq="5min")
    assert s.merge(mk(nxt)) == "ok" and s.df.index[-1] == nxt[-1]
    hole = pd.date_range(nxt[-1] + pd.Timedelta(minutes=30), periods=2, freq="5min")
    assert s.merge(mk(hole)) == "gap"                                    # missed bars -> caller downloads again
    now = nxt[-1] + pd.Timedelta(minutes=2)                              # last bar still forming
    assert s.closed(now, 1.0).index[-1] == nxt[-2]
    now = nxt[-1] + pd.Timedelta(minutes=5, seconds=2)                   # closed + settled
    assert s.closed(now, 1.0).index[-1] == nxt[-1]
    ok("bar cache: overlap merge, gap detection, forming-bar exclusion")


def test_quote_filter():
    f = QuoteFilter(3.0)
    q = lambda px, ts: Quote("X", px, px, px, ts)
    assert f.accept(q(100, 0)) and f.accept(q(101, 2))
    assert f.accept(q(90, 4)) is None and f.accept(q(101.5, 6))          # lone spike rejected
    assert f.accept(q(90, 8)) is None and f.accept(q(89.5, 10))          # persistent gap confirmed by 2nd tick
    assert f.accept(Quote("X", float("nan"), 1, 1, 12)) is None
    assert f.accept(Quote("X", -1, -1, -1, 12)) is None
    ok("tick filter: lone spikes rejected, real gaps accepted after confirmation, NaN/negative rejected")


# ------------------------------------------------------------------ 13) rate budget + providers
def test_rate_budget():
    t = [0.0]
    b = RateBudget(8, clock=lambda: t[0])
    hits = []
    while t[0] < 300:
        if b.try_acquire(1):
            hits.append(t[0])
        t[0] += 0.05
    worst = max(sum(1 for h in hits if x <= h < x + 60) for x in np.arange(0, 240, 1.0))
    assert worst <= 8, f"{worst} calls in one minute exceeds the plan (8)"
    b.penalize(15)
    assert not b.try_acquire(1)
    t[0] += 16
    ok(f"credit budget: never more than {worst}/8 calls in any 60 s window; 429 penalty blocks calls")


class _Resp:
    def __init__(self, status=200, headers=None):
        self.status_code, self.headers = status, headers or {}


def _td(js, status=200):
    td = TwelveData(Params(), api_key="SECRETKEY123")
    td._request = lambda url, params: (_Resp(status), js)
    return td


def test_providers_parse():
    cfg = SymbolCfg("BTC/USD", "crypto", "UTC")
    assert _td({"price": "70123.45"}).quote(cfg).last == 70123.45
    for bad in ({"price": "nan"}, {"price": "0"}, {"price": None}, {}):
        try:
            _td(bad).quote(cfg); raise AssertionError(f"accepted {bad}")
        except ProviderError as e:
            assert e.kind == "data"
    td = _td({"status": "error", "code": 429, "message": "out of credits apikey=SECRETKEY123"})
    try:
        td.quote(cfg); raise AssertionError
    except ProviderError as e:
        assert e.kind == "rate" and "SECRETKEY123" not in str(e) and not td.budget.try_acquire(1)
    try:
        _td({"status": "error", "code": 401, "message": "bad key"}).quote(cfg); raise AssertionError
    except ProviderError as e:
        assert e.kind == "auth"
    vals = [{"datetime": "2026-01-05 10:00:00", "open": "1", "high": "2", "low": "0.5", "close": "1.5", "volume": "9"},
            {"datetime": "2026-01-05 10:05:00", "open": "x", "high": "2", "low": "0.5", "close": "1.5"}]
    assert len(_td({"values": vals}).bars(cfg, 5)) == 1                  # the corrupt row is dropped
    bn = Binance(Params())
    bn._request = lambda url, params: (_Resp(200), {"bidPrice": "100.0", "askPrice": "100.5"})
    q = bn.quote(cfg)
    assert q.bid == 100.0 and q.ask == 100.5
    bn._request = lambda url, params: (_Resp(200), {"bidPrice": "101", "askPrice": "100"})
    try:
        bn.quote(cfg); raise AssertionError
    except ProviderError as e:
        assert e.kind == "data"
    rows = [[1767607200000, "1", "2", "0.5", "1.5", "9", 0], [1767607500000, "1.5", "2", "1", "1.7", "9", 0]]
    bn._request = lambda url, params: (_Resp(200), rows)
    assert len(bn.bars(cfg, 5)) == 2
    assert "SECRET" not in redact("https://x/?symbol=A&apikey=SECRET&z=1") and "123:ABC" not in redact("bot123:ABC-d/x")
    ok("providers: parsing, error classes, bad values rejected, API key never appears in errors")


def test_config_validation():
    p = Params()
    try:
        validate_config(Params(risk_pct=0.5), [SymbolCfg("X", "crypto", "UTC", provider="fake")]); raise AssertionError
    except ConfigError:
        pass
    try:
        validate_config(p, [SymbolCfg("X", "crypto", "Mars/Base")]); raise AssertionError
    except ConfigError:
        pass
    ok("config validation refuses dangerous / malformed settings")


def test_backtest_still_works():
    from backtest import run_backtest, summarize
    rng = np.random.default_rng(3)
    n = 4000
    idx = pd.date_range("2026-01-05", periods=n, freq="5min")
    c = 100 + np.cumsum(rng.normal(0, 0.15, n))
    df = pd.DataFrame({"open": c - 0.01, "high": c + 0.2, "low": c - 0.2, "close": c, "volume": 1.0}, index=idx)
    bot.signals = REAL_SIGNALS
    trades, acct = run_backtest(df, SymbolCfg("T/USD", "crypto", "UTC", cost_bps=1.0), Params(), 1000.0)
    s = summarize(trades, 1000.0)
    assert s["trades"] > 0 and math.isfinite(acct.equity)
    ok(f"backtest engine path unchanged ({s['trades']} trades on synthetic data)")


# ------------------------------------------------------------------ guardian inside the real loop
def test_guardian_shock_exit_in_loop():
    p = Params(initial_equity=1000, risk_pct=0.01, max_leverage=10, quote_poll_sec=2.0,
               max_consec_losses=99, max_trades_per_day=99, stop_atr=6.0, target_atr=9.0)
    e = Env(ramp(T0 + 450, -0.06), p=p)          # slides 3.6 ATR in 60 s: a crash, stop is 6 ATR away
    e.ss.atr = 1.0
    assert e.run(900, until=lambda: bool(e.trades()))
    tr = e.trades()[0]
    assert tr["reason"] == "guardian_shock", tr["reason"]
    assert float(tr["R"]) > -0.7, tr["R"]        # left far before the 6-ATR stop (-1R)
    assert e.rt.guard.entry_block("TEST/USD") is not None
    ok(f"guardian exited a crashing position at {float(tr['R']):.2f}R (stop would be -1R) + cool-down")


def test_guardian_halt_blocks_entries_in_loop():
    e = Env(ramp(T0 + 450, 0.1))
    e.rt.guard.halted_reason = "test"
    open(e.rt.guard.halt_file, "w").close()
    e.run(900)
    assert not e.trades() and e.ss.eng.pos is None
    assert e.n.has("blocked by guardian")
    ok("guardian HALT blocks new entries in the live loop")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nall {len(PASSED)} robustness checks passed")
