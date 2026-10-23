"""Gold / FX tests (v5): trading hours, closed-market filler bars, USD/JPY P&L in dollars, daily flat at 16:45 NY,
RSI(2) signal rule, no look-ahead, live/backtest parity, regression vs RESEARCH.md.
Run: python test_fx.py      (no network, no API key)
"""
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

import bot
import fx_hours
import test_robust as tr
from backtest import load_history, run_backtest, summarize
from bot import epoch_of
from config import FX_SET, GOLD, USDJPY, ConfigError, Params, SymbolCfg, validate_config
from strategy import add_indicators, signals

logging.getLogger().setLevel(logging.CRITICAL)
PASSED = []
NY = "America/New_York"


def ok(name):
    PASSED.append(name)
    print(f"  ok  {name}")


def fx_cfg(sym="EUR/USD", **kw):
    base = dict(cost_bps=0.0, provider="fake", strategy="rsi2", allow_short=True, stop_atr=1.0, target_atr=1.5,
                interval="5min", interval_minutes=5, flatten_eod=True, eod_time="16:45")
    base.update(kw)
    return SymbolCfg(sym, "fx", NY, **base)


def env_at(ny_dt, cfg, price, side=1):
    t0 = epoch_of(ny_dt, NY)
    sig = pd.Timestamp(ny_dt)
    return tr.Env(price, tz=NY, cfg=cfg, start=t0, sig_bars=(sig,), sig_side=side)


MON_10 = datetime(2026, 10, 19, 10, 0)          # Monday 10:00 New York


# ------------------------------------------------------------------ 1) hours
def test_hours():
    o = fx_hours.is_open
    assert o(datetime(2026, 10, 19, 10, 0), True) and o(datetime(2026, 10, 19, 3, 0), False)
    assert not o(datetime(2026, 10, 19, 17, 30), True), "gold daily break 17-18"
    assert o(datetime(2026, 10, 19, 17, 30), False), "FX trades through 17-18"
    assert not o(datetime(2026, 10, 23, 17, 0), False), "Friday 17:00 close"
    assert not o(datetime(2026, 10, 24, 12, 0), False), "Saturday"
    assert not o(datetime(2026, 10, 25, 16, 59), False) and o(datetime(2026, 10, 25, 17, 0), False)
    assert not o(datetime(2026, 10, 25, 17, 30), True) and o(datetime(2026, 10, 25, 18, 0), True)
    assert not o(datetime(2026, 12, 25, 10, 0), False)
    assert fx_hours.next_open(datetime(2026, 10, 24, 12, 0), True) == datetime(2026, 10, 25, 18, 0)
    assert GOLD.metal and not USDJPY.metal
    ok("trading hours: Sun 17:00/18:00 -> Fri 17:00, gold break 17-18, holidays closed")


# ------------------------------------------------------------------ 2) USD/JPY P&L converted to USD
def test_usdjpy_pnl_in_usd():
    cfg = fx_cfg("USD/JPY")
    e = env_at(MON_10, cfg, tr.ramp(epoch_of(MON_10, NY) + 450, 0.05), side=1)
    assert e.run(900, until=lambda: bool(e.trades()))
    t = e.trades()[0]
    assert t["reason"] == "target", t
    gain = e.rt.acct.equity - 1000
    assert 5 < gain < 20, f"P&L must be in USD (~$15 for +1.5R at 1% risk), got {gain:.2f}"
    assert 1.4 < float(t["R"]) < 1.55, t["R"]
    ok(f"USD/JPY profit booked in dollars: +${gain:.2f} = {float(t['R']):+.2f}R (not yen)")


def test_xxxusd_and_short():
    e = env_at(MON_10, fx_cfg("XAU/USD"), tr.ramp(epoch_of(MON_10, NY) + 450, -0.05), side=-1)
    assert e.run(900, until=lambda: bool(e.trades()))
    t = e.trades()[0]
    assert t["side"] == "short" and t["reason"] == "target" and abs(float(t["R"]) - 1.5) < 0.02, t
    ok(f"gold SHORT -> target at {float(t['R']):+.2f}R")


# ------------------------------------------------------------------ 3) daily flat at 16:45 New York
def test_daily_flat():
    start = datetime(2026, 10, 19, 16, 30)
    e = env_at(start, fx_cfg(), tr.base, side=1)
    assert e.run(1500, until=lambda: bool(e.trades())), "position was not flattened"
    t = e.trades()[0]
    assert t["reason"] == "eod", t
    closed_at = bot.local_naive(e.clock.now, NY)
    assert closed_at.strftime("%H:%M") in ("16:45", "16:46"), closed_at
    ok(f"open position closed by the daily flat at {closed_at:%H:%M} New York")


def test_no_entry_after_cutoff():
    start = datetime(2026, 10, 19, 16, 40)                       # entry bar 16:45 -> after the cutoff
    e = env_at(start, fx_cfg(), tr.ramp(epoch_of(start, NY) + 450, 0.05), side=1)
    e.run(900)
    assert e.ss.eng.pos is None and not e.trades()
    ok("no new entries after 16:42 New York")


# ------------------------------------------------------------------ 4) closed market: no polling, filler bars dropped
def test_weekend_idle():
    sat = datetime(2026, 10, 24, 12, 0)
    e = env_at(sat, fx_cfg(), tr.base)
    e.run(600)
    assert e.mkt.bar_calls == 0 and e.mkt.quote_calls == 0
    ok("weekend: zero data requests (credits saved)")


def test_filler_bars_dropped():
    e = env_at(MON_10, fx_cfg(), tr.base)
    idx = pd.date_range("2026-10-23 15:00", "2026-10-25 18:00", freq="5min")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0.0}, index=idx)
    e.rt._apply_bars(e.ss, df, e.clock.now)
    kept = e.ss.store.df.index
    assert len(kept) > 0 and all(fx_hours.is_open(t.to_pydatetime(), False) for t in kept)
    assert not any(t.weekday() == 5 for t in kept)
    ok(f"closed-market filler bars removed ({len(idx) - len(kept)} of {len(idx)} dropped)")


def test_gold_break_no_polling_storm():
    cfg = fx_cfg("XAU/USD", interval="1h", interval_minutes=60)
    t = datetime(2026, 10, 19, 17, 10)                           # inside gold's daily break
    e = env_at(t, cfg, tr.base)
    e.ss.last_eval = pd.Timestamp("2026-10-19 16:00")
    e.ss.needs_full = False
    e.ss.store.df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [0.0]},
                                 index=pd.DatetimeIndex([pd.Timestamp("2026-10-19 16:00")]))
    e.rt._schedule_bars(e.ss, e.clock.now)
    assert e.mkt.bar_calls == 0, "polled during the break"
    e.clock.now = epoch_of(datetime(2026, 10, 19, 18, 30), NY)    # open again, but the 18:00 bar is not closed
    e.rt._schedule_bars(e.ss, e.clock.now)
    assert e.mkt.bar_calls == 0, "polled for a bar that cannot exist yet (17:00 bar)"
    e.clock.now = epoch_of(datetime(2026, 10, 19, 19, 0, 5), NY)
    e.rt._schedule_bars(e.ss, e.clock.now)
    assert e.mkt.bar_calls == 1
    ok("gold daily break: no request storm; the 18:00 bar is fetched right after 19:00")


# ------------------------------------------------------------------ 5) signal rule / look-ahead / parity
def gold_1h():
    return load_history(GOLD)


def test_rsi2_rule():
    p = Params()
    d = signals(add_indicators(gold_1h(), p), p, GOLD)
    c = d["close"]
    expL = (c > d["ema200"]) & (d["rsi2"] < 10)
    expS = (c < d["ema200"]) & (d["rsi2"] > 90)
    live = d.iloc[200:]
    assert ((live.side == 1) <= expL.iloc[200:]).all() and ((live.side == -1) <= expS.iloc[200:]).all()
    assert (live.side == 1).sum() > 20 and (live.side == -1).sum() > 20
    ok(f"RSI(2) rule matches spec ({(live.side == 1).sum()} long / {(live.side == -1).sum()} short raw signals)")


def test_no_lookahead_and_parity():
    p = Params()
    df = gold_1h()
    full = signals(add_indicators(df, p), p, GOLD)
    rng = np.random.default_rng(5)
    for cut in rng.integers(700, len(df) - 1, 30):
        assert signals(add_indicators(df.iloc[:cut], p), p, GOLD).side.iloc[-1] == full.side.iloc[cut - 1]
    n, mism, tot = GOLD.history_bars, 0, 0
    for i in range(n, len(df)):
        s = signals(add_indicators(df.iloc[i - n + 1:i + 1], p), p, GOLD).side.iloc[-1]
        if s or full.side.iloc[i]:
            tot += 1
            mism += int(s != full.side.iloc[i])
    assert mism / max(tot, 1) < 0.02, (mism, tot)
    ok(f"no look-ahead; live 600-bar window == backtest ({mism}/{tot} mismatches)")


# ------------------------------------------------------------------ 6) config + regression
def test_config():
    os.environ.setdefault("TWELVE_DATA_API_KEY", "x")
    validate_config(Params(), FX_SET)
    bad = SymbolCfg("EUR/USD", "fx", "UTC", strategy="rsi2")
    try:
        validate_config(Params(), [bad])
        raise AssertionError("fx with tz=UTC accepted")
    except ConfigError:
        pass
    ok("gold/FX configs valid; wrong timezone refused")


def test_backtest_regression():
    p = Params()
    tot = []
    for cfg in FX_SET:
        t, _ = run_backtest(load_history(cfg), cfg, p, p.initial_equity)
        tot += t
    s = summarize(tot, p.initial_equity)
    assert s["trades"] >= 150 and s["avg_R_per_trade"] > 0.1 and s["profit_factor"] > 1.2, s
    assert all(t["exit_ts"].strftime("%H:%M") <= "17:00" for t in tot), "something was held past the daily flat"
    ok(f"backtest gold+FX: {s['trades']} trades, win rate {s['win_rate_%']}%, {s['avg_R_per_trade']:+.3f}R/trade, "
       f"PF {s['profit_factor']}")


if __name__ == "__main__":
    real = bot.signals
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            bot.signals = real
            fn()
    bot.signals = real
    print(f"\nall {len(PASSED)} fx tests passed")
