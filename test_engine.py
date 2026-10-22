"""Sanity tests for the engine and strategy. Run: python test_engine.py"""
import pandas as pd
import numpy as np
from config import Params, SymbolCfg
from engine import Account, SymbolEngine
from strategy import add_indicators, signals

p = Params(initial_equity=1000, risk_pct=0.01, max_leverage=10)
crypto = SymbolCfg("TEST/USD", "crypto", "UTC", cost_bps=0.0)
ts = pd.Timestamp("2026-01-05 10:00")
day = ts.date()


def fresh(scfg=crypto):
    a = Account(1000, p)
    return a, SymbolEngine(scfg, p, a)


# 1) sizing: risk 1% of 1000 = 10; stop distance = 1*ATR(2) = 2 -> qty 5
a, e = fresh()
pos = e.try_enter(ts, 100.0, +1, 2.0, day)
assert abs(pos.qty - 5.0) < 1e-9 and abs(pos.stop - 98.0) < 1e-9 and abs(pos.target - 103.0) < 1e-9

# 2) target hit -> +1.5R
tr = e.on_bar_close(ts, 100, 103.5, 99.5, 103, day, False)
assert tr["reason"] == "target" and abs(tr["pnl"] - 15.0) < 1e-9 and abs(tr["R"] - 1.5) < 1e-9

# 3) stop hit -> -1R
a, e = fresh(); e.try_enter(ts, 100.0, +1, 2.0, day)
tr = e.on_bar_close(ts, 100, 101, 97.5, 98, day, False)
assert tr["reason"] == "stop" and abs(tr["R"] + 1.0) < 1e-9

# 4) both touched in one bar -> stop wins (conservative)
a, e = fresh(); e.try_enter(ts, 100.0, +1, 2.0, day)
tr = e.on_bar_close(ts, 100, 104, 97, 100, day, False)
assert tr["reason"] == "stop"

# 5) costs reduce pnl
c2 = SymbolCfg("TEST/USD", "crypto", "UTC", cost_bps=10.0)
a, e = fresh(c2); e.try_enter(ts, 100.0, +1, 2.0, day)
tr = e.on_bar_close(ts, 100, 103.5, 99.5, 103, day, False)
assert tr["cost"] > 0 and tr["pnl"] < tr["pnl_gross"]

# 6) shorts are blocked unless allowed
a, e = fresh(); assert e.try_enter(ts, 100.0, -1, 2.0, day) is None

# 7) leverage cap: risk-based qty would be 500 but max notional = 1000*10 -> 100 units
a, e = fresh(); pos = e.try_enter(ts, 100.0, +1, 0.02, day)
assert abs(pos.qty * 100.0 - 10000.0) < 1e-6

# 8) consecutive-loss limit blocks new entries the same day
a, e = fresh()
for _ in range(p.max_consec_losses):
    e.try_enter(ts, 100.0, +1, 2.0, day)
    e.on_bar_close(ts, 100, 101, 97, 98, day, False)
assert e.try_enter(ts, 100.0, +1, 2.0, day) is None
# ...but a new day resets it
assert e.try_enter(ts + pd.Timedelta(days=1), 100.0, +1, 2.0, (ts + pd.Timedelta(days=1)).date()) is not None

# 9) daily loss cap
p2 = Params(initial_equity=1000, risk_pct=0.02, daily_loss_cap=0.03, max_consec_losses=99)
a = Account(1000, p2); e = SymbolEngine(crypto, p2, a)
for _ in range(2):
    e.try_enter(ts, 100.0, +1, 2.0, day); e.on_bar_close(ts, 100, 101, 97, 98, day, False)
assert e.try_enter(ts, 100.0, +1, 2.0, day) is None   # equity < 97% of day start

# 10) entry window respected for stocks
stock = SymbolCfg("QQQ", "stock", "America/New_York", flatten_eod=True, windows=[("09:35", "11:00")])
a, e = fresh(stock)
assert e.try_enter(pd.Timestamp("2026-01-05 12:00"), 100.0, +1, 0.5, day) is None
assert e.try_enter(pd.Timestamp("2026-01-05 09:40"), 100.0, +1, 0.5, day) is not None

# 11) EOD flatten
tr = e.on_bar_close(pd.Timestamp("2026-01-05 15:55"), 100, 100.1, 99.9, 100.05, day, True)
assert tr["reason"] == "eod"

# 12) no look-ahead: signal at bar i must not change if future bars change
rng = np.random.default_rng(1)
n = 400
idx = pd.date_range("2026-01-05", periods=n, freq="5min")
close = 100 + np.cumsum(rng.normal(0, 0.1, n))
df = pd.DataFrame({"open": close - 0.02, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1.0}, index=idx)
s1 = signals(add_indicators(df, p), p, crypto)["side"].iloc[:300].values
df2 = df.copy(); df2.iloc[300:, :4] = df2.iloc[300:, :4] * 1.5
s2 = signals(add_indicators(df2, p), p, crypto)["side"].iloc[:300].values
assert (s1 == s2).all()

print("all tests passed")
