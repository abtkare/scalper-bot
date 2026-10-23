"""Scalping mode tests (v4): signal logic, no look-ahead, live/backtest parity, engine exits for longs AND
shorts through the real bot loop, trading-mode switching from Telegram / dashboard, regression vs RESEARCH.md.
Run: python test_scalp.py      (no network, no API key)
"""
import logging
import os
import tempfile

import numpy as np
import pandas as pd

import bot
import test_robust as tr
from backtest import resample_bars, run_backtest, summarize
from config import (BREAKOUT_QQQ, SCALP_QQQ, ConfigError, Params, SymbolCfg, read_trading_mode,
                    symbols_for_mode, validate_config)
from data import load_csv
from strategy import add_indicators, signals
from telegram_ctl import TelegramControl

logging.getLogger().setLevel(logging.CRITICAL)
PASSED = []
HERE = os.path.dirname(os.path.abspath(__file__))


def ok(name):
    PASSED.append(name)
    print(f"  ok  {name}")


def qqq15():
    return resample_bars(load_csv(os.path.join(HERE, "data", "QQQ_5min.csv")), 15)


# ------------------------------------------------------------------ 1) signal = the documented rule
def test_rule():
    p = Params()
    d = signals(add_indicators(qqq15(), p), p, SCALP_QQQ)
    c, k = d["close"], SCALP_QQQ.vwap_k
    exp_l = (c > d["ema200"]) & (c < d["vwap"] - k * d["atr"])
    exp_s = (c < d["ema200"]) & (c > d["vwap"] + k * d["atr"])
    live = d.iloc[200:]
    assert ((live.side == 1) <= exp_l.iloc[200:]).all() and ((live.side == -1) <= exp_s.iloc[200:]).all()
    assert (live.side == 1).sum() > 50 and (live.side == -1).sum() > 50, "both sides must fire"
    assert (d.side.iloc[:200] == 0).all(), "warm-up must be flat"
    # VWAP resets each session: first bar of a day = its own typical price
    first = d.groupby(d.index.normalize()).head(1)
    tp = (first.high + first.low + first.close) / 3
    assert np.allclose(first.vwap, tp), "VWAP must reset at each session open"
    ok(f"rule matches spec: {(live.side == 1).sum()} long / {(live.side == -1).sum()} short signals, VWAP resets daily")


# ------------------------------------------------------------------ 2) no look-ahead
def test_no_lookahead():
    p = Params()
    df = qqq15()
    full = signals(add_indicators(df, p), p, SCALP_QQQ)
    rng = np.random.default_rng(3)
    for cut in rng.integers(400, len(df) - 1, 40):
        part = signals(add_indicators(df.iloc[:cut], p), p, SCALP_QQQ)
        assert part.side.iloc[-1] == full.side.iloc[cut - 1], f"future data changed the signal at {df.index[cut - 1]}"
    ok("signal on bar t never depends on bars after t (40 random cut points)")


# ------------------------------------------------------------------ 3) live window (600 bars) == backtest signal
def test_live_parity():
    p = Params()
    df = qqq15()
    full = signals(add_indicators(df, p), p, SCALP_QQQ)
    n = SCALP_QQQ.history_bars
    mism = tot = 0
    for i in range(n, len(df)):
        w = df.iloc[i - n + 1:i + 1]
        s = signals(add_indicators(w, p), p, SCALP_QQQ).side.iloc[-1]
        if s != 0 or full.side.iloc[i] != 0:
            tot += 1
            mism += int(s != full.side.iloc[i])
    rate = mism / max(tot, 1)
    assert rate < 0.02, f"live window disagrees with the backtest on {rate:.1%} of signals"
    ok(f"live 600-bar window agrees with the full-history backtest ({mism}/{tot} signal mismatches)")


# ------------------------------------------------------------------ 4) real bot loop: short + long, target and stop
def _scalp_test_cfg():
    return SymbolCfg("TEST/USD", "crypto", "UTC", cost_bps=0.0, provider="fake", strategy="scalp",
                     allow_short=True, stop_atr=2.0, target_atr=0.75)


def test_loop_short_target():
    e = tr.Env(tr.ramp(tr.T0 + 450, -0.05), cfg=_scalp_test_cfg(), sig_side=-1)
    assert e.run(900, until=lambda: bool(e.trades())), "short never closed"
    t = e.trades()[0]
    assert t["side"] == "short" and t["reason"] == "target", t
    assert abs(float(t["R"]) - 0.375) < 0.02, f"target 0.75 ATR / stop 2 ATR = +0.375R, got {t['R']}"
    ok(f"live loop: SHORT entry -> target exit at {float(t['R']):+.3f}R")


def test_loop_short_stop():
    e = tr.Env(tr.ramp(tr.T0 + 450, 0.05), cfg=_scalp_test_cfg(), sig_side=-1)
    assert e.run(1200, until=lambda: bool(e.trades()))
    t = e.trades()[0]
    assert t["side"] == "short" and t["reason"] == "stop" and -1.1 < float(t["R"]) <= -0.99, t
    ok(f"live loop: SHORT stop exit at {float(t['R']):+.2f}R")


def test_loop_long_target():
    e = tr.Env(tr.ramp(tr.T0 + 450, 0.05), cfg=_scalp_test_cfg(), sig_side=1)
    assert e.run(900, until=lambda: bool(e.trades()))
    t = e.trades()[0]
    assert t["side"] == "long" and t["reason"] == "target" and abs(float(t["R"]) - 0.375) < 0.02, t
    ok(f"live loop: LONG entry -> target exit at {float(t['R']):+.3f}R")


def test_breakout_never_shorts():
    cfg = SymbolCfg("TEST/USD", "crypto", "UTC", cost_bps=0.0, provider="fake", strategy="breakout")
    e = tr.Env(tr.ramp(tr.T0 + 450, -0.05), cfg=cfg, sig_side=-1)
    e.run(900)
    assert not e.trades() and e.ss.eng.pos is None
    ok("breakout mode still never opens shorts")


# ------------------------------------------------------------------ 5) trading modes
def test_modes():
    assert [s.symbol for s in symbols_for_mode("breakout")] == ["QQQ"]          # old name of "qqq"
    assert [s.symbol for s in symbols_for_mode("qqq")] == ["QQQ"]
    assert [s.symbol for s in symbols_for_mode("fx")] == ["XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY"]
    assert [s.symbol for s in symbols_for_mode("all")] == ["QQQ", "XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY"]
    assert [s.symbol for s in symbols_for_mode("scalp")] == ["QQQ-SCALP"]
    assert [s.symbol for s in symbols_for_mode("both")] == ["QQQ", "QQQ-SCALP"]
    assert SCALP_QQQ.ticker == "QQQ" and BREAKOUT_QQQ.ticker == "QQQ"
    os.environ["TWELVE_DATA_API_KEY"] = "x"
    for m in ("all", "fx", "qqq", "breakout", "scalp", "both"):
        validate_config(Params(), symbols_for_mode(m))
    d = tempfile.mkdtemp()
    f = os.path.join(d, "mode.txt")
    old = os.environ.pop("TRADING_MODE", None)
    try:
        assert read_trading_mode(f) == "all"                           # default
        os.environ["TRADING_MODE"] = "scalp"
        assert read_trading_mode(f) == "scalp"                         # env
        open(f, "w").write("both\n")
        assert read_trading_mode(f) == "both"                          # file (phone) wins
        open(f, "w").write("breakout\n")
        assert read_trading_mode(f) == "qqq"                           # legacy name from v4
        open(f, "w").write("garbage\n")
        assert read_trading_mode(f) == "scalp"                         # bad file ignored
        os.environ["TRADING_MODE"] = "yolo"
        try:
            read_trading_mode(f)
            raise AssertionError("bad TRADING_MODE accepted")
        except ConfigError:
            pass
    finally:
        os.environ.pop("TRADING_MODE", None)
        if old is not None:
            os.environ["TRADING_MODE"] = old
    bad = SymbolCfg("X", "stock", "America/New_York", strategy="scalp", vwap_k=0)
    try:
        validate_config(Params(), [bad])
        raise AssertionError("vwap_k=0 accepted")
    except ConfigError:
        pass
    ok("modes all/fx/qqq/scalp/both build valid configs; file > env > default; legacy names ok; bad values refused")


def test_mode_switch_refused_with_position():
    e = tr.Env(tr.ramp(tr.T0 + 450, 0.0), cfg=_scalp_test_cfg(), sig_side=1)
    mf = os.path.join(e.tmp, "trading_mode.txt")
    assert e.run(400, until=lambda: e.ss.eng.pos is not None)
    okk, msg = e.rt.request_mode("both", mf)
    assert not okk and "مرفوض" in msg and not os.path.exists(mf)
    e.ss.eng.pos = None
    okk, msg = e.rt.request_mode("both", mf)
    assert okk and open(mf).read().strip() == "both"
    okk, _ = e.rt.request_mode("nonsense", mf)
    assert not okk
    ok("mode switch refused while a position is open; accepted (file written) when flat")


def test_telegram_and_dashboard_mode():
    calls = []
    ctl = TelegramControl(lambda: {"mode": "all"}, {"stop": "s", "flatten": "f", "halt": "h"},
                          token="", chat="", mode_fn=lambda m: calls.append(m) or "done")
    assert "all" in ctl.handle("/mode")
    assert ctl.handle("/mode scalp") == "done" and calls == ["scalp"]
    assert "/mode" in ctl.handle("/help")
    import dashboard
    assert {"mode_all", "mode_fx", "mode_qqq", "mode_scalp"} <= set(dashboard.COMMANDS)
    assert all(f"m_{m}" in dashboard.PAGE for m in ("all", "fx", "qqq", "scalp"))
    ok("/mode works from Telegram; dashboard has the mode buttons (all / fx / qqq / scalp)")


def test_leftover_position_exit_only():
    cfg = _scalp_test_cfg()
    cfg.exit_only = True
    e = tr.Env(tr.ramp(tr.T0 + 450, 0.05), cfg=cfg, sig_side=1)
    e.run(900)
    assert e.ss.eng.pos is None and not e.trades(), "exit-only symbol must never open a trade"
    ok("exit-only (leftover strategy from a previous mode) never opens new trades")


# ------------------------------------------------------------------ 6) regression vs RESEARCH.md
def test_backtest_regression():
    p = Params()
    trades, _ = run_backtest(qqq15(), SCALP_QQQ, p, p.initial_equity)
    s = summarize(trades, p.initial_equity)
    assert s["trades"] >= 150 and s["win_rate_%"] >= 70 and s["avg_R_per_trade"] > 0 and s["profit_factor"] > 1.2, s
    sides = {t["side"] for t in trades}
    assert sides == {"long", "short"}
    ok(f"backtest QQQ scalp: {s['trades']} trades, win rate {s['win_rate_%']}%, "
       f"{s['avg_R_per_trade']:+.3f}R/trade, PF {s['profit_factor']}, DD {s['max_drawdown_%']}%")


if __name__ == "__main__":
    real = bot.signals
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            bot.signals = real
            fn()
    bot.signals = real
    print(f"\nall {len(PASSED)} scalp tests passed")
