"""v14 tests: swing trailing stop, slippage (backtest + live measurement), chart/analysis feed.
Run: python test_v14.py   (no network)"""
import json
import logging
import math
from dataclasses import replace

import numpy as np
import pandas as pd

import analysis
import watchlist as wl
from backtest import load_history, run_backtest, summarize
from config import GOLD, EURUSD, BREAKOUT_QQQ, SCALP_QQQ, Params, SymbolCfg, validate_config, ConfigError
from engine import Account, NO_TARGET_MULT, Position, SymbolEngine
from strategy import add_indicators, plan_levels, signals

logging.getLogger().setLevel(logging.CRITICAL)
PASSED = []


def ok(name):
    PASSED.append(name)
    print(f"  ok  {name}")


def eng_for(**kw):
    cfg = SymbolCfg("T", "crypto", "UTC", cost_bps=0.0, stop_atr=1.0, target_atr=2.0, allow_short=True, **kw)
    p = Params(initial_equity=1000, risk_pct=0.01, max_leverage=50)
    return SymbolEngine(cfg, p, Account(1000, p))


def test_trail_rules():
    e = eng_for(trail="swing", trail_after_r=0.5, trail_buf_atr=0.1)
    ts = pd.Timestamp("2026-01-05 10:00")
    e.try_enter(ts, 100.0, 1, 1.0, ts.date())
    assert e.pos.stop == 99.0 and e.pos.stop0 == 99.0
    assert not e.trail_update(99.8, 101.0, 1.0, 100.3), "moved before +0.5R"
    assert e.trail_update(99.8, 101.0, 1.0, 100.6) and abs(e.pos.stop - 99.7) < 1e-12
    assert not e.trail_update(99.5, 101.0, 1.0, 100.8), "stop was WIDENED"
    assert not e.trail_update(100.9, 101.0, 1.0, 100.8), "stop above the close"
    assert e.trail_update(100.5, 101.0, 1.0, 101.2) and abs(e.pos.stop - 100.4) < 1e-12
    tr = e.on_bar_close(ts + pd.Timedelta(minutes=30), 101.0, 101.1, 100.3, 100.35, ts.date(), False)
    assert tr["reason"] == "trail" and tr["R"] > 0 and tr["trail_moves"] == 2 and tr["stop0"] == 99.0, tr
    # short mirror
    e = eng_for(trail="swing", trail_after_r=0.0)
    e.try_enter(ts, 100.0, -1, 1.0, ts.date())
    assert e.trail_update(np.nan, 100.4, 1.0, 99.9) and abs(e.pos.stop - 100.5) < 1e-12
    assert not e.trail_update(np.nan, 100.8, 1.0, 99.9)
    # no trail configured -> never moves
    e = eng_for()
    e.try_enter(ts, 100.0, 1, 1.0, ts.date())
    assert not e.trail_update(99.9, 101, 1.0, 105.0) and e.pos.stop == 99.0
    ok("trailing stop: waits for +0.5R, follows confirmed swings, never widens, short side mirrored")


def test_no_target_plan():
    e = eng_for(trail="swing", trail_no_target=True)
    ts = pd.Timestamp("2026-01-05 10:00")
    plan = e.plan_entry(ts, 100.0, 1, 1.0, ts.date())
    assert abs(plan.target_dist - 2.0 * NO_TARGET_MULT) < 1e-9 and plan.stop_dist == 1.0
    ok(f"no-target lines send a far safety target ({NO_TARGET_MULT:g}x) to the broker, stop unchanged")


def test_backtest_slippage():
    df = load_history(BREAKOUT_QQQ)
    p = Params()
    p.initial_equity = 10000.0
    a = summarize(run_backtest(df, BREAKOUT_QQQ, p, 1e4, slip_bps=0.0)[0], 1e4)
    b = summarize(run_backtest(df, BREAKOUT_QQQ, p, 1e4, slip_bps=1.0)[0], 1e4)
    assert b["avg_R_per_trade"] < a["avg_R_per_trade"] and b["net_pnl"] < a["net_pnl"]
    tr = run_backtest(df, BREAKOUT_QQQ, p, 1e4, slip_bps=1.0)[0]
    stops = [t for t in tr if t["reason"] == "stop"]
    assert stops and all(t["slip_out_bps"] > 0.99 for t in stops), "stop exits must be filled worse"
    assert all(t["slip_out_bps"] == 0 for t in tr if t["reason"] == "target"), "targets are limit orders"
    assert Params().bt_slip_bps == 0.25
    ok(f"backtest slippage: QQQ {a['avg_R_per_trade']:+.3f}R -> {b['avg_R_per_trade']:+.3f}R at 1 bps; "
       f"stops slip, targets do not; gate uses 0.25 bps")


def test_adopted_exits():
    p = Params()
    p.initial_equity = 1e4
    for cfg in (GOLD, EURUSD):
        df = load_history(cfg)
        base = replace(cfg, trail="", trail_no_target=False)
        cut = df.index[int(len(df) * 0.6)]
        for c in (base, cfg):
            tr = run_backtest(df, c, p, 1e4)[0]
            c.res = [summarize([t for t in tr if f(t["exit_ts"])], 1e4)["avg_R_per_trade"]
                     for f in (lambda x: x < cut, lambda x: x >= cut)]
        assert cfg.res[0] > base.res[0] and cfg.res[1] > base.res[1], (cfg.symbol, base.res, cfg.res)
    assert wl.make_cfg("XAU/USD", "metal", "rsi2").trail == "swing"
    assert wl.make_cfg("EUR/USD", "fx", "rsi2").trail_no_target is False
    assert wl.make_cfg("QQQ", "etf", "breakout").trail == "" and wl.make_cfg("QQQ", "etf", "scalp").trail == ""
    validate_config(Params(), [wl.make_cfg("XAU/USD", "metal", "rsi2")]) if False else None
    try:
        validate_config(Params(), [replace(GOLD, trail="bogus")])
        raise AssertionError("bad trail accepted")
    except ConfigError:
        pass
    ok("adopted exits (gold, EUR/USD) beat the fixed exit in both halves; QQQ lines unchanged")


def test_position_compat():
    old = {"side": 1, "qty": 1, "entry": 100, "stop": 99, "target": 102, "entry_ts": "2026-01-05T10:00:00",
           "risk_amt": 1.0}
    pos = Position.from_dict(old)                      # state file written by v10
    assert pos.stop0 == 99 and pos.trail_moves == 0 and pos.planned == 0
    assert Position.from_dict(pos.to_dict()).stop0 == 99
    ok("v10 state files load (stop0 defaults to the saved stop)")


def test_chart_and_analysis():
    p = Params()
    for cfg in (BREAKOUT_QQQ, SCALP_QQQ, GOLD):
        df = load_history(cfg).iloc[-700:]
        d = signals(add_indicators(df, p), p, cfg)
        plan = plan_levels(d, p, cfg)
        ch = analysis.chart(d, cfg)
        n = len(ch["t"])
        assert n == analysis.CHART_BARS and all(len(ch[k]) == n for k in "ohlc")
        assert all(ch["l"][i] <= min(ch["o"][i], ch["c"][i]) and ch["h"][i] >= max(ch["o"][i], ch["c"][i])
                   for i in range(n))
        assert ch["t"] == sorted(ch["t"]) and ch["lines"]
        for i, kind, v in ch["piv"]:
            assert 0 <= i < n and (v == ch["h"][i] if kind == "H" else v == ch["l"][i])
        an = analysis.analyze(d, plan, cfg)
        assert an["desc"] and an["verdict"] and len(an["lines"]) >= 3
        an2 = analysis.analyze(d, plan, cfg, {"side": "long", "stop": 1.0})
        assert "مفتوحة" in an2["verdict"]
        json.dumps(ch), json.dumps(an, ensure_ascii=False)
    # epochs: naive New York 09:30 in winter = 14:30 UTC
    assert analysis.epochs([pd.Timestamp("2026-01-05 09:30")], "America/New_York")[0] == \
        int(pd.Timestamp("2026-01-05 14:30", tz="UTC").timestamp())
    ok("chart feed: 80 candles, valid OHLC, swing pivots on real highs/lows, Arabic analysis + verdict")


def test_live_trail_and_slip():
    import test_robust as R
    from bot import epoch_of
    from datetime import datetime
    cfg = SymbolCfg("TEST/USD", "crypto", "UTC", cost_bps=0.0, provider="fake", trail="swing",
                    trail_after_r=0.0, trail_no_target=True)
    # uptrend with 50-minute waves -> swing lows form every few bars and rise
    price = lambda t: 100.0 + max(0.0, t - R.T0 - 300) * 0.0002 + 0.5 * math.sin((t - R.T0) / 3000 * 2 * math.pi)
    e = R.Env(price, cfg=cfg)
    e.run(6 * 3600, dt=1.0, until=lambda: (e.ss.eng.pos is not None and e.ss.eng.pos.trail_moves >= 2) or bool(e.trades()))
    pos = e.ss.eng.pos
    assert pos is not None and pos.trail_moves >= 2 and pos.stop > pos.stop0, (pos, e.n.msgs[-5:])
    assert e.n.has("TRAIL TEST/USD")
    st = e.rt.status(e.clock.now)["symbols"]["TEST/USD"]
    assert st["position"]["target"] is None and st["position"]["stop0"] == pos.stop0 and st["no_target"]
    ch = e.rt.charts()["TEST/USD"]
    assert ch["pos"]["stop"] == pos.stop and ch["pos"]["target"] is None and len(ch["t"]) > 10
    assert st["analysis"] and "مفتوحة" in st["analysis"]["verdict"]
    json.dumps(e.rt.charts()), json.dumps(e.rt.status(e.clock.now), default=str)
    # the next dip below the trailed stop closes the trade as "trail", with slippage recorded
    e.mkt.price = lambda t: pos.stop - 1.0
    assert e.run(120, until=lambda: bool(e.trades()))
    tr = e.trades()[0]
    assert tr["reason"] == "trail" and float(tr["slip_out_bps"]) > 0, tr
    assert e.ss.slip["n"] == 1 and "slip_in_bps" in tr and "slip_fill_bps" in tr
    s2 = e.rt.snapshot()["symbols"]["TEST/USD"]["slip"]
    assert s2["n"] == 1
    R.bot.signals = R.REAL_SIGNALS
    ok(f"live loop: stop trailed {pos.trail_moves}x behind swings, exit booked as 'trail', slippage measured "
       f"({float(tr['slip_out_bps']):.1f} bps gap through the stop)")


if __name__ == "__main__":
    for fn in (test_trail_rules, test_no_target_plan, test_backtest_slippage, test_adopted_exits,
               test_position_compat, test_chart_and_analysis, test_live_trail_and_slip):
        fn()
    print(f"all {len(PASSED)} v14 tests passed")
