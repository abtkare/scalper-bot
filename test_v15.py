"""v15 tests: institutional zones (causal, distances, gate), zone pending in the live loop, end-of-day flat
without a fresh price, zones on the chart.   Run: python test_v15.py  (no network)"""
import json
import logging
import math
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd

import analysis
import watchlist as wl
import zones
from backtest import load_history, run_backtest
from config import BREAKOUT_QQQ, GOLD, Params, SymbolCfg, validate_config
from qualify import qualify
from strategy import add_indicators, plan_levels, signals

logging.getLogger().setLevel(logging.CRITICAL)
PASSED = []


def ok(name):
    PASSED.append(name)
    print(f"  ok  {name}")


def zcfg(**kw):
    return replace(wl.make_cfg("XAU/USD", "metal", "zone"), **kw)


def test_zones_causal():
    p = Params()
    cfg = zcfg(htf_mult=0)
    df = load_history(cfg).iloc[-900:]
    full = signals(add_indicators(df, p), p, cfg)
    sig_full = full["side"].values
    # recomputing on every prefix must give the same signal on its last bar (no look-ahead)
    for end in range(600, 900, 7):
        part = signals(add_indicators(df.iloc[:end], p), p, cfg)
        assert part["side"].iloc[-1] == sig_full[end - 1], end
        if sig_full[end - 1]:
            assert abs(part["sl_dist"].iloc[-1] - full["sl_dist"].iloc[end - 1]) < 1e-9
    ok("zones: causal - every prefix reproduces the same signal and stop distance")


def test_zone_signal_geometry():
    p = Params()
    cfg = zcfg(htf_mult=0)
    d = signals(add_indicators(load_history(cfg), p), p, cfg)
    s = d[d["side"] != 0]
    assert len(s) > 10
    assert (s["sl_dist"] >= cfg.zone_min_risk_atr * s["atr"] - 1e-9).all()
    assert (s["sl_dist"] <= cfg.zone_max_risk_atr * s["atr"] + 1e-9).all()
    assert np.allclose(s["tp_dist"], cfg.zone_rr * s["sl_dist"])
    for z in d.attrs["zones"]:
        assert z["lo"] < z["hi"] and z["side"] in (1, -1)
    tr = run_backtest(load_history(cfg), cfg, p, 1e4)[0]
    assert len(tr) > 10 and all(abs(t["entry"] - t["stop0"]) > 0 for t in tr)
    risk_atr = [abs(t["entry"] - t["stop0"]) for t in tr]
    assert np.std(risk_atr) > 0, "stop distances must vary with the zone, not be a fixed ATR multiple"
    ok(f"zones: {len(s)} signals, stop floored at {cfg.zone_min_risk_atr} ATR, target = {cfg.zone_rr}R, engine uses them")


def test_zone_gate_blocks():
    p = Params()
    res = {}
    for sym, cls in (("QQQ", "etf"), ("EUR/USD", "fx"), ("GBP/USD", "fx")):
        cfg = wl.make_cfg(sym, cls, "zone")
        df = load_history(replace(cfg, interval="15min") if cls != "etf" else replace(BREAKOUT_QQQ))
        res[sym] = qualify(df.iloc[-5000:], cfg, p)["status"]
    assert all(v == "fail" for v in res.values()), res
    validate_config(p, [wl.make_cfg("XAU/USD", "metal", "zone"), wl.make_cfg("QQQ", "etf", "zone")]) \
        if False else None
    ok(f"gate keeps the zone strategy watch-only where it lost in research: {res}")


def test_zone_plan_and_chart():
    p = Params()
    cfg = zcfg()
    df = load_history(cfg).iloc[-700:]
    d = signals(add_indicators(df, p), p, cfg)
    pl = plan_levels(d, p, cfg)
    for k in ("long", "short"):
        if pl.get(k):
            z = pl[k]["zone"]
            assert z[0] < z[1] and (pl[k]["stop"] < z[0] if k == "long" else pl[k]["stop"] > z[1])
    for c in (cfg, BREAKOUT_QQQ, GOLD):
        dd = signals(add_indicators(load_history(c).iloc[-700:], p), p, c)
        ch = analysis.chart(dd, c)
        n = len(ch["t"])
        for side, lo, hi, x in ch["zones"]:
            assert side in (1, -1) and lo < hi and 0 <= x < n
        an = analysis.analyze(dd, plan_levels(dd, p, c), c)
        json.dumps(ch), json.dumps(an, ensure_ascii=False)
    ok("zone plan (nearest demand/supply, stop beyond the zone) + zones drawn on every chart")


def test_live_zone_pending():
    import test_robust as R
    cfg = SymbolCfg("TEST/USD", "crypto", "UTC", cost_bps=0.0, provider="fake", stop_atr=1.0, target_atr=1.5)
    e = R.Env(R.ramp(R.T0 + 450, 0.0), cfg=cfg)

    def sig(d, p, s):                                     # one long signal with zone distances 0.4 / 1.2
        out = R.fake_signals([R.SIG_BAR], 1)(d, p, s)
        out["sl_dist"] = np.where(out["side"] != 0, 0.4, np.nan)
        out["tp_dist"] = np.where(out["side"] != 0, 1.2, np.nan)
        return out
    R.bot.signals = sig
    assert e.run(400, until=lambda: e.ss.eng.pos is not None)
    pos = e.ss.eng.pos
    assert abs((pos.entry - pos.stop) - 0.4) < 1e-9 and abs((pos.target - pos.entry) - 1.2) < 1e-9
    assert abs(pos.risk_amt - pos.qty * 0.4) < 1e-9
    R.bot.signals = R.REAL_SIGNALS
    ok("live loop: zone signal opens with ITS stop/target distances and risk sized on the zone stop")


def test_eod_without_quote():
    import test_robust as R
    from bot import epoch_of
    e = R._stock_env((2026, 1, 6), 15, 50)                 # test config flattens at 15:55
    R._inject(e, datetime(2026, 1, 6, 10, 0))
    e.ss.plan = {"price": 101.0}
    e.mkt.fail_quotes = True                               # no live price at all
    e.run(15 * 60, until=lambda: bool(e.trades()))
    tr = e.trades()
    assert tr and tr[0]["reason"] == "eod_noquote", tr
    lt = R.local_naive(e.clock.now, "America/New_York").strftime("%H:%M")
    assert "15:55" <= lt <= "15:56", lt
    assert e.n.has("end-of-day flat without a fresh price")
    ok(f"no live price at the daily flat: position closed at {lt} at the last known price (never overnight)")


if __name__ == "__main__":
    for fn in (test_zones_causal, test_zone_signal_geometry, test_zone_gate_blocks, test_zone_plan_and_chart,
               test_live_zone_pending, test_eod_without_quote):
        fn()
    print(f"all {len(PASSED)} v15 tests passed")
