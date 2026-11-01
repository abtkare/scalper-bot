"""v16 tests: charts right after a restart while the market is closed, money flow per candle, timeframe switch.
Run: python test_v16.py   (no network)"""
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

import analysis
from backtest import load_history
from config import BREAKOUT_QQQ, GOLD, Params, SymbolCfg
from strategy import add_indicators, signals

logging.getLogger().setLevel(logging.CRITICAL)
PASSED = []


def ok(name):
    PASSED.append(name)
    print(f"  ok  {name}")


def test_money_flow():
    df = pd.DataFrame({"open": [10, 10, 10.0], "high": [11, 11, 11.0], "low": [9, 9, 9.0],
                       "close": [11, 9, 10.0], "volume": [100, 100, 100.0]},
                      index=pd.date_range("2026-01-05 10:00", periods=3, freq="15min"))
    mf = analysis.money_flow(df)
    usd = 100 * np.array([31, 29, 30.0]) / 3
    assert mf["hasv"] and np.allclose(mf["buy"] + mf["sell"], usd)
    assert np.isclose(mf["buy"][0], usd[0]) and np.isclose(mf["sell"][1], usd[1])      # close at high / at low
    assert np.isclose(mf["buy"][2], usd[2] / 2)
    assert not analysis.money_flow(df.assign(volume=0.0))["hasv"]
    ok("money flow: volume x price split into buying/selling by the close position; no volume -> hasv False")


def test_chart_volume_and_tf():
    p = Params()
    for cfg, hasv in ((BREAKOUT_QQQ, True), (GOLD, False)):
        df = load_history(cfg).iloc[-600:]
        d = signals(add_indicators(df, p), p, cfg)
        ch = analysis.chart(d, cfg)
        assert ch["hasv"] is hasv and (len(ch.get("vb", [])) == len(ch["t"]) if hasv else "vb" not in ch)
        lines = dict(analysis.analyze(d, {}, cfg)["lines"])
        assert "💰 تدفق الأموال" in lines and (("لا يوجد" in lines["💰 تدفق الأموال"]) != hasv)
        for m in analysis.TF_MULTS[cfg.interval_minutes]:
            x = analysis.chart_tf(df, cfg, p, m)
            assert "error" not in x and x["iv"] == analysis.tf_label(cfg.interval_minutes * m)
            n = len(x["t"])
            assert n >= 10 and all(x["l"][i] <= x["h"][i] for i in range(n)) and x["t"] == sorted(x["t"])
            if m > 1:
                assert np.isclose(max(x["h"]), df["high"].iloc[-len(df):].resample(
                    f"{cfg.interval_minutes * m}min" if cfg.interval_minutes * m < 1440 else "1D",
                    label="left", closed="left").max().dropna().iloc[-n:].max(), rtol=1e-6)
            json.dumps(x)
    ok("charts: money-flow bars where volume exists, timeframe switch 15m->30m/1h/4h and 1h->4h/1d")


def test_startup_chart_when_closed():
    import test_robust as R
    from bot import epoch_of
    cfg = SymbolCfg("QQQ", kind="stock", tz="America/New_York", cost_bps=0.0, flatten_eod=True, provider="fake",
                    windows=[("09:45", "15:30")])
    start = epoch_of(datetime(2026, 1, 10, 11, 0, 0), "America/New_York")          # a Saturday
    e = R.Env(lambda t: 100.0, tz="America/New_York", cfg=cfg, start=start, sig_bars=[])
    R.bot.signals = lambda d, p, s: R.fake_signals([], 1)(d, p, s).assign(side=1)   # a signal on every bar
    e.run(120, dt=1.0)
    assert e.mkt.bar_calls == 1, e.mkt.bar_calls                 # exactly ONE start-up download
    assert e.rt.charts().get("QQQ") and e.ss.pending is None and e.ss.eng.pos is None
    assert not any("too late" in m for m in e.n.msgs)
    tf = e.rt.chart_tf({"id": ["QQQ"], "tf": ["4"]})
    assert "error" in tf or tf["iv"] == "20min"
    e.run(600, dt=5.0)
    assert e.mkt.bar_calls == 1                                  # nothing more while the market is closed
    R.bot.signals = R.REAL_SIGNALS
    ok("restart on a weekend: one download per stream -> chart visible, no entry, no further API calls")


if __name__ == "__main__":
    for fn in (test_money_flow, test_chart_volume_and_tf, test_startup_chart_when_closed):
        fn()
    print(f"all {len(PASSED)} v16 tests passed")
