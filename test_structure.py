"""Market-structure reading + higher-timeframe filter: causal, deterministic, blocks counter-trend entries."""
import numpy as np
import pandas as pd

import structure as st
from config import get_params
from strategy import add_indicators, plan_levels, signals
from watchlist import make_cfg


def bars(n=900, seed=1):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-01-05 09:30", periods=n, freq="15min")
    c = 100 + np.cumsum(rng.normal(0, 0.3, n)) + np.sin(np.arange(n) / 40) * 4
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.random(n) * 0.2
    l = np.minimum(o, c) - rng.random(n) * 0.2
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1000.0}, index=t)


def test_causal():
    p = get_params()
    d = add_indicators(bars(), p)
    full = st.analyze(d)
    for end in (300, 451, 777):
        part = st.analyze(d.iloc[:end])
        for col in ("st_trend", "swing_hi", "swing_lo"):
            a, b = full[col].iloc[:end].values, part[col].values
            assert np.allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1)), (col, end)
    h1 = st.htf_trend(d, 60)
    h2 = st.htf_trend(d.iloc[:600], 60)
    assert (h1.iloc[:600].values == h2.values).all()


def test_swings_confirmed_late():
    d = pd.DataFrame({"high": [1, 2, 5, 2, 1, 1, 1.0], "low": [0, 1, 4, 1, 0, 0, 0.0]})
    sh, _ = st.swings(d.high.values, d.low.values, 2)
    assert np.isnan(sh[2]) and sh[4] == 5                 # visible only after 2 more bars closed


def test_filter_blocks_counter_trend():
    p = get_params()
    cfg = make_cfg("QQQ", "etf", "breakout")
    assert cfg.htf_mult == 4
    d = signals(add_indicators(bars(), p), p, cfg)
    assert "htf_trend" in d.columns
    assert not ((d.side == 1) & (d.htf_trend < 0)).any()
    assert not ((d.side == -1) & (d.htf_trend > 0)).any()
    pl = plan_levels(d, p, cfg)
    assert "structure" in pl and pl["structure"]["text"]
    if int(d.htf_trend.iloc[-1]) < 0:
        assert pl["long"] is None


def test_other_lines_unchanged():
    for a in (("QQQ", "etf", "scalp"), ("XAU/USD", "metal", "rsi2"), ("EUR/USD", "fx", "rsi2")):
        assert make_cfg(*a).htf_mult == 0


if __name__ == "__main__":
    for f in (test_causal, test_swings_confirmed_late, test_filter_blocks_counter_trend, test_other_lines_unchanged):
        f()
    print("structure tests OK")
