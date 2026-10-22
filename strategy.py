"""Strategy: EMA-pullback scalping on 5-minute bars.

Long : EMA20 > EMA50 (uptrend), bar dips to/below EMA20 but closes back above it
       on a green bar, RSI(14) in a non-extreme band.
Short: mirror image (only used when allow_short=True).

A signal is evaluated on a CLOSED bar; the order is executed at the NEXT bar's
open. Nothing here looks into the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import Params, SymbolCfg


def add_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    d = df.copy()
    c = d["close"]
    d["ema_f"] = c.ewm(span=p.ema_fast, adjust=False).mean()
    d["ema_s"] = c.ewm(span=p.ema_slow, adjust=False).mean()

    delta = c.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    a = 1.0 / p.rsi_len
    rs = up.ewm(alpha=a, adjust=False).mean() / dn.ewm(alpha=a, adjust=False).mean()
    d["rsi"] = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

    pc = c.shift(1)
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1
    ).max(axis=1)
    d["atr"] = tr.ewm(alpha=1.0 / p.atr_len, adjust=False).mean()
    return d


def stop_target(p: Params, scfg: SymbolCfg):
    return (scfg.stop_atr or p.stop_atr), (scfg.target_atr or p.target_atr)


def signals(d: pd.DataFrame, p: Params, scfg: SymbolCfg) -> pd.DataFrame:
    """Adds column `side`: +1 long setup, -1 short setup, 0 none (on bar close)."""
    up_trend = d["ema_f"] > d["ema_s"]
    dn_trend = d["ema_f"] < d["ema_s"]
    if scfg.strategy == "breakout":
        # Donchian breakout in the direction of the EMA trend. Uses only CLOSED bars (shift(1)).
        hh = d["high"].rolling(scfg.donchian_n).max().shift(1)
        ll = d["low"].rolling(scfg.donchian_n).min().shift(1)
        long_sig = up_trend & (d["close"] > hh)
        short_sig = dn_trend & (d["close"] < ll)
    else:
        long_sig, short_sig = _pullback(d, p, up_trend, dn_trend)
    return _finish(d, p, scfg, long_sig, short_sig)


def _pullback(d, p, up_trend, dn_trend):
    long_sig = (
        up_trend
        & (d["low"] <= d["ema_f"])
        & (d["close"] > d["ema_f"])
        & (d["close"] > d["open"])
        & d["rsi"].between(p.rsi_min, p.rsi_max)
    )
    short_sig = (
        dn_trend
        & (d["high"] >= d["ema_f"])
        & (d["close"] < d["ema_f"])
        & (d["close"] < d["open"])
        & d["rsi"].between(100.0 - p.rsi_max, 100.0 - p.rsi_min)
    )
    return long_sig, short_sig


def _finish(d, p, scfg, long_sig, short_sig):
    _, target_atr = stop_target(p, scfg)
    # Cost filter: the target must be worth at least N x the round-trip cost.
    target_bps = target_atr * d["atr"] / d["close"] * 1e4
    worth_it = target_bps >= p.min_target_cost_multiple * 2.0 * scfg.cost_bps

    side = np.where(long_sig & worth_it, 1, np.where(short_sig & worth_it, -1, 0))
    out = d.copy()
    out["side"] = side
    warmup = max(p.ema_slow * 2, p.atr_len * 2, p.rsi_len * 2)
    out.iloc[:warmup, out.columns.get_loc("side")] = 0
    return out
