"""Strategies (signal on a CLOSED bar, order at the NEXT bar's open - nothing looks into the future):

  pullback : EMA20/50 pullback (original v2, lost money after costs - kept for study)
  breakout : Donchian breakout in the EMA trend direction (v3, QQQ 15 min)
  scalp    : VWAP mean reversion (v4 scalping mode, QQQ 15 min, long + short) - see below
  rsi2     : RSI(2) mean reversion with EMA200 trend filter (v5, gold + FX, 1 h, New York morning)

Pullback details:

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
    up2 = up.ewm(alpha=0.5, adjust=False).mean()
    dn2 = dn.ewm(alpha=0.5, adjust=False).mean()
    d["rsi2"] = (100.0 - 100.0 / (1.0 + up2 / dn2)).fillna(50.0)

    pc = c.shift(1)
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1
    ).max(axis=1)
    d["atr"] = tr.ewm(alpha=1.0 / p.atr_len, adjust=False).mean()

    # scalp mode: long-term trend + session VWAP (resets every calendar day of the bar timestamps)
    d["ema200"] = c.ewm(span=200, adjust=False).mean()
    vol = d["volume"].astype(float).fillna(0.0) if "volume" in d.columns else pd.Series(0.0, index=d.index)
    day = d.index.normalize()
    tp = (d["high"] + d["low"] + c) / 3.0
    cum_v = vol.groupby(day).cumsum()
    cum_pv = (tp * vol).groupby(day).cumsum()
    # no volume (some feeds) -> fall back to the session's running average typical price
    avg_tp = tp.groupby(day).cumsum() / tp.groupby(day).cumcount().add(1)
    d["vwap"] = (cum_pv / cum_v.where(cum_v > 0)).fillna(avg_tp)
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
    elif scfg.strategy == "scalp":
        long_sig, short_sig = _scalp(d, scfg)
    elif scfg.strategy == "rsi2":
        long_sig, short_sig = _rsi2(d, scfg)
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


def _scalp(d, scfg):
    """VWAP mean reversion (scalping mode).
    Long : price stretched more than k*ATR BELOW the session VWAP while above the EMA200 (dip in an uptrend).
    Short: price stretched more than k*ATR ABOVE the session VWAP while below the EMA200 (pop in a downtrend).
    Exit (engine): small target (target_atr) / wider stop (stop_atr) / end-of-day flatten."""
    k = scfg.vwap_k
    c = d["close"]
    long_sig = (c > d["ema200"]) & (c < d["vwap"] - k * d["atr"])
    short_sig = (c < d["ema200"]) & (c > d["vwap"] + k * d["atr"])
    return long_sig, short_sig


def _rsi2(d, scfg):
    """RSI(2) mean reversion (gold / FX, 1 h). Buy a sharp 1-2 bar dip in an uptrend, sell a sharp pop in a
    downtrend. Entry hours (New York morning) and the daily flat come from the symbol config."""
    th = scfg.rsi2_th
    c = d["close"]
    long_sig = (c > d["ema200"]) & (d["rsi2"] < th)
    short_sig = (c < d["ema200"]) & (d["rsi2"] > 100.0 - th)
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
    if scfg.strategy in ("scalp", "rsi2"):
        warmup = max(warmup, 200)                     # EMA200 needs history before it means anything
    out.iloc[:warmup, out.columns.get_loc("side")] = 0
    return out
