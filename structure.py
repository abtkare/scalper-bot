"""Market structure - how an experienced trader reads the candles, as cheap deterministic code.

Everything here is CAUSAL: a swing high/low only exists once `k` bars have closed after it (that is when a
trader can actually see it on the chart). Nothing looks into the future, so it can be backtested honestly
and run on every bar with negligible CPU.

Per bar it answers:
  trend      +1 up (higher highs + higher lows) / -1 down (lower highs + lower lows) / 0 range
  event      "bos_up"/"bos_dn"   break of structure WITH the trend (trend continues)
             "choch_up"/"choch_dn" change of character AGAINST the trend (possible reversal starts)
  swing_hi / swing_lo   last confirmed swing levels (where a trend leg started / where it should stop)
  room_up / room_dn     distance (in ATR) from the close to the next opposite swing = room for a target
  stretch    distance from EMA20 in ATR (large = extended leg, likely to pause)
  leg_atr    size of the current leg in ATR since the last opposite swing (trend age / exhaustion)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def swings(high: np.ndarray, low: np.ndarray, k: int):
    """Fractal pivots. Returns arrays (value at the bar where the pivot BECOMES visible, NaN elsewhere)."""
    n = len(high)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(k, n - k):
        w_h = high[i - k:i + k + 1]
        w_l = low[i - k:i + k + 1]
        if high[i] == w_h.max() and (w_h == high[i]).sum() == 1:
            sh[i + k] = high[i]                     # confirmed k bars later
        if low[i] == w_l.min() and (w_l == low[i]).sum() == 1:
            sl[i + k] = low[i]
    return sh, sl


def analyze(d: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Adds structure columns to a frame that already has open/high/low/close/atr/ema_f."""
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    atr = d["atr"].values if "atr" in d.columns else (d["high"] - d["low"]).rolling(14).mean().values
    sh, sl = swings(h, l, k)
    n = len(d)
    trend = np.zeros(n, dtype=int)
    event = np.array([""] * n, dtype=object)
    last_hi = np.full(n, np.nan)
    last_lo = np.full(n, np.nan)
    leg = np.zeros(n)

    hi1 = hi0 = lo1 = lo0 = np.nan          # last and previous confirmed swing high / low
    t = 0
    broke_hi = broke_lo = False             # each swing level can be "broken" once
    for i in range(n):
        if not np.isnan(sh[i]):
            hi0, hi1, broke_hi = hi1, sh[i], False
        if not np.isnan(sl[i]):
            lo0, lo1, broke_lo = lo1, sl[i], False
        # swing sequence -> trend
        if not np.isnan(hi0) and not np.isnan(lo0):
            if hi1 > hi0 and lo1 > lo0:
                t = 1
            elif hi1 < hi0 and lo1 < lo0:
                t = -1
        # a CLOSE beyond the last swing = break (continuation if with trend, change of character if against)
        if not np.isnan(hi1) and not broke_hi and c[i] > hi1:
            broke_hi = True
            event[i] = "bos_up" if t >= 0 else "choch_up"
            t = 1
        elif not np.isnan(lo1) and not broke_lo and c[i] < lo1:
            broke_lo = True
            event[i] = "bos_dn" if t <= 0 else "choch_dn"
            t = -1
        trend[i] = t
        last_hi[i], last_lo[i] = hi1, lo1
        a = atr[i] if atr[i] and not np.isnan(atr[i]) else np.nan
        if t > 0 and not np.isnan(lo1):
            leg[i] = (c[i] - lo1) / a
        elif t < 0 and not np.isnan(hi1):
            leg[i] = (hi1 - c[i]) / a

    out = d.copy()
    out["st_trend"] = trend
    out["st_event"] = event
    out["swing_hi"] = last_hi
    out["swing_lo"] = last_lo
    out["room_up"] = (last_hi - c) / atr          # negative = already above the last swing high
    out["room_dn"] = (c - last_lo) / atr
    out["stretch"] = (c - d["ema_f"].values) / atr if "ema_f" in d.columns else np.nan
    out["leg_atr"] = leg
    return out


def htf_trend(d: pd.DataFrame, minutes: int, k: int = 3) -> pd.Series:
    """Higher-timeframe structure trend mapped onto the lower-timeframe bars - only from HTF bars that have
    already CLOSED (top-down reading: the bigger picture decides the direction)."""
    o = d.resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    pc = o["close"].shift(1)
    tr = pd.concat([o["high"] - o["low"], (o["high"] - pc).abs(), (o["low"] - pc).abs()], axis=1).max(axis=1)
    o["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    s = analyze(o, k)["st_trend"]
    s.index = s.index + pd.Timedelta(minutes=minutes)        # usable only after the HTF bar closes
    return s.reindex(d.index, method="ffill").fillna(0).astype(int)


def describe(row) -> str:
    """One-line Arabic reading of the current structure (for the dashboard / alerts)."""
    tr = {1: "اتجاه صاعد (قمم وقيعان أعلى)", -1: "اتجاه هابط (قمم وقيعان أدنى)", 0: "تذبذب بلا اتجاه"}[int(row["st_trend"])]
    ev = {"bos_up": " · كسر قمة مع الاتجاه (استمرار)", "bos_dn": " · كسر قاع مع الاتجاه (استمرار)",
          "choch_up": " · تغيّر الطابع لأعلى (بداية انعكاس محتمل)",
          "choch_dn": " · تغيّر الطابع لأسفل (بداية انعكاس محتمل)"}.get(row["st_event"], "")
    return f"{tr}{ev} · امتداد الموجة {row['leg_atr']:.1f} ATR"
