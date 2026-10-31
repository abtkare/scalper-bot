"""Institutional supply / demand zones (v15) - "trade WITH the institutions".

How a discretionary day trader reads it, turned into causal code (nothing looks into the future):

  1. Footprint: a DISPLACEMENT - a strong candle (body >= disp_atr x ATR, or a 3-bar run of that size) that
     closes beyond the last confirmed swing (break of structure). Big money moving price fast leaves this.
     Where the feed has volume (US ETFs) the run must also trade >= vol_mult x the average volume.
  2. Zone: the last OPPOSITE candle before the run = where the orders were placed ("order block").
     Bullish run -> DEMAND zone [low, high] of the last red candle.  Bearish run -> SUPPLY zone of the last green.
  3. Entry: the FIRST time price comes back into the zone (first support / first resistance) while the trend
     (and the higher-timeframe trend) still points the same way, and the candle rejects the zone
     (closes back out of it in the trade direction). One trade per zone; every new break of structure
     creates a new zone -> repeated entries in the trend direction.
  4. Stop just beyond the far edge of the zone (zone invalid = institutions are not defending it).
     Target: tp_mode "rr" = rr x risk; "swing" = the high/low of the displacement run (liquidity), min 1R.

Outputs per bar: long/short signal, sl_dist / tp_dist (price distance from the signal close) and the zones
that are still active (for the trade plan and the chart).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import structure


def find(d: pd.DataFrame, scfg, htf=None):
    """Returns (long_sig, short_sig, sl_dist, tp_dist, active) - `active` = list of live zones at the last bar."""
    o, h, l, c = (d[k].values.astype(float) for k in ("open", "high", "low", "close"))
    atr = d["atr"].values.astype(float)
    n = len(d)
    st = structure.analyze(d, scfg.swing_k)
    ev = st["st_event"].values
    trend = st["st_trend"].values
    vol = d["volume"].values.astype(float) if "volume" in d.columns else np.zeros(n)
    use_vol = scfg.zone_vol_mult > 0 and np.nanmax(vol[-min(n, 500):] if n else [0]) > 0
    vavg = pd.Series(vol).rolling(50, min_periods=20).mean().values
    htf_v = htf.values if htf is not None else np.zeros(n, dtype=int)

    long_sig = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)
    sl = np.full(n, np.nan)
    tp = np.full(n, np.nan)
    zones = []                        # dicts: side, lo, hi, born, peak, used
    run = 3
    swh, swl = st["swing_hi"].values, st["swing_lo"].values
    for i in range(n):
        a = atr[i]
        if not (a > 0 and math.isfinite(a)):
            continue
        if scfg.zone_mode == "sweep":
            # liquidity sweep: the bar runs the stops beyond the last swing (against the trend) and closes back
            if i == 0 or not (math.isfinite(swl[i - 1]) and math.isfinite(swh[i - 1])):
                continue
            if trend[i] >= 0 and htf_v[i] >= 0 and (scfg.htf_mult == 0 or htf_v[i] > 0 or trend[i] > 0) \
                    and l[i] < swl[i - 1] and c[i] > swl[i - 1] and trend[i] == 1:
                risk = c[i] - (l[i] - scfg.zone_buf_atr * a)
                risk = max(risk, scfg.zone_min_risk_atr * a)          # floor: costs must stay small vs the risk
                if risk <= scfg.zone_max_risk_atr * a:
                    long_sig[i], sl[i] = True, risk
                    tp[i] = max(swh[i - 1] - c[i], risk) if scfg.zone_tp == "swing" else scfg.zone_rr * risk
            elif htf_v[i] <= 0 and trend[i] == -1 and h[i] > swh[i - 1] and c[i] < swh[i - 1]:
                risk = (h[i] + scfg.zone_buf_atr * a) - c[i]
                risk = max(risk, scfg.zone_min_risk_atr * a)          # floor: costs must stay small vs the risk
                if risk <= scfg.zone_max_risk_atr * a:
                    short_sig[i], sl[i] = True, risk
                    tp[i] = max(c[i] - swl[i - 1], risk) if scfg.zone_tp == "swing" else scfg.zone_rr * risk
            continue
        # ---- 1) entries on zones created EARLIER (first retest, rejection close)
        keep = []
        for z in zones:
            if i - z["born"] > scfg.zone_max_age:
                continue
            if z["side"] == 1:
                if c[i] < z["lo"]:
                    continue                                   # closed through demand: zone broken
                touched = l[i] <= z["hi"]
                ok_dir = trend[i] >= 0 and htf_v[i] >= 0
                rej = c[i] > z["hi"] if scfg.zone_entry == "reject" else c[i] >= z["lo"]
                if touched and i > z["born"]:
                    if ok_dir and rej and not long_sig[i] and not short_sig[i]:
                        stop = z["lo"] - scfg.zone_buf_atr * a
                        risk = c[i] - stop
                        risk = max(risk, scfg.zone_min_risk_atr * a)          # floor: costs must stay small vs the risk
                        if risk <= scfg.zone_max_risk_atr * a:
                            long_sig[i] = True
                            sl[i] = risk
                            tp[i] = (max(z["peak"] - c[i], risk) if scfg.zone_tp == "swing"
                                     else scfg.zone_rr * risk)
                    continue                                   # first touch consumes the zone either way
            else:
                if c[i] > z["hi"]:
                    continue
                touched = h[i] >= z["lo"]
                ok_dir = trend[i] <= 0 and htf_v[i] <= 0
                rej = c[i] < z["lo"] if scfg.zone_entry == "reject" else c[i] <= z["hi"]
                if touched and i > z["born"]:
                    if ok_dir and rej and not long_sig[i] and not short_sig[i]:
                        stop = z["hi"] + scfg.zone_buf_atr * a
                        risk = stop - c[i]
                        risk = max(risk, scfg.zone_min_risk_atr * a)          # floor: costs must stay small vs the risk
                        if risk <= scfg.zone_max_risk_atr * a:
                            short_sig[i] = True
                            sl[i] = risk
                            tp[i] = (max(c[i] - z["peak"], risk) if scfg.zone_tp == "swing"
                                     else scfg.zone_rr * risk)
                    continue
            keep.append(z)
        zones = keep
        # ---- 2) new footprint on this CLOSED bar -> new zone (usable from the next bar on)
        e = ev[i]
        if e not in ("bos_up", "choch_up", "bos_dn", "choch_dn"):
            continue
        up = e.endswith("_up")
        j0 = max(0, i - run + 1)
        move = (c[i] - o[j0]) if up else (o[j0] - c[i])
        body = (c[i] - o[i]) if up else (o[i] - c[i])
        if not (body >= scfg.zone_disp_atr * a or move >= 1.5 * scfg.zone_disp_atr * a):
            continue
        if use_vol and not (vavg[i] > 0 and vol[j0:i + 1].max() >= scfg.zone_vol_mult * vavg[i]):
            continue
        if scfg.zone_mode == "retest":
            lvl = swh[i - 1] if up else swl[i - 1]           # the swing that was just broken
            if i == 0 or not math.isfinite(lvl):
                continue
            lo, hi = (lvl - 0.3 * a, lvl + 0.1 * a) if up else (lvl - 0.1 * a, lvl + 0.3 * a)
            peak = float(h[j0:i + 1].max()) if up else float(l[j0:i + 1].min())
            zones = [z for z in zones if z["side"] != (1 if up else -1)] if scfg.zone_one_side else zones
            zones.append({"side": 1 if up else -1, "lo": lo, "hi": hi, "born": i, "peak": peak})
            zones = zones[-6:]
            continue
        k = None
        for j in range(i, max(-1, i - 10), -1):              # last opposite candle before / inside the run
            if (up and c[j] < o[j]) or (not up and c[j] > o[j]):
                k = j
                break
        if k is None:
            continue
        lo, hi = float(l[k]), float(h[k])
        if hi - lo <= 0 or hi - lo > 3 * a:
            continue
        peak = float(h[k:i + 1].max()) if up else float(l[k:i + 1].min())
        zones = [z for z in zones if z["side"] != (1 if up else -1)] if scfg.zone_one_side else zones
        zones.append({"side": 1 if up else -1, "lo": lo, "hi": hi, "born": i, "peak": peak})
        zones = zones[-6:]
    active = [{"side": z["side"], "lo": z["lo"], "hi": z["hi"], "age": n - 1 - z["born"], "peak": z["peak"],
               "born": z["born"]} for z in zones]
    idx = d.index
    return (pd.Series(long_sig, index=idx), pd.Series(short_sig, index=idx),
            pd.Series(sl, index=idx), pd.Series(tp, index=idx), active)
