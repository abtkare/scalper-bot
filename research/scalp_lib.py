"""Scalping research core: stop / target / time-stop / signal-exit, numba compiled.
Semantics match engine.py: signal on bar close -> entry next bar open; stop first if both touched;
gap through stop filled at the open; per-side cost; daily limits."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from numba import njit
from fastbt import load, resample, ind, stats, split


def add_scalp_ind(d):
    d = d.copy()
    c = d.close
    d["ema5"] = c.ewm(span=5, adjust=False).mean()
    day = d.index.normalize()
    tp = (d.high + d.low + c) / 3
    pv = (tp * d.volume).groupby(day).cumsum()
    vv = d.volume.groupby(day).cumsum().replace(0, np.nan)
    d["vwap"] = (pv / vv).fillna(c)
    return d


@njit(cache=True)
def core(o, h, l, c, atr, sd, xs, gapok, inwin, eodb, dayid, stop_atr, tgt_atr, cost_bps, max_bars,
         max_trades, max_consec, daily_cap_R):
    n = len(o)
    I0 = np.empty(n, np.int64); I1 = np.empty(n, np.int64); S = np.empty(n, np.int64)
    RR = np.empty(n); K = np.empty(n, np.int64)
    k = 0; inpos = False
    s = 0; e = 0.0; st = 0.0; tg = 0.0; sdist = 0.0; i0 = 0
    day = -1; nt = 0; cl = 0; dayR = 0.0
    for i in range(1, n):
        if dayid[i] != day:
            day = dayid[i]; nt = 0; cl = 0; dayR = 0.0
        if (not inpos) and sd[i - 1] != 0 and gapok[i - 1] and atr[i - 1] > 0:
            if nt < max_trades and cl < max_consec and dayR > -daily_cap_R and inwin[i]:
                s = sd[i - 1]; e = o[i]; sdist = stop_atr * atr[i - 1]
                st = e - s * sdist; tg = e + s * tgt_atr * atr[i - 1]; i0 = i
                inpos = True; nt += 1
        if not inpos:
            continue
        x = 0.0; kind = -1
        if s == 1:
            if l[i] <= st:
                x = min(st, o[i]) if i > i0 else st; kind = 0
            elif h[i] >= tg:
                x = max(tg, o[i]) if i > i0 else tg; kind = 1
        else:
            if h[i] >= st:
                x = max(st, o[i]) if i > i0 else st; kind = 0
            elif l[i] <= tg:
                x = min(tg, o[i]) if i > i0 else tg; kind = 1
        if kind < 0 and xs[i] * s > 0:          # signal exit on bar close (xs=+1 exits longs, -1 exits shorts)
            x = c[i]; kind = 4
        if kind < 0 and max_bars > 0 and i - i0 + 1 >= max_bars:
            x = c[i]; kind = 2
        if kind < 0 and eodb[i]:
            x = c[i]; kind = 3
        if kind >= 0:
            R = (s * (x - e) - (e + x) * cost_bps / 1e4) / sdist
            I0[k] = i0; I1[k] = i; S[k] = s; RR[k] = R; K[k] = kind; k += 1
            dayR += R
            cl = cl + 1 if R < 0 else 0
            inpos = False
    return I0[:k], I1[:k], S[:k], RR[:k], K[:k]


REASONS = np.array(["stop", "target", "time", "eod", "signal"])


class P2:
    def __init__(self, d, itv_min, windows=None, flatten_eod=None):
        self.d = d
        self.o, self.h, self.l, self.c, self.atr = (d[k].values.astype(float) for k in ("open", "high", "low", "close", "atr"))
        idx = d.index
        hhmm = idx.strftime("%H:%M").values
        g = np.diff(idx.values).astype("timedelta64[s]").astype(float)
        self.gapok = np.append(g <= itv_min * 60 * 1.5, False)
        self.inwin = np.ones(len(d), bool) if not windows else np.array([any(a <= x <= b for a, b in windows) for x in hhmm])
        self.eodb = ((hhmm >= flatten_eod) | ~self.gapok) if flatten_eod else np.zeros(len(d), bool)
        self.dayid = idx.normalize().values.astype("datetime64[D]").astype(np.int64)
        self.idx = idx

    def run(self, side, stop_atr, tgt_atr, cost_bps, max_bars=0, xs=None, allow_short=True, warmup=200,
            max_trades=10, max_consec=3, daily_cap_R=6.0):
        sd = np.asarray(side, dtype=np.int64).copy()
        if not allow_short:
            sd[sd < 0] = 0
        sd[:warmup] = 0
        xs = np.zeros(len(sd), np.int64) if xs is None else np.asarray(xs, dtype=np.int64)
        i0, i1, s, R, k = core(self.o, self.h, self.l, self.c, self.atr, sd, xs, self.gapok, self.inwin, self.eodb,
                               self.dayid, float(stop_atr), float(tgt_atr), float(cost_bps), int(max_bars),
                               int(max_trades), int(max_consec), float(daily_cap_R))
        return pd.DataFrame({"entry_ts": self.idx[i0], "exit_ts": self.idx[i1], "side": s, "R": R,
                             "reason": REASONS[k] if len(k) else []})


def fam_signals(d, fam, a):
    c, o, h, l = d.close, d.open, d.high, d.low
    trendL = c > d.ema200
    trendS = c < d.ema200
    if fam == "rsi2":
        L = trendL & (d.rsi2 < a["th"]); S = trendS & (d.rsi2 > 100 - a["th"])
    elif fam == "bb":
        L = trendL & (c < d.bb_lo); S = trendS & (c > d.bb_hi)
    elif fam == "vwap":
        L = trendL & (c < d.vwap - a["k"] * d.atr); S = trendS & (c > d.vwap + a["k"] * d.atr)
    elif fam == "down3":
        dn = (c < c.shift(1)); up = (c > c.shift(1))
        n = a["n"]
        L = trendL & dn.rolling(n).sum().eq(n); S = trendS & up.rolling(n).sum().eq(n)
    elif fam == "rsi2bb":
        L = trendL & (d.rsi2 < a["th"]) & (c < d.bb_lo); S = trendS & (d.rsi2 > 100 - a["th"]) & (c > d.bb_hi)
    return np.where(L, 1, np.where(S, -1, 0)).astype(np.int64)


def exit_signal(d, kind):
    if kind == "none":
        return np.zeros(len(d), np.int64)
    if kind == "ema5":       # long exits when close back above EMA5; short when below
        return np.where(d.close > d.ema5, 1, np.where(d.close < d.ema5, -1, 0)).astype(np.int64)
    if kind == "rsi2_70":
        return np.where(d.rsi2 > 70, 1, np.where(d.rsi2 < 30, -1, 0)).astype(np.int64)
    if kind == "vwap":
        return np.where(d.close > d.vwap, 1, np.where(d.close < d.vwap, -1, 0)).astype(np.int64)
    raise ValueError(kind)
