"""Fast R-unit backtester mirroring engine.py semantics (signal on bar close, entry next open,
stop/target from wicks, stop first if both touched, per-side cost, daily limits).
Adds optional time-stop (max_bars) and trailing nothing. Used for research only."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "..")


def load(sym, itv="5min"):
    from data import load_csv, csv_path
    import os
    return load_csv(os.path.join("..", csv_path(sym, itv)))


def resample(df, rule):
    o = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return o.dropna()


def ind(df, ema_f=20, ema_s=50, rsi_len=14, atr_len=14):
    d = df.copy()
    c = d.close
    d["ema_f"] = c.ewm(span=ema_f, adjust=False).mean()
    d["ema_s"] = c.ewm(span=ema_s, adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    def rsi(n):
        up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        return (100 - 100 / (1 + up / dn)).fillna(50)
    d["rsi"] = rsi(rsi_len)
    d["rsi2"] = rsi(2)
    pc = c.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / atr_len, adjust=False).mean()
    m = c.rolling(20).mean(); s = c.rolling(20).std()
    d["bb_lo"], d["bb_hi"], d["bb_mid"] = m - 2 * s, m + 2 * s, m
    return d


def run(d, side, stop_atr, tgt_atr, cost_bps, max_bars=0, windows=None, flatten_eod=None,
        max_trades=10, max_consec=3, daily_cap_R=6.0, itv_min=5, warmup=200, min_tgt_cost_mult=0.0,
        allow_short=True):
    o, h, l, c, atr = (d[k].values for k in ("open", "high", "low", "close", "atr"))
    idx = d.index
    hhmm = idx.strftime("%H:%M").values
    days = idx.normalize().values
    sd = np.asarray(side)
    if not allow_short:
        sd = np.where(sd < 0, 0, sd)
    if min_tgt_cost_mult > 0:
        worth = tgt_atr * atr / c * 1e4 >= min_tgt_cost_mult * 2 * cost_bps
        sd = np.where(worth, sd, 0)
    sd[:warmup] = 0
    gap = np.diff(idx.values).astype("timedelta64[s]").astype(float)
    maxgap = itv_min * 60 * 1.5
    trades = []
    pos = None
    day = None; nt = cl = 0; dayR = 0.0
    for i in range(1, len(d)):
        if days[i] != day:
            day = days[i]; nt = cl = 0; dayR = 0.0
        # entry
        if pos is None and sd[i - 1] != 0 and gap[i - 1] <= maxgap and atr[i - 1] > 0:
            ok = nt < max_trades and cl < max_consec and dayR > -daily_cap_R
            if ok and windows:
                ok = any(a <= hhmm[i] <= b for a, b in windows)
            if ok:
                s = int(sd[i - 1]); e = o[i]; sdist = stop_atr * atr[i - 1]
                pos = [s, e, e - s * sdist, e + s * tgt_atr * atr[i - 1], sdist, i]
                nt += 1
        if pos is None:
            continue
        s, e, st, tg, sdist, i0 = pos
        ex = None
        if s == 1:
            if l[i] <= st: ex = (min(st, o[i]) if i > i0 else st, "stop")
            elif h[i] >= tg: ex = (tg, "target")
        else:
            if h[i] >= st: ex = (max(st, o[i]) if i > i0 else st, "stop")
            elif l[i] <= tg: ex = (tg, "target")
        if ex is None and max_bars and i - i0 + 1 >= max_bars:
            ex = (c[i], "time")
        if ex is None and flatten_eod and hhmm[i] >= flatten_eod:
            ex = (c[i], "eod")
        if ex is None and i + 1 < len(d) and gap[i] > maxgap and flatten_eod:
            ex = (c[i], "eod")
        if ex:
            x = ex[0]
            R = (s * (x - e) - (e + x) * cost_bps / 1e4) / sdist
            trades.append((idx[i0], idx[i], s, R, ex[1]))
            dayR += R
            cl = cl + 1 if R < 0 else 0
            pos = None
    return pd.DataFrame(trades, columns=["entry_ts", "exit_ts", "side", "R", "reason"])


def stats(t, R_pct=0.005):
    if len(t) == 0:
        return dict(n=0, avgR=0, pf=0, wr=0, ret=0, dd=0)
    r = t.R.values
    eq = np.cumprod(1 + r * R_pct)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    gl = -r[r < 0].sum()
    return dict(n=len(r), avgR=round(r.mean(), 3), pf=round(r[r > 0].sum() / gl, 2) if gl else 99,
                wr=round((r > 0).mean() * 100, 1), ret=round((eq[-1] - 1) * 100, 1), dd=round(dd * 100, 1))


def split(t, d, frac=0.6):
    cut = d.index[int(len(d) * frac)]
    return t[t.exit_ts < cut], t[t.exit_ts >= cut]


# ---------------------------------------------------------------- compiled core (numba)
try:
    from numba import njit
except Exception:                       # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]


@njit(cache=True)
def _core(o, h, l, c, atr, sd, gapok, inwin, eodb, dayid, stop_atr, tgt_atr, cost_bps, max_bars,
          max_trades, max_consec, daily_cap_R):
    n = len(o)
    out_i0 = np.empty(n, np.int64); out_i1 = np.empty(n, np.int64)
    out_s = np.empty(n, np.int64); out_R = np.empty(n); out_k = np.empty(n, np.int64)
    k = 0
    inpos = False
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
                x = tg; kind = 1
        else:
            if h[i] >= st:
                x = max(st, o[i]) if i > i0 else st; kind = 0
            elif l[i] <= tg:
                x = tg; kind = 1
        if kind < 0 and max_bars > 0 and i - i0 + 1 >= max_bars:
            x = c[i]; kind = 2
        if kind < 0 and eodb[i]:
            x = c[i]; kind = 3
        if kind >= 0:
            R = (s * (x - e) - (e + x) * cost_bps / 1e4) / sdist
            out_i0[k] = i0; out_i1[k] = i; out_s[k] = s; out_R[k] = R; out_k[k] = kind; k += 1
            dayR += R
            cl = cl + 1 if R < 0 else 0
            inpos = False
    return out_i0[:k], out_i1[:k], out_s[:k], out_R[:k], out_k[:k]


class Prep:
    """Pre-computed arrays for one (symbol, timeframe) so each run is just the compiled loop."""
    def __init__(self, d, itv_min, windows=None, flatten_eod=None):
        self.d = d
        self.o, self.h, self.l, self.c, self.atr = (d[k].values.astype(float) for k in ("open", "high", "low", "close", "atr"))
        idx = d.index
        hhmm = idx.strftime("%H:%M").values
        g = np.diff(idx.values).astype("timedelta64[s]").astype(float)
        self.gapok = np.append(g <= itv_min * 60 * 1.5, False)
        self.inwin = np.ones(len(d), bool) if not windows else np.array(
            [any(a <= x <= b for a, b in windows) for x in hhmm])
        if flatten_eod:
            self.eodb = (hhmm >= flatten_eod) | ~self.gapok
        else:
            self.eodb = np.zeros(len(d), bool)
        self.dayid = (idx.normalize().values.astype("datetime64[D]").astype(np.int64))
        self.idx = idx

    def run(self, side, stop_atr, tgt_atr, cost_bps, max_bars=0, allow_short=True, warmup=200,
            max_trades=10, max_consec=3, daily_cap_R=6.0):
        sd = np.asarray(side, dtype=np.int64).copy()
        if not allow_short:
            sd[sd < 0] = 0
        sd[:warmup] = 0
        i0, i1, s, R, k = _core(self.o, self.h, self.l, self.c, self.atr, sd, self.gapok, self.inwin,
                                self.eodb, self.dayid, float(stop_atr), float(tgt_atr), float(cost_bps),
                                int(max_bars), int(max_trades), int(max_consec), float(daily_cap_R))
        return pd.DataFrame({"entry_ts": self.idx[i0], "exit_ts": self.idx[i1], "side": s, "R": R,
                             "reason": np.array(["stop", "target", "time", "eod"])[k] if len(k) else []})
