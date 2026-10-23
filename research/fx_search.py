"""Grid search on gold + FX (15 min / 1 h, Twelve Data, NY time, weekends removed).
Train = first 60%, test = last 40%. Per-side cost in bps of price (assumption for an ECN/cTrader account)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scalp_lib import core, REASONS, fam_signals, exit_signal, add_scalp_ind
from fastbt import ind, resample, stats, split
from data import load_csv

COST = {"XAU-USD": 0.5, "EUR-USD": 0.5, "GBP-USD": 0.6, "USD-JPY": 0.5}
WINS = {"all": ("00:00", "16:15"), "ldn_ny": ("03:00", "12:00"), "ny": ("08:00", "12:00")}


class PX:
    def __init__(self, d, itv_min, win, daily_flat):
        self.d = d
        self.o, self.h, self.l, self.c, self.atr = (d[k].values.astype(float) for k in ("open", "high", "low", "close", "atr"))
        idx = d.index
        hhmm = idx.strftime("%H:%M").values
        dow = idx.dayofweek.values
        g = np.diff(idx.values).astype("timedelta64[s]").astype(float)
        self.gapok = np.append(g <= itv_min * 60 * 1.5, False)
        a, b = win
        self.inwin = (hhmm >= a) & (hhmm <= b) & (dow != 6)          # no entries on the Sunday open
        last = (idx + pd.Timedelta(minutes=itv_min)).strftime("%H:%M").values
        fri = (dow == 4) & (last >= "16:45")
        self.eodb = (fri | ((last >= "16:45") & daily_flat) | ~self.gapok)
        self.dayid = (idx - pd.Timedelta(hours=17)).normalize().values.astype("datetime64[D]").astype(np.int64)
        self.idx = idx

    def run(self, side, stop, tgt, cost, mb=0, xs=None, short=True, warmup=200):
        sd = np.asarray(side, dtype=np.int64).copy()
        if not short:
            sd[sd < 0] = 0
        sd[:warmup] = 0
        xs = np.zeros(len(sd), np.int64) if xs is None else xs
        i0, i1, s, R, k = core(self.o, self.h, self.l, self.c, self.atr, sd, xs, self.gapok, self.inwin, self.eodb,
                               self.dayid, float(stop), float(tgt), float(cost), int(mb), 10, 3, 6.0)
        return pd.DataFrame({"entry_ts": self.idx[i0], "exit_ts": self.idx[i1], "side": s, "R": R,
                             "reason": REASONS[k] if len(k) else []})


def sigs(d, fam, a):
    if fam == "breakout":
        n = a["n"]; hh = d.high.rolling(n).max().shift(1); ll = d.low.rolling(n).min().shift(1)
        up, dn = d.ema_f > d.ema_s, d.ema_f < d.ema_s
        return np.where(up & (d.close > hh), 1, np.where(dn & (d.close < ll), -1, 0)).astype(np.int64)
    if fam == "orb":      # London-open range breakout: range 00:00-03:00 NY, trade the break
        day = d.index.normalize(); hh_ = d.index.strftime("%H:%M")
        rng = d[(hh_ < "03:00")]
        hi = rng.high.groupby(rng.index.normalize()).max().reindex(day).values
        lo = rng.low.groupby(rng.index.normalize()).min().reindex(day).values
        after = (hh_ >= "03:00") & (hh_ <= "10:00")
        c = d.close.values; pc = d.close.shift(1).values
        L = after & (c > hi) & (pc <= hi); S = after & (c < lo) & (pc >= lo)
        return np.where(L, 1, np.where(S, -1, 0)).astype(np.int64)
    return fam_signals(d, fam, a)


GR = {"breakout": [{"n": n} for n in (12, 24, 48)], "orb": [{}],
      "rsi2": [{"th": t} for t in (5, 10)], "bb": [{}], "vwap": [{"k": k} for k in (1.0, 1.5)],
      "down3": [{"n": 3}]}

if __name__ == "__main__":
    rows = []
    for sym, cost in COST.items():
        base = load_csv(f"../data/{sym}_15min.csv")
        for tf, im in (("15min", 15), ("1h", 60)):
            df = base if tf == "15min" else resample(base, "1h")
            d = add_scalp_ind(ind(df))
            for wn, w in WINS.items():
                for daily_flat in (True, False):
                    P = PX(d, im, w, daily_flat)
                    for fam, gl in GR.items():
                        if fam == "orb" and tf == "1h":
                            continue
                        for a in gl:
                            sd = sigs(d, fam, a)
                            for stop in (0.75, 1.0, 1.5, 2.0):
                                for tgt in (0.5, 0.75, 1.0, 1.5, 2.5):
                                    for mb in (0, 8):
                                        for short in (True, False):
                                            t = P.run(sd, stop, tgt, cost, mb, short=short)
                                            if len(t) < 40:
                                                continue
                                            tr, te = split(t, d)
                                            a0, a1, a2 = stats(t), stats(tr), stats(te)
                                            rows.append(dict(sym=sym, tf=tf, win=wn, dflat=daily_flat, fam=fam, arg=str(a),
                                                             stop=stop, tgt=tgt, mb=mb, short=short, n=a0["n"], wr=a0["wr"],
                                                             R=a0["avgR"], pf=a0["pf"], ret=a0["ret"], dd=a0["dd"],
                                                             R_tr=a1["avgR"], R_te=a2["avgR"], wr_te=a2["wr"], n_te=a2["n"]))
        print(sym, "done", len(rows), flush=True)
    pd.DataFrame(rows).to_csv("fx_grid.csv", index=False)
