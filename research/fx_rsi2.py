import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from fx_search import PX, COST
from scalp_lib import add_scalp_ind, fam_signals
from fastbt import ind, resample, stats, split
from data import load_csv
SY = list(COST)
D = {}
for s in SY:
    b = load_csv(f"../data/{s}_15min.csv")
    D[s] = {tf: add_scalp_ind(ind(b if tf == "15min" else resample(b, tf))) for tf in ("15min", "30min", "1h")}
IM = {"15min": 15, "30min": 30, "1h": 60}

def pooled(tf, win, th, stop, tgt, cost_mult=1.0, short=True, extra=None):
    out = []
    per = {}
    for s in SY:
        d = D[s][tf]; P = PX(d, IM[tf], win, True)
        sd = fam_signals(d, "rsi2", {"th": th})
        t = P.run(sd, stop, tgt, COST[s] * cost_mult, short=short)
        t["sym"] = s; out.append(t); per[s] = stats(t)["avgR"]
    t = pd.concat(out).sort_values("exit_ts")
    return t, per

print("== plateau: pooled avg R (4 symbols), window x threshold, 1h, stop 1.0 tgt 1.5")
for win in (("07:00", "12:00"), ("08:00", "12:00"), ("08:00", "14:00"), ("03:00", "12:00"), ("00:00", "16:15"), ("12:00", "16:15"), ("18:00", "23:59")):
    row = []
    for th in (5, 10, 15, 20):
        t, per = pooled("1h", win, th, 1.0, 1.5)
        s = stats(t); row.append(f"th{th}: n={s['n']} R={s['avgR']:+.3f} wr={s['wr']}")
    print(win, " | ".join(row))
print("\n== timeframe (window 08-12, th10, stop1 tgt1.5)")
for tf in ("15min", "30min", "1h"):
    t, per = pooled(tf, ("08:00", "12:00"), 10, 1.0, 1.5); print(tf, stats(t), per)
print("\n== chosen: 1h, 08:00-12:00, th10, stop 1.0, tgt 1.5")
t, per = pooled("1h", ("08:00", "12:00"), 10, 1.0, 1.5)
print("pooled", stats(t)); print("per symbol", per)
cut = t.exit_ts.quantile(0.6)
print("train", stats(t[t.exit_ts < cut]), "\ntest ", stats(t[t.exit_ts >= cut]))
m = t.groupby(t.exit_ts.dt.to_period("M")).R
print(pd.DataFrame({"n": m.size(), "sumR": m.sum().round(2), "wr": m.apply(lambda r: round((r > 0).mean()*100))}))
print("long/short:", t.groupby("side").R.agg(["size", "mean"]).round(3).to_dict())
print("reasons:", t.reason.value_counts().to_dict())
print("\n== cost sensitivity (x baseline cost)")
for cm in (0, 1, 2, 3, 4):
    print(cm, stats(pooled("1h", ("08:00", "12:00"), 10, 1.0, 1.5, cm)[0]))
print("\n== placebo (random entries, same window/counts/exits)")
rng = np.random.default_rng(11); res = []
real = stats(t)["avgR"]
for _ in range(200):
    out = []
    for s in SY:
        d = D[s]["1h"]; P = PX(d, 60, ("08:00", "12:00"), True)
        sd = fam_signals(d, "rsi2", {"th": 10}); k = int((sd[200:] != 0).sum())
        cand = np.where(P.inwin)[0]; cand = cand[cand > 200]
        r = np.zeros(len(sd), np.int64); pick = rng.choice(cand, size=min(k, len(cand)), replace=False)
        r[pick] = rng.choice([1, -1], size=len(pick))
        out.append(P.run(r, 1.0, 1.5, COST[s]))
    res.append(stats(pd.concat(out))["avgR"])
res = np.array(res)
print("placebo mean", res.mean().round(3), "p95", np.percentile(res, 95).round(3), "real", real, "p", (res >= real).mean())
