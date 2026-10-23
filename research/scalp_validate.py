import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scalp_lib import *
from data import load_csv

def prep(df):
    d = add_scalp_ind(ind(df)); return d, P2(d, 15, [("09:40", "15:30")], "15:45")

EX = [(1.5, 0.75), (2.0, 0.75), (1.5, 1.0), (2.0, 0.5), (3.0, 0.5)]
dq, Pq = prep(resample(load("QQQ"), "15min"))
print("=== QQQ 15min VWAP reversion k=1.0 long+short, cost 1bps")
for s, t in EX:
    tt = Pq.run(fam_signals(dq, "vwap", {"k": 1.0}), s, t, 1.0)
    print(s, t, stats(tt))

print("\n=== Unseen symbols (same rules, never used in the search)")
for sym in ("SPY", "IWM", "DIA"):
    d, P = prep(load_csv(f"../data/{sym}_15min.csv"))
    for s, t in EX:
        for k in (1.0,):
            tt = P.run(fam_signals(d, "vwap", {"k": k}), s, t, 1.0)
            print(sym, s, t, stats(tt))

print("\n=== QQQ cost sensitivity (stop 2.0, tgt 0.75)")
sd = fam_signals(dq, "vwap", {"k": 1.0})
for c in (0, 0.5, 1, 1.5, 2, 3):
    print(c, stats(Pq.run(sd, 2.0, 0.75, c)))

print("\n=== QQQ by quarter (stop 2.0, tgt 0.75)")
tt = Pq.run(sd, 2.0, 0.75, 1.0)
q = tt.groupby(tt.exit_ts.dt.to_period("Q"))
print(pd.DataFrame({"n": q.R.size(), "sumR": q.R.sum().round(1), "wr": q.R.apply(lambda r: round((r > 0).mean() * 100, 1))}))
print("reasons:", tt.reason.value_counts().to_dict())
hold = (tt.exit_ts - tt.entry_ts).dt.total_seconds() / 60 + 15
print("median holding minutes:", hold.median())

print("\n=== Placebo: random entries same count/side mix, same exits")
rng = np.random.default_rng(7); real = stats(tt)["avgR"]; res = []
cand = np.where(Pq.inwin)[0]
for _ in range(300):
    r = np.zeros(len(sd), np.int64)
    pick = rng.choice(cand, size=(sd != 0).sum(), replace=False)
    r[pick] = rng.choice([1, -1], size=len(pick))
    res.append(stats(Pq.run(r, 2.0, 0.75, 1.0))["avgR"])
res = np.array(res)
print("placebo mean avgR", res.mean().round(3), " real", real, " p(placebo>=real)", (res >= real).mean())
