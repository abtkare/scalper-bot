import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scalp_lib import *

sym = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
cost = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
base = load(sym)
GR = {"rsi2": [{"th": t} for t in (5, 10, 20)], "bb": [{}], "vwap": [{"k": k} for k in (0.5, 1.0, 1.5)],
      "down3": [{"n": n} for n in (3, 4)], "rsi2bb": [{"th": 10}]}
rows = []
for tf, im in (("5min", 5), ("15min", 15)):
    df = base if tf == "5min" else resample(base, tf)
    d = add_scalp_ind(ind(df))
    P = P2(d, im, [("09:40", "15:30")], "15:45")
    XS = {k: exit_signal(d, k) for k in ("none", "ema5", "rsi2_70", "vwap")}
    for fam, gl in GR.items():
        for a in gl:
            sd = fam_signals(d, fam, a)
            for stop in (1.0, 1.5, 2.0, 3.0):
                for tgt in (0.3, 0.5, 0.75, 1.0, 1.5):
                    for mb in (0, 3, 6, 12):
                        for xk, xs in XS.items():
                            for short in (False, True):
                                t = P.run(sd, stop, tgt, cost, max_bars=mb, xs=xs, allow_short=short)
                                if len(t) < 30:
                                    continue
                                tr, te = split(t, d)
                                a0, a1, a2 = stats(t), stats(tr), stats(te)
                                rows.append(dict(tf=tf, fam=fam, arg=str(a), stop=stop, tgt=tgt, mb=mb, xexit=xk, short=short,
                                                 n=a0["n"], wr=a0["wr"], R=a0["avgR"], pf=a0["pf"], ret=a0["ret"], dd=a0["dd"],
                                                 wr_tr=a1["wr"], R_tr=a1["avgR"], wr_te=a2["wr"], R_te=a2["avgR"], n_te=a2["n"]))
    print(tf, "done", len(rows), flush=True)
out = pd.DataFrame(rows)
out.to_csv(f"scalp_grid_{sym}.csv", index=False)
print("configs:", len(out))
print("positive both halves:", ((out.R_tr > 0) & (out.R_te > 0)).sum())
print("\n-- win rate >= 80% (any expectancy):", (out.wr >= 80).sum(), " of which positive both halves:",
      ((out.wr >= 80) & (out.R_tr > 0) & (out.R_te > 0)).sum())
print("-- win rate >= 90%:", (out.wr >= 90).sum(), " positive both:", ((out.wr >= 90) & (out.R_tr > 0) & (out.R_te > 0)).sum())
print("-- win rate >= 95%:", (out.wr >= 95).sum(), " positive both:", ((out.wr >= 95) & (out.R_tr > 0) & (out.R_te > 0)).sum())
good = out[(out.R_tr > 0) & (out.R_te > 0)].sort_values("wr", ascending=False)
print("\nTop by win rate among configs positive in BOTH halves:")
print(good.head(30).to_string(index=False))
print("\nTop by avg R (both halves positive, n>=150):")
print(good[good.n >= 150].sort_values("R", ascending=False).head(20).to_string(index=False))
