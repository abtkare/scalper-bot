"""Win-rate vs expectancy frontier: same entries, different stop/target ratios.
Shows that win rate is mostly a function of target/stop geometry, not of skill."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from fastbt import load, resample, ind, Prep, stats, split

base = load("QQQ")
d15 = ind(resample(base, "15min"))
P = Prep(d15, 15, [("09:45", "15:30")], "15:45")
hh = d15.high.rolling(24).max().shift(1)
sig = np.where((d15.ema_f > d15.ema_s) & (d15.close > hh), 1, 0)
rng = np.random.default_rng(1)
rows = []
for stop in (0.75, 1.5, 3.0, 5.0):
    for tgt in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
        t = P.run(sig, stop, tgt, 1.0, allow_short=False, max_trades=10, max_consec=99, daily_cap_R=99)
        s = stats(t)
        rows.append(dict(stop_ATR=stop, target_ATR=tgt, trades=s["n"], win_rate=s["wr"], avg_R=s["avgR"], PF=s["pf"], ret_pct=s["ret"]))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("out_frontier.csv", index=False)
