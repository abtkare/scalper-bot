"""v15 diagnostic: real zone signals vs the same stops/targets at random times, with and without costs.
   python research/v15_placebo.py"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.argv=['x','ob']
from dataclasses import replace
import numpy as np, backtest, strategy
exec(open('research/v15_zones.py').read().split('if __name__')[0])
orig = strategy.signals
def check(kw, label, shuffle=False, cost0=False, seed=0):
    rng=np.random.default_rng(seed); allt=[]
    for sym,(c,df,cut) in DATA.items():
        cfg=replace(c,**kw, cost_bps=0.0 if cost0 else c.cost_bps)
        def sig(d,p,s):
            out=orig(d,p,s)
            if shuffle:
                idx=np.flatnonzero(out["side"].values!=0); pos=rng.choice(np.arange(250,len(out)),len(idx),replace=False)
                for col in ("side","sl_dist","tp_dist"):
                    v=out[col].values.copy(); nv=np.zeros_like(v) if col=="side" else np.full_like(v,np.nan)
                    nv[pos]=v[idx]; out[col]=nv
            return out
        backtest.signals=sig
        allt+=run_backtest(df,cfg,p,1e4,slip_bps=0.0)[0]
    backtest.signals=orig
    s=stats(allt); print(f"{label:<40} n={s['n']} R={s['R']:+.3f} wr={s['wr']:.0f}%")
for mode in ("ob","retest","sweep"):
    kw=dict(zone_mode=mode,zone_disp_atr=1.0,zone_entry="reject",zone_tp="rr",zone_rr=1.5,htf_mult=4)
    check(kw,f"{mode} real (costs)"); check(kw,f"{mode} real, NO costs",cost0=True)
    check(kw,f"{mode} RANDOM timing, NO costs",shuffle=True,cost0=True)
