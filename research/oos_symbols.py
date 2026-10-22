import sys; sys.path.insert(0,'.')
from fastbt import *
import numpy as np, pandas as pd
def sig(d,n):
    hh=d.high.rolling(n).max().shift(1); return np.where((d.close>hh)&(d.ema_f>d.ema_s),1,0)
allt=[]
for sym in ('QQQ','SPY','IWM','DIA'):
    df=resample(load(sym),'15min'); d=ind(df); P=Prep(d,15,[('09:35','15:30')],'15:50')
    bh=(d.close.iloc[-1]/d.close.iloc[200]-1)*100
    out=[]
    for n in (12,24,48):
        for s,t in ((0.75,2.0),(1.0,2.0),(1.5,3.0)):
            tt=P.run(sig(d,n),s,t,1.0,allow_short=False); out.append(f"n{n} {s}/{t}: {len(tt)}tr {tt.R.mean():+.3f}R")
            if (n,s,t)==(24,0.75,2.0): tt['sym']=sym; allt.append(tt)
    print(f"{sym} (buy&hold {bh:+.0f}%):", ' | '.join(out))
    # placebo for the chosen config
    rng=np.random.default_rng(1); mask=(d.ema_f>d.ema_s).values; cand=np.where(mask)[0]; rr=[]
    base=P.run(sig(d,24),0.75,2.0,1.0,allow_short=False).R.mean()
    for k in range(200):
        sd=np.zeros(len(d),int); sd[rng.choice(cand,int(len(cand)*.12),replace=False)]=1; rr.append(P.run(sd,0.75,2.0,1.0,allow_short=False).R.mean())
    print(f"   placebo uptrend-random mean {np.mean(rr):+.3f}R, p(chance>=strategy)={np.mean(np.array(rr)>=base):.3f}")
t=pd.concat(allt); print('\nPORTFOLIO n24 0.75/2.0:', stats(t.sort_values('exit_ts')), 'trades/month ~', round(len(t)/12,1))
