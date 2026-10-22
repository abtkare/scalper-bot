import sys; sys.path.insert(0,'.')
from fastbt import *
import numpy as np
df=resample(load('QQQ'),'15min'); d=ind(df); P=Prep(d,15,[('09:35','15:30')],'15:50')
c,h=d.close,d.high
res={}
for n in (12,24,48):
    hh=h.rolling(n).max().shift(1); sig=np.where((c>hh)&(d.ema_f>d.ema_s),1,0)
    t=P.run(sig,0.75,2.0,1,allow_short=False); res[n]=(len(t),t.R.mean())
    print('breakout n=%d'%n, len(t), round(t.R.mean(),3))
rng=np.random.default_rng(0)
for label,mask in (('random any time',np.ones(len(d),bool)),('random in uptrend (ema20>ema50)',(d.ema_f>d.ema_s).values)):
    out=[]
    for k in range(300):
        sig=np.zeros(len(d),int); cand=np.where(mask)[0]; pick=rng.choice(cand,size=int(len(cand)*0.12),replace=False); sig[pick]=1
        t=P.run(sig,0.75,2.0,1,allow_short=False); out.append(t.R.mean())
    out=np.array(out); print(label,'mean',round(out.mean(),3),'95th pct',round(np.percentile(out,95),3),' p(breakout n=24 beaten by chance)=',round((out>=res[24][1]).mean(),3))
