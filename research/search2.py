import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0,'.')
from fastbt import *

def fam_signals(d, fam, a):
    c,o,h,l=d.close,d.open,d.high,d.low
    up=d.ema_f>d.ema_s; dn=d.ema_f<d.ema_s
    if fam=='pullback':
        L=up&(l<=d.ema_f)&(c>d.ema_f)&(c>o)&d.rsi.between(40,65)
        S=dn&(h>=d.ema_f)&(c<d.ema_f)&(c<o)&d.rsi.between(35,60)
    elif fam=='meanrev':   # RSI2 extreme in direction of the 200 EMA trend
        L=(c>d.ema200)&(d.rsi2<a['th'])&(c<d.bb_lo if a['bb'] else True)
        S=(c<d.ema200)&(d.rsi2>100-a['th'])&(c>d.bb_hi if a['bb'] else True)
    elif fam=='fade':      # pure mean reversion, no trend filter
        L=(d.rsi2<a['th'])&(c<d.bb_lo)
        S=(d.rsi2>100-a['th'])&(c>d.bb_hi)
    elif fam=='breakout':  # Donchian N breakout with trend filter
        n=a['n']; hh=h.rolling(n).max().shift(1); ll=l.rolling(n).min().shift(1)
        L=(c>hh)&up; S=(c<ll)&dn
    return np.where(L,1,np.where(S,-1,0))

GR={'pullback':[{}],
    'meanrev':[{'th':t,'bb':b} for t in (5,10,15) for b in (0,1)],
    'fade':[{'th':t} for t in (5,10)],
    'breakout':[{'n':n} for n in (12,24,48)]}
EXITS=[(s,t,mb) for s in (0.75,1.0,1.5,2.0) for t in (0.5,0.75,1.0,1.5,2.0,3.0) for mb in (0,12,36)]
SYMS={"BTC/USD":6,"ETH/USD":6,"SOL/USD":6,"QQQ":1}
rows=[]
if __name__=="__main__":
    import sys as _s
    SYMS={"BTC/USD":6,"ETH/USD":6,"SOL/USD":6,"QQQ":1}
    for sym,cost in SYMS.items():
        base=load(sym)
        for tf in ('5min','15min','1h'):
            df=base if tf=='5min' else resample(base,tf)
            if sym=='QQQ' and tf=='1h': continue
            d=ind(df); im={'5min':5,'15min':15,'1h':60}[tf]
            win=[('09:35','15:30')] if sym=='QQQ' else None
            eod='15:50' if sym=='QQQ' else None
            P=Prep(d,im,win,eod)
            for fam,gl in GR.items():
                for a in gl:
                    sd=fam_signals(d,fam,a)
                    for s,t,mb in EXITS:
                        for short in (False,True):
                            tt=P.run(sd,s,t,cost,max_bars=mb,allow_short=short)
                            tr,te=split(tt,d)
                            a1,a2=stats(tr),stats(te)
                            rows.append(dict(sym=sym,tf=tf,fam=fam,arg=str(a),stop=s,tgt=t,mb=mb,short=short,
                                n_tr=a1['n'],R_tr=a1['avgR'],pf_tr=a1['pf'],n_te=a2['n'],R_te=a2['avgR'],pf_te=a2['pf'],
                                n=len(tt),R=stats(tt)['avgR'],ret=stats(tt)['ret'],dd=stats(tt)['dd']))
        print(sym,'done',flush=True)
    pd.DataFrame(rows).to_csv('grid_%s.csv'%'_'.join(a.replace('/','') for a in _s.argv[1:]),index=False)
