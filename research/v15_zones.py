"""v15 research: institutional supply/demand zones (zones.py).
Settings are CHOSEN on the first 60% of every history (pooled over all instruments) and JUDGED on the last 40%.
Then: slippage stress and a random-entry placebo with the same stops/targets.  python research/v15_zones.py"""
import itertools, os, sys
from dataclasses import replace
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SymbolCfg, get_params
from backtest import load_history, run_backtest, summarize


def etf(t, iv=15):
    return SymbolCfg(symbol=f"{t}-Z{iv}", provider_symbol=t, kind="stock", tz="America/New_York",
                     interval=f"{iv}min", interval_minutes=iv, strategy="zone", allow_short=True, cost_bps=1.0,
                     flatten_eod=True, eod_time="15:45", windows=[("09:45", "15:30")])


def fx(t, iv, cost):
    return SymbolCfg(symbol=f"{t}-Z{iv}", provider_symbol=t, kind="fx", tz="America/New_York",
                     interval="1h" if iv == 60 else f"{iv}min", interval_minutes=iv, strategy="zone",
                     allow_short=True, cost_bps=cost, flatten_eod=True, eod_time="16:45", windows=[("03:00", "12:00")])


UNIVERSE = [etf("QQQ"), etf("QQQ", 5), etf("SPY"), etf("IWM"), etf("DIA"),
            fx("XAU/USD", 15, 0.5), fx("XAU/USD", 60, 0.5), fx("EUR/USD", 15, 0.5), fx("EUR/USD", 60, 0.5),
            fx("GBP/USD", 15, 0.6), fx("GBP/USD", 60, 0.6), fx("USD/JPY", 15, 0.5), fx("USD/JPY", 60, 0.5)]
MODE = sys.argv[1] if len(sys.argv) > 1 else "ob"
GRID = dict(zone_mode=[MODE], zone_disp_atr=[1.0, 1.5], zone_min_risk_atr=[0.75, 1.25],
            tp=[("rr", 1.5), ("rr", 2.5)], htf_mult=[0, 4])
p = get_params(); p.initial_equity = 1e4
DATA = {}
for c in UNIVERSE:
    df = load_history(c)
    if df is not None:
        DATA[c.symbol] = (c, df, df.index[int(len(df) * 0.6)])


def run(cfg, df, slip=None):
    return run_backtest(df, cfg, p, 1e4, slip_bps=slip)[0]


def split(tr, cut):
    return [t for t in tr if t["exit_ts"] < cut], [t for t in tr if t["exit_ts"] >= cut]


def stats(tr):
    if not tr:
        return dict(n=0, R=0.0, wr=0.0, pf=0.0)
    R = np.array([t["R"] for t in tr]); pn = np.array([t["pnl"] for t in tr])
    gl = -pn[pn < 0].sum()
    return dict(n=len(tr), R=R.mean(), wr=100 * (pn > 0).mean(), pf=pn[pn > 0].sum() / gl if gl > 0 else 9.9)


if __name__ == "__main__":
    combos = list(itertools.product(*GRID.values()))
    rows = []
    for mode, disp, mr, (tpm, rr), htf in combos:
        kw = dict(zone_mode=mode, zone_disp_atr=disp, zone_min_risk_atr=mr, zone_entry="reject", zone_tp=tpm,
                  zone_rr=rr, htf_mult=htf)
        per = {}
        for sym, (c, df, cut) in DATA.items():
            per[sym] = split(run(replace(c, **kw), df), cut)
        tr_all = [t for a, b in per.values() for t in a]
        te_all = [t for a, b in per.values() for t in b]
        rows.append((kw, stats(tr_all), stats(te_all), per))
        s1, s2 = stats(tr_all), stats(te_all)
        print(f"{disp} minrisk{mr} {tpm}{rr} htf{htf}: TRAIN n={s1['n']} R={s1['R']:+.3f} wr={s1['wr']:.0f}% | "
              f"TEST n={s2['n']} R={s2['R']:+.3f} wr={s2['wr']:.0f}%", flush=True)
    best = max(rows, key=lambda r: r[1]["R"] if r[1]["n"] >= 150 else -9)
    kw, s1, s2, per = best
    print("\n=== chosen on TRAIN only:", kw, f"train {s1['R']:+.3f}R  ->  test {s2['R']:+.3f}R (n={s2['n']})")
    print("per instrument (train / test avg R, n):")
    for sym, (a, b) in per.items():
        x, y = stats(a), stats(b)
        print(f"  {sym:<14} {x['R']:+.3f} ({x['n']}) / {y['R']:+.3f} ({y['n']})  wr {y['wr']:.0f}%")
    import json; json.dump(kw, open("/tmp/claude-0/-home-claude/e83e8d7e-133a-5685-84e6-1fca06a1b27c/scratchpad/zone_best_{MODE}.json", "w"))
