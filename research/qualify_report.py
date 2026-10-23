"""Run the live qualification gate offline on the local history for every default watchlist stream."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import watchlist as wl
from backtest import load_history
from config import Params
from qualify import qualify, HISTORY_BARS

extra = [{"symbol": "SPY", "class": "etf", "strategies": ["scalp", "breakout", "rsi2"]},
         {"symbol": "IWM", "class": "etf", "strategies": ["scalp", "breakout", "rsi2"]}]
items = [wl.validate_entry(e) for e in wl.DEFAULT + extra]
for e in items:
    for st in ("scalp", "rsi2", "breakout"):
        cfg = wl.make_cfg(e["symbol"], e["class"], st)
        df = load_history(cfg)
        tag = "*" if st in e["strategies"] else " "
        if df is None:
            print(f"{tag}{cfg.symbol:16s} {st:9s} no local data")
            continue
        r = qualify(df.tail(HISTORY_BARS), cfg, Params())
        print(f"{tag}{cfg.symbol:16s} {st:9s} {cfg.interval:5s} bars={r['bars']:5d} n={r['n']:4d} wr={r['wr']} "
              f"avgR={r['avgR']} pf={r['pf']} tr={r['R_tr']} te={r['R_te']} -> {r['status'].upper()}  {r['reason']}")
