"""v11 research: (1) slippage sensitivity, (2) trailing stops behind confirmed swings, (3) new index lines.
Adoption rule for a variant: avg R better than the baseline in BOTH halves (first 60% / last 40%)
AND the full-sample profit factor not lower. Run:  python research/v11_exits.py"""
import os, sys
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SYMBOLS, BREAKOUT_QQQ, SCALP_QQQ, get_params
from backtest import load_history, run_backtest, summarize


def halves(df, trades, eq):
    cut = df.index[int(len(df) * 0.6)]
    a = summarize(trades, eq)
    tr = summarize([t for t in trades if t["exit_ts"] < cut], eq)
    te = summarize([t for t in trades if t["exit_ts"] >= cut], eq)
    g = lambda s, k: s.get(k, 0.0) if s.get("trades") else 0.0
    return dict(n=a.get("trades", 0), wr=g(a, "win_rate_%"), R=g(a, "avg_R_per_trade"), pf=g(a, "profit_factor"),
                Rtr=g(tr, "avg_R_per_trade"), Rte=g(te, "avg_R_per_trade"), ret=g(a, "return_%"), dd=g(a, "max_drawdown_%"))


def fmt(name, r):
    return (f"  {name:<34} n={r['n']:>4} wr={r['wr']:>5}% R={r['R']:+.3f} (tr {r['Rtr']:+.3f} / te {r['Rte']:+.3f}) "
            f"PF={r['pf']:<5} ret={r['ret']:+.2f}% dd={r['dd']}%")


p = get_params()
p.initial_equity = 10000.0
VARIANTS = [("trail after 0.5R", dict(trail="swing", trail_after_r=0.5)),
            ("trail after 1.0R", dict(trail="swing", trail_after_r=1.0)),
            ("trail 0R (from start)", dict(trail="swing", trail_after_r=0.0)),
            ("trail 1.0R, no target", dict(trail="swing", trail_after_r=1.0, trail_no_target=True)),
            ("trail 0.5R, no target", dict(trail="swing", trail_after_r=0.5, trail_no_target=True))]

idx = [replace(BREAKOUT_QQQ, symbol=t, provider_symbol=t) for t in ("SPY", "IWM", "DIA")] + \
      [replace(SCALP_QQQ, symbol=f"{t}-SCALP", provider_symbol=t) for t in ("SPY", "IWM", "DIA")]
for cfg in SYMBOLS + idx:
    df = load_history(cfg)
    if df is None:
        continue
    print(f"\n#### {cfg.symbol} ({cfg.strategy}, {cfg.interval}, {len(df)} bars)")
    base = halves(df, run_backtest(df, cfg, p, p.initial_equity, slip_bps=0.0)[0], p.initial_equity)
    print(fmt("baseline", base))
    for s in (0.5, 1.0, 2.0):
        print(fmt(f"baseline + slip {s} bps", halves(df, run_backtest(df, cfg, p, p.initial_equity, slip_bps=s)[0], p.initial_equity)))
    for name, kw in VARIANTS:
        r = halves(df, run_backtest(df, replace(cfg, **kw), p, p.initial_equity, slip_bps=0.0)[0], p.initial_equity)
        ok = r["Rtr"] > base["Rtr"] and r["Rte"] > base["Rte"] and r["pf"] >= base["pf"]
        print(fmt(name, r) + ("   <== better in both halves" if ok else ""))
