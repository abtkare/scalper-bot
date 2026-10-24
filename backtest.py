"""Backtest the strategy on historical CSV data.

    python backtest.py QQQ
    python backtest.py --all

Fills: entry at the NEXT bar's open after a signal; stop/target from bar highs/lows
(stop wins if both are touched in one bar); per-side cost from config.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from datetime import time as dtime

import numpy as np
import pandas as pd

from config import SYMBOLS, SymbolCfg, Params, get_params, load_env, symbols_for_mode
from data import csv_path, load_csv
import fx_hours
import market_calendar as cal
from engine import Account, SymbolEngine
from strategy import add_indicators, signals
import structure


def resample_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    o = df.resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return o.dropna()


def load_history(scfg: SymbolCfg):
    """Exact interval file, else build it from 5-min / 15-min history. FX: closed-market bars removed."""
    df = None
    path = csv_path(scfg.ticker, scfg.interval)
    if os.path.exists(path):
        df = load_csv(path)
    else:
        for base in (5, 15):
            f = csv_path(scfg.ticker, f"{base}min")
            if os.path.exists(f) and scfg.interval_minutes % base == 0 and scfg.interval_minutes > base:
                df = load_csv(f)
                break
    if df is None:
        return None
    if scfg.kind == "fx":
        df = df[[fx_hours.bar_is_open(t.to_pydatetime(), scfg.metal) for t in df.index]]
    step = (df.index[1:] - df.index[:-1]).min() if len(df) > 1 else None
    if step is not None and step < timedelta(minutes=scfg.interval_minutes):
        df = resample_bars(df, scfg.interval_minutes)
    return df


def run_backtest(df: pd.DataFrame, scfg: SymbolCfg, p: Params, equity0: float, slip_bps: float = None):
    """slip_bps: adverse slippage on every MARKET fill (entries, stops, eod) on top of cost_bps.
    None = Params.bt_slip_bps (the qualification gate and backtest.py use the same value)."""
    d = signals(add_indicators(df, p), p, scfg)
    acct = Account(equity0, p)
    eng = SymbolEngine(scfg, p, acct)
    eng.slip_bps = p.bt_slip_bps if slip_bps is None else slip_bps
    sw_lo = sw_hi = None
    if scfg.trail == "swing":
        st = structure.analyze(d, scfg.swing_k)
        sw_lo, sw_hi = st["swing_lo"].values, st["swing_hi"].values

    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    side, atr = d["side"].values, d["atr"].values
    idx = d.index
    hhmm = idx.strftime("%H:%M")
    max_gap = timedelta(minutes=scfg.interval_minutes * 1.5)
    itv = timedelta(minutes=scfg.interval_minutes)
    flat_min = 16 * 60 - (int(scfg.eod_time[:2]) * 60 + int(scfg.eod_time[3:]))
    flat_at = {}
    trades = []

    for i in range(len(d)):
        ts, day = idx[i], idx[i].date()
        # Enter only if the previous bar is really the immediately preceding one.
        if i > 0 and side[i - 1] != 0 and (ts - idx[i - 1]) <= max_gap:
            eng.try_enter(ts, o[i], int(side[i - 1]), float(atr[i - 1]), day, planned=float(c[i - 1]))
        last_of_day = False
        if scfg.flatten_eod and scfg.kind == "fx":
            last_of_day = (ts + itv).strftime("%H:%M") >= scfg.eod_time     # daily flat (live: 16:45 tick)
        elif scfg.flatten_eod:
            # same rule as the live bot: flat at (session close - flat_min); 12:45 on early-close days
            if day not in flat_at:
                sess = cal.session(day)
                flat_at[day] = (sess[1] if sess else datetime.combine(day, dtime(16, 0))) - timedelta(minutes=flat_min)
            last_of_day = (ts + itv).to_pydatetime() >= flat_at[day]
        tr = eng.on_bar_close(ts, o[i], h[i], l[i], c[i], day, last_of_day)
        if tr:
            trades.append(tr)
        elif sw_lo is not None and eng.pos is not None:
            eng.trail_update(sw_lo[i], sw_hi[i], float(atr[i]), float(c[i]))     # bar i closed -> next bar

    if eng.pos is not None:
        tr = eng.force_close(idx[-1], c[-1], "end_of_data")
        if tr:
            trades.append(tr)
    return trades, acct


def summarize(trades: list, equity0: float) -> dict:
    if not trades:
        return {"trades": 0}
    t = pd.DataFrame(trades).sort_values("exit_ts")
    wins, losses = t[t.pnl > 0], t[t.pnl <= 0]
    eq = pd.concat([pd.Series([equity0]), equity0 + t["pnl"].cumsum()], ignore_index=True)
    gross_loss = -losses.pnl.sum()
    return {
        "trades": len(t),
        "win_rate_%": round(100 * len(wins) / len(t), 1),
        "avg_R_per_trade": round(t.R.mean(), 3),
        "avg_win_R": round(wins.R.mean(), 2) if len(wins) else 0.0,
        "avg_loss_R": round(losses.R.mean(), 2) if len(losses) else 0.0,
        "profit_factor": round(wins.pnl.sum() / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "gross_pnl": round(t.pnl_gross.sum(), 2),
        "costs": round(t.cost.sum(), 2),
        "net_pnl": round(t.pnl.sum(), 2),
        "return_%": round(100 * t.pnl.sum() / equity0, 2),
        "max_drawdown_%": round(100 * (eq / eq.cummax() - 1).min(), 2),
    }


def show(title: str, stats: dict) -> None:
    print(f"\n== {title}")
    for k, v in stats.items():
        print(f"   {k:>16}: {v}")


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", help="e.g. QQQ or BTC/USD")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--slip", type=float, default=None, help="extra adverse slippage per market fill, bps")
    ap.add_argument("--slip", type=float, default=None, help="extra adverse slippage per market fill, bps")
    ap.add_argument("--mode", choices=("all", "fx", "qqq", "breakout", "scalp", "both"),
                    help="test the symbols of one trading mode")
    args = ap.parse_args()
    p = get_params()
    os.makedirs("out", exist_ok=True)

    if args.mode:
        targets = symbols_for_mode(args.mode)
    else:
        targets = SYMBOLS if args.all or not args.symbol else [s for s in SYMBOLS if s.symbol == args.symbol]
    if not targets:
        raise SystemExit(f"{args.symbol} is not configured in config.SYMBOLS")

    for scfg in targets:
        df = load_history(scfg)
        if df is None:
            path = csv_path(scfg.ticker, scfg.interval)
            print(f"[{scfg.symbol}] missing {path} -> run: python data.py download {scfg.ticker} {scfg.interval} 10000")
            continue
        trades, acct = run_backtest(df, scfg, p, p.initial_equity, slip_bps=args.slip)
        print(f"\n######## {scfg.symbol}  strategy={scfg.strategy}  bars={len(df)}  {df.index[0]} -> {df.index[-1]}")
        show("ALL DATA", summarize(trades, p.initial_equity))

        if trades:
            cut = df.index[int(len(df) * 0.6)]
            t = pd.DataFrame(trades)
            show("FIRST 60% (in-sample)", summarize([x for x in trades if x["exit_ts"] < cut], p.initial_equity))
            show("LAST 40% (out-of-sample)", summarize([x for x in trades if x["exit_ts"] >= cut], p.initial_equity))
            t.to_csv(os.path.join("out", f"backtest_trades_{scfg.file_stub}.csv"), index=False)
        print("\nIf out-of-sample is much worse than in-sample, the strategy is over-fitted. Do not trade it.")


if __name__ == "__main__":
    main()
