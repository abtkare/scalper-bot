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

from config import SYMBOLS, SymbolCfg, Params, get_params, load_env
from data import csv_path, load_csv
import market_calendar as cal
from engine import Account, SymbolEngine
from strategy import add_indicators, signals


def resample_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    o = df.resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return o.dropna()


def run_backtest(df: pd.DataFrame, scfg: SymbolCfg, p: Params, equity0: float):
    d = signals(add_indicators(df, p), p, scfg)
    acct = Account(equity0, p)
    eng = SymbolEngine(scfg, p, acct)

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
            eng.try_enter(ts, o[i], int(side[i - 1]), float(atr[i - 1]), day)
        last_of_day = False
        if scfg.flatten_eod:
            # same rule as the live bot: flat at (session close - flat_min); 12:45 on early-close days
            if day not in flat_at:
                sess = cal.session(day)
                flat_at[day] = (sess[1] if sess else datetime.combine(day, dtime(16, 0))) - timedelta(minutes=flat_min)
            last_of_day = (ts + itv).to_pydatetime() >= flat_at[day]
        tr = eng.on_bar_close(ts, o[i], h[i], l[i], c[i], day, last_of_day)
        if tr:
            trades.append(tr)

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
    args = ap.parse_args()
    p = get_params()
    os.makedirs("out", exist_ok=True)

    targets = SYMBOLS if args.all or not args.symbol else [s for s in SYMBOLS if s.symbol == args.symbol]
    if not targets:
        raise SystemExit(f"{args.symbol} is not configured in config.SYMBOLS")

    for scfg in targets:
        path = csv_path(scfg.symbol, scfg.interval)
        five = csv_path(scfg.symbol, "5min")
        if os.path.exists(path):
            df = load_csv(path)
        elif os.path.exists(five) and scfg.interval_minutes % 5 == 0 and scfg.interval_minutes > 5:
            df = resample_bars(load_csv(five), scfg.interval_minutes)     # build 15-min bars from 5-min history
        else:
            print(f"[{scfg.symbol}] missing {path} -> run: python data.py download {scfg.symbol} {scfg.interval} 10000")
            continue
        trades, acct = run_backtest(df, scfg, p, p.initial_equity)
        print(f"\n######## {scfg.symbol}  bars={len(df)}  {df.index[0]} -> {df.index[-1]}")
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
