"""Qualification gate: a stream may trade automatically only if its strategy made money on THAT symbol's
own recent history (same engine, same costs as live), in both halves of the sample.

It is a filter against obvious mismatches (e.g. the scalping rules lost on SPY/IWM/DIA), NOT a proof of a
future edge - a stream that passes is still paper-traded and watched by the guardian.
"""
from __future__ import annotations

import time

import pandas as pd

import fx_hours
from config import Params, SymbolCfg

MIN_TRADES = 30
MIN_AVG_R = 0.03
MIN_PF = 1.10
HISTORY_BARS = 5000          # Twelve Data: 1 credit for up to 5000 bars; Binance: 5 free requests
MAX_AGE_H = 24.0             # re-qualify once a day


def prepare(df: pd.DataFrame, cfg: SymbolCfg) -> pd.DataFrame:
    if cfg.kind == "fx" and not df.empty:
        df = df[[fx_hours.bar_is_open(t.to_pydatetime(), cfg.metal) for t in df.index]]
    return df


def qualify(df: pd.DataFrame, cfg: SymbolCfg, p: Params) -> dict:
    from backtest import run_backtest, summarize               # local import: backtest imports config
    df = prepare(df, cfg)
    res = {"status": "fail", "at": time.time(), "bars": int(len(df)), "n": 0, "wr": None, "avgR": None,
           "pf": None, "R_tr": None, "R_te": None, "reason": ""}
    if len(df) < 400:
        res.update(status="error", reason=f"تاريخ غير كافٍ ({len(df)} شمعة)")
        return res
    q = Params(**{k: getattr(p, k) for k in p.__dataclass_fields__})
    q.initial_equity = 10000.0                                 # sizing granularity must not distort R
    trades, _ = run_backtest(df, cfg, q, q.initial_equity)
    s = summarize(trades, q.initial_equity)
    n = s.get("trades", 0)
    res.update(n=n, from_ts=str(df.index[0]), to_ts=str(df.index[-1]))
    if n:
        cut = df.index[int(len(df) * 0.6)]
        tr = summarize([t for t in trades if t["exit_ts"] < cut], q.initial_equity)
        te = summarize([t for t in trades if t["exit_ts"] >= cut], q.initial_equity)
        res.update(wr=s["win_rate_%"], avgR=s["avg_R_per_trade"], pf=s["profit_factor"],
                   R_tr=tr.get("avg_R_per_trade", 0.0) if tr.get("trades") else None,
                   R_te=te.get("avg_R_per_trade", 0.0) if te.get("trades") else None, dd=s["max_drawdown_%"])
    reasons = []
    if n < MIN_TRADES:
        reasons.append(f"صفقات قليلة ({n} < {MIN_TRADES})")
    else:
        if res["avgR"] is not None and res["avgR"] < MIN_AVG_R:
            reasons.append(f"متوسط الربح {res['avgR']:+.3f}R")
        if res["pf"] is not None and res["pf"] < MIN_PF:
            reasons.append(f"معامل الربح {res['pf']}")
        if (res["R_tr"] or 0) <= 0 or (res["R_te"] or 0) <= 0:
            reasons.append("خاسر في أحد نصفي الاختبار")
    res["status"] = "ok" if not reasons else "fail"
    res["reason"] = "، ".join(reasons) if reasons else "ربح في كامل الفترة وفي نصفيها"
    return res
