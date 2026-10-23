"""Watchlist: any index ETF, currency pair, metal or crypto, each with one or more strategies.

Every (symbol, strategy) pair becomes one "stream" = one SymbolCfg the bot watches. A stream is only
traded automatically after it passed a backtest on that symbol's own recent history (see qualify.py);
otherwise it is "watch only": the bot still shows its pre-computed entry / stop / target plan.

The list lives in watchlist.json (edited from the dashboard). If the file is missing, DEFAULT is used.

Asset classes and their templates (all numbers are research defaults - RESEARCH.md):
  etf    : US ETFs/stocks (QQQ, SPY, DIA = Dow Jones, IWM ...). NYSE session, flat 15:45 NY.
  fx     : currency pairs (EUR/USD ...). Sun 17:00 -> Fri 17:00 NY, flat 16:45 NY.
  metal  : spot metals (XAU/USD gold, XAG/USD silver). Like fx + daily break 17-18 NY.
  crypto : BTC/USD, ETH/USD ... via Binance (free data, no credits), 24/7, no daily flat.
Strategies:
  scalp    : 15-min VWAP mean reversion, small target 0.75 ATR / stop 2 ATR, long + short
  rsi2     : 1-hour RSI(2) mean reversion with EMA200 trend filter, target 1.5 / stop 1 ATR, long + short
  breakout : Donchian breakout in the EMA trend direction, long only
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from typing import Dict, List, Optional

from config import ConfigError, SymbolCfg

WATCHLIST_FILE = "watchlist.json"
CLASSES = ("etf", "fx", "metal", "crypto")
STRATEGIES = ("scalp", "rsi2", "breakout")
STRAT_AR = {"scalp": "سكالبينج", "rsi2": "ارتداد RSI2", "breakout": "اختراق"}
CLASS_AR = {"etf": "مؤشر / صندوق", "fx": "عملات", "metal": "معادن", "crypto": "عملة رقمية"}

DEFAULT: List[dict] = [
    {"symbol": "QQQ", "class": "etf", "strategies": ["breakout", "scalp"]},
    {"symbol": "DIA", "class": "etf", "strategies": ["scalp", "breakout"]},       # Dow Jones ETF
    {"symbol": "XAU/USD", "class": "metal", "strategies": ["rsi2", "scalp"]},
    {"symbol": "EUR/USD", "class": "fx", "strategies": ["rsi2", "scalp"]},
    {"symbol": "GBP/USD", "class": "fx", "strategies": ["rsi2", "scalp"]},
    {"symbol": "USD/JPY", "class": "fx", "strategies": ["rsi2", "scalp"]},
    {"symbol": "BTC/USD", "class": "crypto", "strategies": ["scalp", "rsi2"]},
    {"symbol": "ETH/USD", "class": "crypto", "strategies": ["scalp", "rsi2"]},
]

# Per-side cost assumptions (bps of price: spread + commission). Replace with your broker's numbers.
COST_BPS = {"XAU/USD": 0.5, "XAG/USD": 2.0, "WTI/USD": 3.0, "BRENT/USD": 3.0,
            "EUR/USD": 0.5, "GBP/USD": 0.6, "USD/JPY": 0.5}
CLASS_COST = {"etf": 1.0, "fx": 0.8, "metal": 1.5, "crypto": 5.0}

_ISO = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK", "TRY", "ZAR", "MXN", "SGD",
        "HKD", "CNH", "PLN", "DKK", "AED", "SAR"}
_METALS = {"XAU", "XAG", "XPT", "XPD"}
_SYM_RE = re.compile(r"^[A-Z0-9.\-]{1,10}(/[A-Z0-9]{2,6})?$")


def guess_class(symbol: str) -> str:
    s = symbol.upper().strip()
    if "/" in s:
        a, b = s.split("/", 1)
        if a in _METALS:
            return "metal"
        if a in _ISO and b in _ISO:
            return "fx"
        if a in ("WTI", "BRENT", "XBR", "XTI"):
            return "metal"            # energy CFDs: same 24/5 hours + daily break as metals
        return "crypto"
    return "etf"


def stream_id(symbol: str, strategy: str, cls: str) -> str:
    """Unique bot id. Keeps the ids used by earlier versions so saved positions/state still match."""
    if symbol == "QQQ" and strategy == "breakout":
        return "QQQ"
    if symbol == "QQQ" and strategy == "scalp":
        return "QQQ-SCALP"
    if cls in ("fx", "metal") and strategy == "rsi2":
        return symbol
    return f"{symbol}-{ {'scalp': 'SCALP', 'rsi2': 'RSI2', 'breakout': 'BRK'}[strategy] }".replace(" ", "")


def make_cfg(symbol: str, cls: str, strategy: str) -> SymbolCfg:
    symbol = symbol.upper().strip()
    if cls not in CLASSES:
        raise ConfigError(f"{symbol}: unknown class {cls!r}")
    if strategy not in STRATEGIES:
        raise ConfigError(f"{symbol}: unknown strategy {strategy!r}")
    sid = stream_id(symbol, strategy, cls)
    cost = COST_BPS.get(symbol, CLASS_COST[cls])
    common = dict(symbol=sid, provider_symbol=symbol if sid != symbol else "", cost_bps=cost, history_bars=600)

    if cls == "etf":
        base = SymbolCfg(kind="stock", tz="America/New_York", flatten_eod=True, eod_time="15:45",
                         windows=[("09:45", "15:30")], **common)
        if strategy == "breakout":
            return replace(base, interval="15min", interval_minutes=15, strategy="breakout", donchian_n=24,
                           stop_atr=0.75, target_atr=2.0, history_bars=300)
        if strategy == "scalp":
            return replace(base, interval="15min", interval_minutes=15, strategy="scalp", vwap_k=1.0,
                           stop_atr=2.0, target_atr=0.75, allow_short=True)
        return replace(base, interval="1h", interval_minutes=60, strategy="rsi2", rsi2_th=10.0,
                       stop_atr=1.0, target_atr=1.5, allow_short=True, windows=[("09:30", "14:30")])

    if cls in ("fx", "metal"):
        base = SymbolCfg(kind="fx", tz="America/New_York", flatten_eod=True, eod_time="16:45", poll_sec=30.0,
                         **common)
        if strategy == "rsi2":
            return replace(base, interval="1h", interval_minutes=60, strategy="rsi2", rsi2_th=10.0,
                           stop_atr=1.0, target_atr=1.5, allow_short=True, windows=[("08:00", "12:00")])
        if strategy == "scalp":
            return replace(base, interval="15min", interval_minutes=15, strategy="scalp", vwap_k=1.0,
                           stop_atr=2.0, target_atr=0.75, allow_short=True, windows=[("03:00", "12:00")],
                           poll_sec=20.0)
        return replace(base, interval="1h", interval_minutes=60, strategy="breakout", donchian_n=12,
                       stop_atr=2.0, target_atr=2.5, windows=[("03:00", "12:00")])

    # crypto (Binance public data: free, no credits)
    base = SymbolCfg(kind="crypto", tz="UTC", provider="binance", poll_sec=5.0, **common)
    if strategy == "scalp":
        return replace(base, interval="15min", interval_minutes=15, strategy="scalp", vwap_k=1.0,
                       stop_atr=2.0, target_atr=0.75, allow_short=True)
    if strategy == "rsi2":
        return replace(base, interval="1h", interval_minutes=60, strategy="rsi2", rsi2_th=10.0,
                       stop_atr=1.0, target_atr=1.5, allow_short=True)
    return replace(base, interval="1h", interval_minutes=60, strategy="breakout", donchian_n=24,
                   stop_atr=1.0, target_atr=2.0)


def validate_entry(e: dict) -> dict:
    sym = str(e.get("symbol", "")).upper().strip()
    if not _SYM_RE.match(sym):
        raise ConfigError(f"invalid symbol {sym!r}")
    cls = e.get("class") or guess_class(sym)
    if cls not in CLASSES:
        raise ConfigError(f"{sym}: class must be one of {CLASSES}")
    strats = [s for s in (e.get("strategies") or ["scalp"]) if s in STRATEGIES]
    if not strats:
        raise ConfigError(f"{sym}: choose at least one strategy {STRATEGIES}")
    off = [x for x in (e.get("off") or []) if isinstance(x, str)]
    return {"symbol": sym, "class": cls, "strategies": list(dict.fromkeys(strats)), "off": off}


def load(path: str = WATCHLIST_FILE) -> List[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        items = raw.get("symbols", raw) if isinstance(raw, dict) else raw
        out, seen = [], set()
        for e in items:
            v = validate_entry(e)
            if v["symbol"] not in seen:
                seen.add(v["symbol"])
                out.append(v)
        if out:
            return out
    except FileNotFoundError:
        pass
    except Exception:
        pass                          # corrupt file -> defaults (the bot must still start)
    return [validate_entry(e) for e in DEFAULT]


def save(items: List[dict], path: str = WATCHLIST_FILE) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"symbols": [validate_entry(e) for e in items]}, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def streams(items: List[dict]) -> List[SymbolCfg]:
    out: List[SymbolCfg] = []
    for e in items:
        for st in e["strategies"]:
            out.append(make_cfg(e["symbol"], e["class"], st))
    return out


def disabled_ids(items: List[dict]) -> set:
    return {x for e in items for x in e.get("off", [])}


def add_symbol(items: List[dict], symbol: str, cls: Optional[str], strategies: List[str]) -> List[dict]:
    v = validate_entry({"symbol": symbol, "class": cls or None, "strategies": strategies})
    rest = [e for e in items if e["symbol"] != v["symbol"]]
    if len(rest) >= 25:
        raise ConfigError("قائمة المراقبة ممتلئة (25 رمزاً كحد أقصى)")
    return rest + [v]


def remove_symbol(items: List[dict], symbol: str) -> List[dict]:
    symbol = symbol.upper().strip()
    out = [e for e in items if e["symbol"] != symbol]
    if len(out) == len(items):
        raise ConfigError(f"{symbol} غير موجود في القائمة")
    if not out:
        raise ConfigError("لا يمكن حذف آخر رمز في القائمة")
    return out


def set_stream_enabled(items: List[dict], sid: str, enabled: bool) -> List[dict]:
    out = []
    found = False
    for e in items:
        e = dict(e)
        ids = {stream_id(e["symbol"], s, e["class"]) for s in e["strategies"]}
        if sid in ids:
            found = True
            off = set(e.get("off", []))
            off.discard(sid) if enabled else off.add(sid)
            e["off"] = sorted(off)
        out.append(e)
    if not found:
        raise ConfigError(f"{sid} غير موجود")
    return out


def class_of(items: List[dict]) -> Dict[str, str]:
    return {stream_id(e["symbol"], s, e["class"]): e["class"] for e in items for s in e["strategies"]}
