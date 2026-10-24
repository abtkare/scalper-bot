"""Per-stream chart data + plain-Arabic analysis for the dashboard (v11).

INFORMATIONAL ONLY: nothing here opens, closes or changes a trade. It reads the same closed bars the strategy
used and explains them: what the strategy waits for, the market structure, the key levels, the last candle,
and how far price is from the next planned entry.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

import structure

CHART_BARS = 80

STRAT_DESC = {
    "scalp": "سكالبينج ارتداد إلى VWAP: يشتري عندما يبتعد السعر كثيراً تحت متوسط اليوم في اتجاه صاعد، "
             "ويبيع عكس ذلك. هدف صغير ووقف أوسع، نسبة ربح عالية وربح صغير لكل صفقة.",
    "rsi2": "ارتداد RSI2: يشتري هبوطاً حاداً قصيراً (RSI2 تحت الحد) والسعر فوق EMA200، ويبيع ارتفاعاً حاداً "
            "والسعر تحته. صفقة واحدة في الساعة كحد أقصى صباح نيويورك.",
    "breakout": "اختراق: يشتري عندما تغلق الشمعة فوق أعلى قمة لآخر فترة، بشرط اتجاه صاعد على الإطارين "
                "(الحالي والأكبر). هدف أكبر من الوقف، نسبة ربح أقل وربح أكبر لكل صفقة.",
    "pullback": "سحب إلى EMA20 في اتجاه قائم.",
}


def _r(x, nd: int = 6) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return float(f"{x:.{nd}g}")


def epochs(idx, tz: str) -> list:
    """Bar start times -> real UNIX seconds (bars are naive exchange-local times)."""
    ix = pd.DatetimeIndex(idx)
    if ix.tz is None:
        ix = ix.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    sec = (ix.tz_convert("UTC").tz_localize(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    out, prev = [], 0
    for x in sec:                                   # NaT (DST fall-back hour) -> previous bar + 1 s
        prev = int(x) if not pd.isna(x) else prev + 1
        out.append(prev)
    return out


def chart(d: pd.DataFrame, scfg, k: Optional[int] = None, n: int = CHART_BARS) -> dict:
    """Last `n` CLOSED bars as compact arrays + overlays + confirmed swing pivots (for the candle chart)."""
    k = k or getattr(scfg, "swing_k", 3)
    tail = d.iloc[-(n + 2 * k + 5):]
    sh, sl = structure.swings(tail["high"].values, tail["low"].values, k)
    off = len(tail) - min(n, len(tail))
    v = tail.iloc[off:]
    piv = []
    for j in range(len(tail)):                    # sh[j] is set at the bar where pivot (j-k) became visible
        for arr, kind in ((sh, "H"), (sl, "L")):
            if not np.isnan(arr[j]) and j - k >= off:
                piv.append([j - k - off, kind, _r(arr[j])])
    out = {"iv": scfg.interval, "t": epochs(v.index, scfg.tz),
           "tz": scfg.tz, "o": [_r(x) for x in v["open"]], "h": [_r(x) for x in v["high"]],
           "l": [_r(x) for x in v["low"]], "c": [_r(x) for x in v["close"]], "piv": piv, "lines": {}}
    names = {"scalp": (("vwap", "VWAP"), ("ema200", "EMA200")), "rsi2": (("ema200", "EMA200"),),
             "breakout": (("ema_f", "EMA20"), ("ema_s", "EMA50"))}.get(scfg.strategy, ())
    for col, label in names:
        if col in v.columns:
            out["lines"][label] = [_r(x) for x in v[col]]
    return out


def _candle(row, prev, atr: float) -> str:
    o, h, l, c = (float(row[x]) for x in ("open", "high", "low", "close"))
    rng = h - l
    if not (rng > 0 and atr > 0):
        return ""
    body = abs(c - o)
    up_w, lo_w = h - max(o, c), min(o, c) - l
    po, pc = float(prev["open"]), float(prev["close"])
    if c > o and pc < po and c >= po and o <= pc and body > abs(pc - po):
        return "ابتلاع شرائي (قوة مشترين)"
    if c < o and pc > po and c <= po and o >= pc and body > abs(pc - po):
        return "ابتلاع بيعي (قوة بائعين)"
    if lo_w >= 2 * body and lo_w >= 0.6 * rng:
        return "مطرقة / ذيل سفلي طويل (رفض للأسعار المنخفضة)"
    if up_w >= 2 * body and up_w >= 0.6 * rng:
        return "شهاب / ذيل علوي طويل (رفض للأسعار المرتفعة)"
    if body <= 0.1 * rng:
        return "دوجي (تردد)"
    if body >= 0.7 * rng and rng >= 1.2 * atr:
        return "شمعة قوية " + ("صاعدة" if c > o else "هابطة")
    return "صاعدة" if c > o else "هابطة"


def analyze(d: pd.DataFrame, plan: dict, scfg, position: Optional[dict] = None) -> dict:
    """Plain-Arabic reading of one stream. `d` = output of strategy.signals() on CLOSED bars."""
    last = d.iloc[-1]
    c, atr = float(last["close"]), float(last["atr"])
    lines = []
    st = (plan or {}).get("structure") or {}
    H = {1: "صاعد", -1: "هابط", 0: "متذبذب"}
    if st:
        t = st.get("text", "")
        if st.get("htf") is not None:
            t += f" · الإطار الأكبر {H.get(st['htf'], '')}"
        lines.append(("🧭 الهيكل", t))
    if scfg.strategy == "scalp" and atr > 0:
        z = (c - float(last["vwap"])) / atr
        side = "فوق" if z >= 0 else "تحت"
        lines.append(("📏 VWAP", f"السعر {side} متوسط اليوم بـ {abs(z):.2f} ATR (الإشارة عند {scfg.vwap_k:g} ATR)"))
        lines.append(("📈 EMA200", "السعر فوقه: شراء فقط" if c > float(last["ema200"]) else "السعر تحته: بيع فقط"))
    elif scfg.strategy == "rsi2":
        r2 = float(last["rsi2"])
        lines.append(("⚡ RSI2", f"{r2:.0f} (شراء تحت {scfg.rsi2_th:g} · بيع فوق {100 - scfg.rsi2_th:g})"))
        lines.append(("📈 EMA200", "السعر فوقه: شراء فقط" if c > float(last["ema200"]) else "السعر تحته: بيع فقط"))
    elif scfg.strategy == "breakout" and atr > 0:
        hh = float(d["high"].iloc[-scfg.donchian_n:].max())
        lines.append(("🚪 الاختراق", f"أعلى قمة لآخر {scfg.donchian_n} شمعة {_r(hh)} · تبعد {(hh - c) / atr:.2f} ATR"))
        lines.append(("📈 EMA20/50", "صاعد" if float(last["ema_f"]) > float(last["ema_s"]) else "غير صاعد (لا شراء)"))
    if st and atr > 0:
        lv = []
        if st.get("swing_hi") is not None:
            lv.append(f"مقاومة (آخر قمة) {st['swing_hi']} على بعد {(st['swing_hi'] - c) / atr:+.1f} ATR")
        if st.get("swing_lo") is not None:
            lv.append(f"دعم (آخر قاع) {st['swing_lo']} على بعد {(st['swing_lo'] - c) / atr:+.1f} ATR")
        if lv:
            lines.append(("🧱 المستويات", " · ".join(lv)))
    if len(d) >= 2:
        cd = _candle(last, d.iloc[-2], atr)
        if cd:
            lines.append(("🕯 آخر شمعة", cd))
    if getattr(scfg, "trail", ""):
        lines.append(("🪜 الخروج", "وقف متحرك خلف آخر قاع/قمة مؤكدة بعد +%gR" % scfg.trail_after_r
                      + (" بلا هدف ثابت" if scfg.trail_no_target else " مع هدف ثابت")))

    # verdict
    if position:
        side = "شراء" if position.get("side") == "long" else "بيع"
        verdict = f"صفقة {side} مفتوحة — الوقف {_r(position.get('stop'))}"
    else:
        best = None
        for key, w in (("long", "شراء"), ("short", "بيع")):
            pl = (plan or {}).get(key)
            if pl and (best is None or abs(pl["dist_pct"]) < abs(best[1]["dist_pct"])):
                best = (w, pl)
        if best is None:
            verdict = "لا فرصة حسب الشروط الآن" + (f" ({plan['note']})" if (plan or {}).get("note") else "")
        else:
            w, pl = best
            dist_atr = abs(pl["trigger"] - c) / atr if atr > 0 else float("nan")
            near = "قريب جداً" if dist_atr <= 0.5 else "قريب" if dist_atr <= 1.5 else "بعيد"
            verdict = f"{near} من إشارة {w}: إغلاق {'تحت' if pl['op'] == '<' else 'فوق'} {_r(pl['trigger'])} ({dist_atr:.1f} ATR)"
    return {"desc": STRAT_DESC.get(scfg.strategy, ""), "lines": [[a, b] for a, b in lines], "verdict": verdict}
