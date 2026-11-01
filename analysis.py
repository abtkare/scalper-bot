"""Per-stream chart data + plain-Arabic analysis for the dashboard (v14; money flow + timeframes v16).

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
import zones as zmod
from dataclasses import replace as _replace

CHART_BARS = 80

STRAT_DESC = {
    "scalp": "سكالبينج ارتداد إلى VWAP: يشتري عندما يبتعد السعر كثيراً تحت متوسط اليوم في اتجاه صاعد، "
             "ويبيع عكس ذلك. هدف صغير ووقف أوسع، نسبة ربح عالية وربح صغير لكل صفقة.",
    "rsi2": "ارتداد RSI2: يشتري هبوطاً حاداً قصيراً (RSI2 تحت الحد) والسعر فوق EMA200، ويبيع ارتفاعاً حاداً "
            "والسعر تحته. صفقة واحدة في الساعة كحد أقصى صباح نيويورك.",
    "breakout": "اختراق: يشتري عندما تغلق الشمعة فوق أعلى قمة لآخر فترة، بشرط اتجاه صاعد على الإطارين "
                "(الحالي والأكبر). هدف أكبر من الوقف، نسبة ربح أقل وربح أكبر لكل صفقة.",
    "pullback": "سحب إلى EMA20 في اتجاه قائم.",
    "zone": "مناطق المؤسسات (تجريبي): بعد شمعة اندفاع قوية تكسر آخر قمة/قاع، تُحدَّد آخر شمعة معاكسة قبلها "
            "كمنطقة طلب/عرض، ويُدخل مع الاتجاه عند أول عودة للسعر إليها ورفضها. الوقف خلف المنطقة. "
            "في الاختبار التاريخي لم تحقق ربحاً بعد التكاليف، لذلك لا تتداول إلا إذا اجتازت بوابة الاختبار.",
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
    # money flowing in each candle (v16): dollar volume split into buying / selling pressure
    mf = money_flow(v)
    out["hasv"] = mf["hasv"]
    if mf["hasv"]:
        out["vb"] = [_r(x, 4) for x in mf["buy"]]
        out["vs"] = [_r(x, 4) for x in mf["sell"]]
    # institutional demand / supply zones (v15) - drawn on every chart, informational only
    zs = inst_zones(d, scfg)
    out["zones"] = [[z["side"], _r(z["lo"]), _r(z["hi"]), max(0, len(v) - 1 - z["age"])] for z in zs]
    return out


def money_flow(d: pd.DataFrame) -> dict:
    """Money that traded in each candle = volume x typical price (USD), split into buying / selling pressure by
    where the candle CLOSED inside its range (close at the high = all buying, at the low = all selling; the
    Chaikin money-flow split). Exchanges only publish TOTAL volume, so the split is an estimate.
    Spot gold / FX have no central exchange -> no real volume (hasv False)."""
    vol = d["volume"].astype(float).fillna(0.0).values if "volume" in d.columns else np.zeros(len(d))
    hasv = bool(len(vol) and (vol[-min(len(vol), 50):] > 0).mean() > 0.5)
    if not hasv:
        return {"hasv": False}
    h, l, c = (d[k].astype(float).values for k in ("high", "low", "close"))
    rng = np.where(h - l > 0, h - l, np.nan)
    clv = np.nan_to_num((c - l) / rng, nan=0.5)            # 0 = closed at the low, 1 = at the high
    usd = vol * (h + l + c) / 3.0
    buy, sell = usd * clv, usd * (1.0 - clv)
    n = min(20, len(usd))
    cmf = float((buy[-n:] - sell[-n:]).sum() / usd[-n:].sum()) if usd[-n:].sum() > 0 else 0.0
    avg = float(np.median(usd[-51:-1])) if len(usd) > 5 else float(np.median(usd))   # median: feeds have spikes
    return {"hasv": True, "buy": buy, "sell": sell, "cmf": cmf, "usd_last": float(usd[-1]),
            "rel_last": float(usd[-1] / avg) if avg > 0 else 0.0, "buy_pct_last": float(clv[-1] * 100)}


TF_MULTS = {15: (1, 2, 4, 16), 5: (1, 3, 12, 48), 60: (1, 4, 24), 30: (1, 2, 8), 1: (1, 5, 15, 60)}


def tf_label(minutes: int) -> str:
    return f"{minutes // 1440}d" if minutes % 1440 == 0 else f"{minutes // 60}h" if minutes % 60 == 0 else f"{minutes}min"


def chart_tf(bars: pd.DataFrame, scfg, p, mult: int) -> dict:
    """The same stream's closed bars re-built on a HIGHER timeframe (mult x its interval) for the chart switch."""
    from strategy import add_indicators
    mins = scfg.interval_minutes * mult
    if mult <= 1:
        df = bars
    else:
        rule = "1D" if mins == 1440 else f"{mins}min"
        df = bars.resample(rule, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    if len(df) < 10:
        return {"error": "شموع غير كافية لهذا الإطار"}
    cfg = _replace(scfg, interval=tf_label(mins), interval_minutes=mins, strategy="pullback" if mult > 1 else scfg.strategy)
    d = add_indicators(df, p)
    out = chart(d, cfg)
    out["lines"] = {"EMA20": [_r(x) for x in d["ema_f"].iloc[-len(out["t"]):]],
                    "EMA50": [_r(x) for x in d["ema_s"].iloc[-len(out["t"]):]]} if mult > 1 else out["lines"]
    out["mult"] = mult
    return out


def inst_zones(d: pd.DataFrame, scfg) -> list:
    """Live institutional zones for display. A 'zone' stream uses its own settings; any other stream the
    research defaults (order blocks after a 1-ATR displacement that broke structure, no HTF filter)."""
    if "zones" in d.attrs:
        return list(d.attrs["zones"])
    try:
        cfg = scfg if scfg.strategy == "zone" else _replace(scfg, zone_mode="ob", htf_mult=0, zone_one_side=False)
        return zmod.find(d.iloc[-400:], cfg)[4]
    except Exception:
        return []


def _usd(x: float) -> str:
    return f"{x / 1e9:.2f} مليار $" if x >= 1e9 else f"{x / 1e6:.1f} مليون $" if x >= 1e6 else f"{x / 1e3:.0f} ألف $"


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
    mf = money_flow(d.iloc[-60:])
    if mf["hasv"]:
        flow = "دخول أموال (ضغط شراء)" if mf["cmf"] > 0.05 else "خروج أموال (ضغط بيع)" if mf["cmf"] < -0.05 else "متوازن"
        lines.append(("💰 تدفق الأموال", f"آخر 20 شمعة: {flow} ({mf['cmf']:+.2f}) · آخر شمعة {_usd(mf['usd_last'])} "
                      f"({mf['rel_last']:.1f}× المعتاد)، شراء ~{mf['buy_pct_last']:.0f}%"))
    else:
        lines.append(("💰 تدفق الأموال", "لا يوجد حجم تداول حقيقي لهذا السوق (الذهب والعملات الفورية بلا بورصة مركزية)"))
    zs = inst_zones(d, scfg)
    if zs and atr > 0:
        dem = sorted([z for z in zs if z["side"] == 1 and z["hi"] <= c * 1.0 + atr * 3], key=lambda z: c - z["hi"])
        sup = sorted([z for z in zs if z["side"] == -1 and z["lo"] >= c - atr * 3], key=lambda z: z["lo"] - c)
        t = []
        if dem:
            z = dem[0]
            t.append(f"طلب (دعم) {_r(z['lo'])}–{_r(z['hi'])} على بعد {(c - z['hi']) / atr:.1f} ATR")
        if sup:
            z = sup[0]
            t.append(f"عرض (مقاومة) {_r(z['lo'])}–{_r(z['hi'])} على بعد {(z['lo'] - c) / atr:.1f} ATR")
        if t:
            lines.append(("🏦 مناطق المؤسسات", " · ".join(t) + " (للمعلومة)"))
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
