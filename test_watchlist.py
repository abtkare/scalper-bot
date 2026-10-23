"""Watchlist tests (v6): templates for every asset class, legacy ids, add/remove/on/off, the qualification gate
blocks auto-trading until a stream passed its own backtest, pre-computed trade plans agree with the real
signal, daily credit tiers, Binance pagination, dashboard watchlist API.
Run: python test_watchlist.py      (no network, no API key)
"""
import base64
import json
import logging
import os
import tempfile
import time
import urllib.request

import numpy as np
import pandas as pd

import bot
import test_robust as tr
import watchlist as wl
from backtest import load_history
from config import ConfigError, Params, SymbolCfg, validate_config
from providers import Binance, RateBudget
from qualify import qualify
from strategy import add_indicators, plan_levels, signals

logging.getLogger().setLevel(logging.CRITICAL)
PASSED = []


def ok(name):
    PASSED.append(name)
    print(f"  ok  {name}")


# ------------------------------------------------------------------ 1) templates / ids
def test_templates():
    os.environ.setdefault("TWELVE_DATA_API_KEY", "x")
    cases = {"QQQ": "etf", "DIA": "etf", "SPY": "etf", "XAU/USD": "metal", "XAG/USD": "metal", "EUR/USD": "fx",
             "USD/JPY": "fx", "BTC/USD": "crypto", "ETH/USD": "crypto", "SOL/USD": "crypto", "WTI/USD": "metal"}
    for sym, cls in cases.items():
        assert wl.guess_class(sym) == cls, (sym, wl.guess_class(sym))
        cfgs = [wl.make_cfg(sym, cls, st) for st in wl.STRATEGIES]
        validate_config(Params(), cfgs)
        assert all(c.ticker == sym for c in cfgs)
    assert wl.stream_id("QQQ", "breakout", "etf") == "QQQ" and wl.stream_id("QQQ", "scalp", "etf") == "QQQ-SCALP"
    assert wl.stream_id("XAU/USD", "rsi2", "metal") == "XAU/USD", "v5 ids kept (saved positions still match)"
    assert wl.stream_id("BTC/USD", "scalp", "crypto") == "BTC/USD-SCALP"
    assert wl.make_cfg("BTC/USD", "crypto", "scalp").provider == "binance"
    ids = [c.symbol for c in wl.streams([wl.validate_entry(e) for e in wl.DEFAULT])]
    assert len(ids) == len(set(ids)), "stream ids must be unique"
    ok(f"templates valid for every class/strategy; {len(ids)} default streams with unique, backward-compatible ids")


def test_file_ops():
    d = tempfile.mkdtemp()
    f = os.path.join(d, "wl.json")
    items = wl.load(f)
    assert [e["symbol"] for e in items][:3] == ["QQQ", "XAU/USD", "EUR/USD"]   # defaults when no file
    assert "DIA" not in [e["symbol"] for e in items]                          # removed 2026-10-23
    ids = [c.symbol for c in wl.streams(items)]
    assert len(ids) == 14 and ids[2] == "XAU/USD", ids                        # gold served right after QQQ
    items = wl.add_symbol(items, "spy", None, ["scalp", "breakout"])
    wl.save(items, f)
    items = wl.load(f)
    assert items[-1] == {"symbol": "SPY", "class": "etf", "strategies": ["scalp", "breakout"], "off": []}
    items = wl.set_stream_enabled(items, "SPY-SCALP", False)
    assert "SPY-SCALP" in wl.disabled_ids(items)
    items = wl.remove_symbol(items, "SPY")
    assert "SPY" not in [e["symbol"] for e in items]
    for bad in ("", "DROP TABLE", "A" * 30):
        try:
            wl.add_symbol(items, bad, None, ["scalp"])
            raise AssertionError(f"accepted {bad!r}")
        except ConfigError:
            pass
    open(f, "w").write("{corrupt")
    assert wl.load(f), "corrupt file must fall back to defaults"
    ok("watchlist file: defaults, add (class auto-detected), on/off, remove, bad symbols refused, corrupt file safe")


# ------------------------------------------------------------------ 2) qualification gate blocks entries
def _gate_env(qual):
    cfg = SymbolCfg("TEST/USD", "crypto", "UTC", cost_bps=0.0, provider="fake", strategy="scalp",
                    allow_short=True, stop_atr=2.0, target_atr=0.75)
    e = tr.Env(tr.ramp(tr.T0 + 450, 0.05), cfg=cfg, sig_side=1)
    e.rt.gate = True
    e.ss.qual = qual
    e.ss.next_qual_try = 1e18                      # no network qualification in the test
    return e


def test_gate():
    e = _gate_env({"status": "fail", "at": time.time(), "reason": "test"})
    e.run(900)
    assert e.ss.eng.pos is None and not e.trades(), "unqualified stream traded"
    assert e.ss.plan is None or isinstance(e.ss.plan, dict)
    e2 = _gate_env(None)
    e2.run(900)
    assert not e2.trades(), "stream traded before its qualification finished"
    e3 = _gate_env({"status": "ok", "at": time.time()})
    assert e3.run(900, until=lambda: bool(e3.trades())), "qualified stream did not trade"
    e4 = _gate_env({"status": "ok", "at": time.time()})
    e4.ss.disabled = True
    e4.run(900)
    assert not e4.trades(), "a stream switched off by the user traded"
    ok("gate: fail / pending / switched-off streams never trade; a qualified stream trades")


def test_qualify_function():
    p = Params()
    good = wl.make_cfg("QQQ", "etf", "breakout")
    r = qualify(load_history(good).tail(5000), good, p)
    assert r["status"] == "ok" and r["n"] >= 30, r
    bad = wl.make_cfg("DIA", "etf", "scalp")
    r2 = qualify(load_history(bad).tail(5000), bad, p)
    assert r2["status"] == "fail" and r2["reason"], r2
    r3 = qualify(load_history(good).tail(100), good, p)
    assert r3["status"] == "error"
    ok(f"qualify(): QQQ breakout OK ({r['n']} trades, {r['avgR']:+.3f}R); DIA scalping FAIL ({r2['reason']})")


def test_qual_persisted():
    e = _gate_env({"status": "ok", "at": 123.0, "n": 50})
    st = e.rt.snapshot()
    rt2 = e.new_runtime()
    rt2.restore(st)
    assert rt2.syms[0].qual["n"] == 50
    ok("qualification result survives a restart (no re-download within 24 h)")


# ------------------------------------------------------------------ 3) plans agree with the real signal
def _agreement(cfg, step=7):
    p = Params()
    df = load_history(cfg)
    full = signals(add_indicators(df, p), p, cfg)
    agree = tot = 0
    for i in range(700, len(df) - 1, step):
        d = signals(add_indicators(df.iloc[:i + 1], p), p, cfg)
        pl = plan_levels(d, p, cfg)
        nxt = float(df["close"].iloc[i + 1])
        L, S = pl["long"], pl["short"]
        want_l = L is not None and (nxt < L["trigger"] if L["op"] == "<" else nxt > L["trigger"]) \
            and ("bound" not in L or nxt > L["bound"])
        want_s = S is not None and (nxt < S["trigger"] if S["op"] == "<" else nxt > S["trigger"]) \
            and ("bound" not in S or nxt < S["bound"])
        got = int(full["side"].iloc[i + 1])
        if got == -1 and not cfg.allow_short:
            got = 0                                   # the engine ignores shorts for long-only streams
        if want_l or want_s or got:
            tot += 1
            agree += int((got == 1) == want_l and (got == -1) == want_s)
    return agree, tot


def test_plans_match_signals():
    res = {}
    for sym, cls, st in (("XAU/USD", "metal", "rsi2"), ("QQQ", "etf", "breakout"), ("QQQ", "etf", "scalp")):
        a, t = _agreement(wl.make_cfg(sym, cls, st))
        res[st] = (a, t)
        need = 0.95 if st != "scalp" else 0.80       # VWAP/ATR move with the new bar -> approximate
        assert t > 20 and a / t >= need, (st, a, t)
    ok("pre-computed plans predict the real next-bar signal: " +
       ", ".join(f"{k} {a}/{t}" for k, (a, t) in res.items()))


def test_plan_shape():
    p = Params()
    cfg = wl.make_cfg("XAU/USD", "metal", "rsi2")
    d = signals(add_indicators(load_history(cfg), p), p, cfg)
    pl = plan_levels(d, p, cfg)
    side = pl["long"] or pl["short"]
    assert side and {"trigger", "op", "entry", "stop", "target", "dist_pct"} <= set(side)
    if pl["long"]:
        L = pl["long"]
        assert L["stop"] < L["entry"] < L["target"]
    if pl["short"]:
        S = pl["short"]
        assert S["target"] < S["entry"] < S["stop"]
    ok("plan has trigger / entry / stop / target on the correct sides")


# ------------------------------------------------------------------ 4) credits / Binance paging
def test_credit_tiers_exact():
    t = [0.0]
    b = RateBudget(1000, clock=lambda: t[0], per_day=100)
    n = {"scan": 0, "bars": 0, "quote": 0}
    for tier in ("scan", "bars", "quote"):
        while b.try_acquire(1.0, tier):
            n[tier] += 1
            t[0] += 1
    assert n == {"scan": 70, "bars": 0, "quote": 30}, n      # bars cap = 100-60 = 40 < 70 already used
    t[0] += 86400                                           # new day resets
    assert b.try_acquire(1.0, "scan")
    ok(f"credit tiers exact: {n}, reset next day")


def test_binance_paging():
    calls = []

    class B(Binance):
        def _get(self, path, params):
            calls.append(params)
            end = params.get("endTime", 10 ** 13)
            lim = params["limit"]
            step = 900_000
            last = (end // step) * step if "endTime" in params else 2_000_000_000_000
            rows = [[last - (lim - 1 - k) * step, "1", "1.1", "0.9", "1", "5"] for k in range(lim)]
            return rows
    b = B(Params())
    df = b.bars(wl.make_cfg("BTC/USD", "crypto", "scalp"), 2500)
    assert len(calls) == 3 and len(df) == 2500 and df.index.is_monotonic_increasing
    assert calls[0]["symbol"] == "BTCUSDT"
    ok("Binance: 2500 bars fetched in 3 pages, BTC/USD mapped to BTCUSDT")


# ------------------------------------------------------------------ 5) dashboard API
def test_dashboard_api():
    import dashboard
    got = []
    port = 18000 + os.getpid() % 1000
    dashboard.Dashboard(lambda: {"symbols": {}}, lambda c: "ok", time.time, "pw", "tok", port=port,
                        watch_fn=lambda a, b: got.append((a, b)) or "done").start()
    time.sleep(0.3)
    auth = "Basic " + base64.b64encode(b"u:pw").decode()

    def post(body, a=auth):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/watchlist", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": a})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, {}
    assert post({"action": "add", "symbol": "SPY", "strategies": ["scalp"]}) == (200, {"ok": True, "message": "done"})
    assert post({"action": "nuke"})[0] == 400
    assert post({"action": "add"}, a="Basic " + base64.b64encode(b"u:wrong").decode())[0] == 401
    assert got == [("add", {"action": "add", "symbol": "SPY", "strategies": ["scalp"]})]
    assert "addSym()" in dashboard.PAGE and "renderWL" in dashboard.PAGE
    ok("dashboard /api/watchlist: add works with the password, wrong password 401, unknown action 400")


def test_runtime_watchlist_actions():
    e = _gate_env({"status": "ok", "at": time.time()})
    f = os.path.join(e.tmp, "wl.json")
    okk, msg, restart = e.rt.watchlist_action("add", {"symbol": "XAG/USD", "strategies": ["rsi2"]}, f)
    assert okk and restart and "XAG/USD" in [x["symbol"] for x in wl.load(f)]
    okk, msg, restart = e.rt.watchlist_action("off", {"id": "XAG/USD"}, f)
    assert okk and not restart and "XAG/USD" in wl.disabled_ids(wl.load(f))
    assert e.run(400, until=lambda: e.ss.eng.pos is not None)
    e.ss.cfg.provider_symbol = "XAG/USD"            # pretend the open position is on XAG/USD
    okk, msg, _ = e.rt.watchlist_action("remove", {"symbol": "XAG/USD"}, f)
    assert not okk and "مرفوض" in msg
    okk, msg, _ = e.rt.watchlist_action("add", {"symbol": "$$$", "strategies": ["scalp"]}, f)
    assert not okk
    ok("runtime: add -> restart, off -> live, remove refused while a position is open, junk refused")


def test_forward_kill_switch():
    e = _gate_env({"status": "ok", "at": time.time()})
    ss = e.ss
    for _ in range(19):
        e.rt._forward_check(ss, -0.3, -1.0)
    assert not ss.auto_off and e.rt._trading(ss), "must wait for 20 forward trades"
    e.rt._forward_check(ss, -0.3, -1.0)
    assert ss.auto_off and not e.rt._trading(ss), "20 losing forward trades must switch the stream off"
    st = e.rt.snapshot()
    rt2 = e.new_runtime()
    rt2.restore(st)
    assert rt2.syms[0].auto_off and rt2.syms[0].fw["n"] == 20, "kill-switch must survive a restart"
    f = os.path.join(e.tmp, "wl2.json")
    wl.save([{"symbol": "TEST/USD", "class": "crypto", "strategies": ["scalp"]}], f)
    rt2.by_name["TEST/USD-SCALP"] = rt2.syms[0]
    okk, msg, _ = rt2.watchlist_action("on", {"id": "TEST/USD-SCALP"}, f)
    assert okk and not rt2.syms[0].auto_off and rt2.syms[0].fw["n"] == 0
    e2 = _gate_env({"status": "ok", "at": time.time()})
    for _ in range(25):
        e2.rt._forward_check(e2.ss, 0.2, 1.0)
    assert not e2.ss.auto_off
    import remote
    link = remote.RemoteLink("", "", lambda: e.rt.status(e.clock.now), lambda c: "", lambda: None)
    cs = link.compact_status()
    assert "fw" in cs and "q" in cs and "tr" in cs
    ok("forward kill-switch: 20 real paper trades averaging < -0.05R switch a stream off (persisted); owner can re-enable")


def test_bar_queue_priority():
    """Bar requests: (a) open position / pending entry, (b) auto-traded, (c) watch-only - the credit bucket serves
    the first callers first, so a watch-only line must never delay gold / EUR/USD at the top of the hour."""
    from broker import PaperBroker
    from infra import StateStore

    class Prov:
        budget = None
    old16 = wl.streams([wl.validate_entry(e) for e in [{"symbol": "QQQ", "class": "etf", "strategies": ["breakout", "scalp"]},
                                                       {"symbol": "DIA", "class": "etf", "strategies": ["scalp", "breakout"]}]
                        + wl.DEFAULT[1:]])
    tmp = tempfile.mkdtemp()
    rt = bot.Runtime(Params(), old16, {"twelvedata": Prov(), "binance": Prov()}, PaperBroker(),
                     StateStore(os.path.join(tmp, "s.json")), tr.Rec(), tr.SyncExecutor(), gate=True,
                     stop_file=os.path.join(tmp, "STOP"), flatten_file=os.path.join(tmp, "FLATTEN"),
                     trades_file=os.path.join(tmp, "t.csv"))
    for sid in ("QQQ", "QQQ-SCALP", "XAU/USD", "EUR/USD"):
        rt.by_name[sid].qual = {"status": "ok", "at": time.time()}
    for sid in rt.by_name:
        rt.by_name[sid].next_qual_try = 1e18
    ids = [s.cfg.symbol for s in rt.syms]
    assert ids.index("DIA-SCALP") < ids.index("XAU/USD")                       # the old watchlist order
    q = [s.cfg.symbol for s in rt.bar_queue()]
    assert q[:4] == ["QQQ", "QQQ-SCALP", "XAU/USD", "EUR/USD"], q              # traded first, in watchlist order
    assert q.index("XAU/USD") < q.index("DIA-SCALP") and q.index("EUR/USD") < q.index("DIA-BRK"), q
    rt.by_name["GBP/USD"].eng.pos = object()                                   # watch-only line holding a position
    q = [s.cfg.symbol for s in rt.bar_queue()]
    assert q[0] == "GBP/USD" and q.index("GBP/USD") < q.index("USD/JPY"), q
    called = []                                                                # step() really uses this order
    rt._schedule_bars = lambda ss, now: called.append(ss.cfg.symbol)
    for name in ("_schedule_quote", "_schedule_qual"):
        setattr(rt, name, lambda ss, now: None)
    for name in ("_harvest_qual", "_harvest", "_guards", "_housekeeping"):
        setattr(rt, name, lambda now: None)
    rt.step(time.time())
    assert called == q, (called, q)
    rt.by_name["GBP/USD"].eng.pos = None
    assert rt.status(time.time())["bar_queue"][:3] == ["QQQ", "QQQ-SCALP", "XAU/USD"]
    ok("bar queue: open position > auto-traded > watch-only; XAU/USD + EUR/USD requested before DIA-SCALP/DIA-BRK")


if __name__ == "__main__":
    real = bot.signals
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            bot.signals = real
            fn()
    bot.signals = real
    print(f"\nall {len(PASSED)} watchlist tests passed")
