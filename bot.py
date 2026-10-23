"""Paper-trading bot (simulated fills, real market data) - low-latency event loop.

    python bot.py

How it is fast:
  * Exits are checked on EVERY live price (every `quote_poll_sec`, default 2 s) - not once per
    5-minute bar. Stops/targets react in seconds; the bar-based check stays as a backstop.
  * The new 5-minute bar is polled every ~1.5 s right after it closes (not a fixed 8 s sleep), the
    signal is evaluated at once and the entry is filled on the first live price after the bar open.
  * All network calls run in worker threads with hard timeouts; the main loop (ticking every 0.2 s)
    NEVER blocks on the network, so a slow API can never delay an exit on another symbol.

How it avoids errors:  see README ("طبقات الحماية") - crash-safe state, single-instance lock, bad-tick
filter, NYSE calendar, credit budget, stale-data guards, broker-reject retry, watchdog.

Kill switch: create a file named STOP -> no new entries (open positions stay managed).
             create a file named FLATTEN -> close everything at the next price (+ no new entries).

LIVE TRADING IS NOT IMPLEMENTED ON PURPOSE (see broker.py for what a live adapter must do).
"""
from __future__ import annotations

import csv
import logging
import math
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace as dc_replace
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd

import fx_hours
import market_calendar as cal
from broker import Broker, Fill, PaperBroker
import watchlist as wl
from config import (ConfigError, Params, SymbolCfg, apply_env_overrides, get_params, load_env,
                    validate_config)
from config import SYMBOLS as ALL_KNOWN
from data import BarStore, redact
from engine import Account, Position, SymbolEngine
from guardian import GuardCfg, Guardian
from infra import Notifier, SingleInstance, StateStore, sd_notify, setup_logging
from providers import ProviderError, Quote, QuoteFilter, make_providers
import qualify
from strategy import add_indicators, plan_levels, signals

STATE_FILE = "state.json"
TRADES_FILE = "trades_live.csv"
STOP_FILE = "STOP"
FLATTEN_FILE = "FLATTEN"
STATUS_FILE = "status.json"
TRADE_FIELDS = ["symbol", "side", "entry_ts", "exit_ts", "entry", "exit", "qty", "pnl_gross", "cost",
                "pnl", "R", "reason", "equity_after"]
log = logging.getLogger("bot")


# ---------------------------------------------------------------- time helpers
def local_naive(epoch: float, tz: str) -> datetime:
    return datetime.fromtimestamp(epoch, ZoneInfo(tz)).replace(tzinfo=None)


def epoch_of(naive_local: datetime, tz: str) -> float:
    return naive_local.replace(tzinfo=ZoneInfo(tz)).timestamp()


def _timed(fn, *a):
    t0 = time.time()
    return fn(*a), time.time() - t0


# ---------------------------------------------------------------- per-symbol runtime state
@dataclass
class Pending:
    side: int
    atr: float
    ref_close: float
    signal_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    nominal: float          # epoch of the entry bar's open
    deadline: float


class SymState:
    def __init__(self, cfg: SymbolCfg, eng: SymbolEngine, bar_prov, quote_prov, p: Params):
        self.cfg, self.eng, self.bar_prov, self.quote_prov = cfg, eng, bar_prov, quote_prov
        self.itv = timedelta(minutes=cfg.interval_minutes)
        self.same_source = cfg.provider == cfg.quote_src
        self.store = BarStore(cfg.interval_minutes, keep=max(300, cfg.history_bars))
        self.filter = QuoteFilter(p.max_tick_jump_pct)
        self.quote: Optional[Quote] = None
        self.last_eval: Optional[pd.Timestamp] = None      # start of the last CLOSED bar evaluated
        self.last_signal_ts: Optional[pd.Timestamp] = None  # last signal bar already handled (no duplicates)
        self.pending: Optional[Pending] = None
        self.needs_full = True
        self.quote_future = self.bar_future = None
        self.quote_started = self.bar_started = 0.0
        self.next_quote_try = self.next_bar_try = 0.0
        self.exit_retry_at = 0.0
        self.errs = {"quote": 0, "bar": 0}
        self.lat: Dict[str, Optional[float]] = {"quote": None, "bar": None}
        self.poll = cfg.poll_sec or p.quote_poll_sec
        self.atr: Optional[float] = None                   # latest ATR (for the guardian's shock check)
        self.qual: Optional[dict] = None                   # qualification result (qualify.py)
        self.qual_future = None
        self.qual_started = 0.0
        self.next_qual_try = 0.0
        self.plan: Optional[dict] = None                   # pre-computed entry/stop/target levels
        self.disabled = False                              # switched off by the user (dashboard)
        self.fw = {"n": 0, "w": 0, "sumR": 0.0, "pnl": 0.0}  # forward (paper) results since v7
        self.auto_off = ""                                 # reason if the forward kill-switch turned it off
        self.flat_min = 16 * 60 - (int(cfg.eod_time[:2]) * 60 + int(cfg.eod_time[3:]))   # 5 for "15:55"


# ---------------------------------------------------------------- the bot
class Runtime:
    def __init__(self, p: Params, symbols, providers: dict, broker: Broker, store: StateStore,
                 notifier: Notifier, executor, clock: Callable[[], float] = time.time,
                 stop_file: str = STOP_FILE, flatten_file: str = FLATTEN_FILE,
                 trades_file: str = TRADES_FILE, mode: str = "watchlist", gate: bool = False,
                 classes: Optional[dict] = None):
        self.p, self.broker, self.store, self.n = p, broker, store, notifier
        self.ex, self.clock = executor, clock
        self.stop_file, self.flatten_file, self.trades_file = stop_file, flatten_file, trades_file
        self.mode = mode
        self.gate = gate                     # True (live): a stream trades only after passing qualify.py
        self.classes = classes or {}
        self.acct = Account(p.initial_equity, p)
        self.syms = [SymState(s, SymbolEngine(s, p, self.acct), providers[s.provider],
                              providers[s.quote_src], p) for s in symbols]
        self.by_name = {s.cfg.symbol: s for s in self.syms}
        self.providers = providers
        self.stopping = threading.Event()
        self.news = None                                         # news.NewsWatcher (context only, never trades)
        self.dirty = False
        self.last_save = 0.0
        self.last_wd = self.last_hb = 0.0
        self.flatten_req = False
        self.step_errors = 0
        self.t0 = clock()
        self.guard = Guardian(GuardCfg.from_env(), lambda m, lvl=logging.WARNING: self.n.send(m, level=lvl),
                              clock=clock, halt_file=os.path.join(os.path.dirname(stop_file), "HALT"))
        self.status_file = os.path.join(os.path.dirname(trades_file), STATUS_FILE)
        self.last_status = 0.0
        self.recent_trades: list = []
        self.guard.on_equity(self.acct.equity)
        self.last_alive = clock()

    # ------------------------------------------------------------ trading mode (from the phone)
    def watchlist_action(self, action: str, data: dict, path: str = "") -> tuple:
        """Dashboard edits of the watchlist. Returns (ok, message, needs_restart).
        add/remove change which streams exist -> restart; on/off is applied live."""
        path = path or wl.WATCHLIST_FILE
        try:
            items = wl.load(path)
            action = (action or "").strip().lower()
            if action == "add":
                items = wl.add_symbol(items, str(data.get("symbol", "")), data.get("class") or None,
                                      list(data.get("strategies") or ["scalp"]))
                new = items[-1]
                cfgs = [wl.make_cfg(new["symbol"], new["class"], st) for st in new["strategies"]]
                validate_config(self.p, cfgs) if os.getenv("TWELVE_DATA_API_KEY") else None
                wl.save(items, path)
                return True, (f"تمت إضافة {new['symbol']} ({wl.CLASS_AR[new['class']]}) - "
                              f"{'، '.join(wl.STRAT_AR[x] for x in new['strategies'])}. يعيد البوت التشغيل الآن، "
                              f"ثم يختبر كل استراتيجية على تاريخ الرمز قبل أي تداول آلي."), True
            if action == "remove":
                sym = str(data.get("symbol", "")).upper().strip()
                busy = [s.cfg.symbol for s in self.syms if s.cfg.ticker == sym and (s.eng.pos or s.pending)]
                if busy:
                    return False, f"مرفوض: صفقة مفتوحة على {', '.join(busy)}. أغلقها أولاً.", False
                items = wl.remove_symbol(items, sym)
                wl.save(items, path)
                return True, f"تم حذف {sym} من قائمة المراقبة. يعيد البوت التشغيل الآن.", True
            if action in ("on", "off"):
                sid = str(data.get("id", ""))
                items = wl.set_stream_enabled(items, sid, action == "on")
                wl.save(items, path)
                ss = self.by_name.get(sid)
                if ss is not None:
                    ss.disabled = action == "off"
                    if action == "on" and ss.auto_off:
                        ss.auto_off = ""                       # the owner decided to give it another chance
                        ss.fw = {"n": 0, "w": 0, "sumR": 0.0, "pnl": 0.0}
                        self.dirty = True
                if action == "off":
                    return True, f"{sid}: أُوقف التداول الآلي (المراقبة والخطة مستمرتان؛ الصفقة المفتوحة تُدار حتى تُغلق).", False
                ok_q = ss is not None and ss.qual and ss.qual.get("status") == "ok"
                return True, (f"{sid}: التداول الآلي مفعّل." if ok_q or not self.gate else
                              f"{sid}: مفعّل، لكنه لن يتداول إلا بعد أن ينجح في اختبار تاريخه."), False
            return False, "إجراء غير معروف", False
        except ConfigError as e:
            return False, str(e), False

    # ------------------------------------------------------------ persistence
    def snapshot(self) -> dict:
        a = self.acct
        return {
            "version": 2,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "equity": a.equity,
            "day": a.day.isoformat() if a.day else None,
            "day_start_equity": a.day_start_equity,
            "trades_today": a.trades_today,
            "consec_losses": a.consec_losses,
            "guardian": self.guard.snapshot(),
            "symbols": {
                s.cfg.symbol: {
                    "last_eval": s.last_eval.isoformat() if s.last_eval is not None else None,
                    "last_signal_ts": s.last_signal_ts.isoformat() if s.last_signal_ts is not None else None,
                    "position": s.eng.pos.to_dict() if s.eng.pos else None,
                    "qual": s.qual,
                    "fw": s.fw,
                    "auto_off": s.auto_off,
                } for s in self.syms
            },
        }

    def restore(self, st: dict) -> None:
        a = self.acct
        a.equity = float(st["equity"])
        a.day = date.fromisoformat(st["day"]) if st.get("day") else None
        a.day_start_equity = float(st.get("day_start_equity", a.equity))
        a.trades_today = int(st.get("trades_today", 0))
        a.consec_losses = int(st.get("consec_losses", 0))
        self.guard.restore(st.get("guardian"))
        self.guard.on_equity(a.equity)
        if "symbols" in st:                                   # v2
            per = st["symbols"]
        else:                                                 # v1 (older bot version)
            per = {}
            for k, v in st.get("last_eval", {}).items():
                per.setdefault(k, {})["last_eval"] = v
            for k, v in st.get("positions", {}).items():
                per.setdefault(k, {})["position"] = v
        for sym, d in per.items():
            ss = self.by_name.get(sym)
            if ss is None:
                if d.get("position"):
                    self.n.send(f"WARNING: saved position on {sym} but it is not in SYMBOLS - "
                                f"it will NOT be managed!", level=logging.ERROR)
                continue
            if d.get("last_eval"):
                ss.last_eval = pd.Timestamp(d["last_eval"])
            if d.get("last_signal_ts"):
                ss.last_signal_ts = pd.Timestamp(d["last_signal_ts"])
            if d.get("position"):
                ss.eng.pos = Position.from_dict(d["position"])
            if isinstance(d.get("qual"), dict):
                ss.qual = d["qual"]
            if isinstance(d.get("fw"), dict):
                ss.fw.update({k: d["fw"][k] for k in ss.fw if k in d["fw"]})
            ss.auto_off = str(d.get("auto_off") or "")
        log.info("state restored: equity=%.2f open=%s", a.equity,
                 [s.cfg.symbol for s in self.syms if s.eng.pos])

    def _save(self, now: float, force: bool = False) -> None:
        if not (force or self.dirty or now - self.last_save > 30):
            return
        try:
            self.store.save(self.snapshot())
            self.dirty = False
            self.last_save = now
        except Exception as e:
            self.n.send(f"STATE SAVE FAILED: {type(e).__name__}: {e}", key="save", every=60, level=logging.ERROR)

    def _log_trade(self, tr: dict) -> None:
        try:
            new = not os.path.exists(self.trades_file)
            with open(self.trades_file, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=TRADE_FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(tr)
        except OSError as e:                                  # never let a full disk block an exit
            self.n.send(f"could not write {self.trades_file}: {e}", key="csv", every=300, level=logging.ERROR)

    # ------------------------------------------------------------ small helpers
    def _local(self, ss: SymState, now: float) -> datetime:
        return local_naive(now, ss.cfg.tz)

    def _utc_day(self, now: float) -> date:
        return datetime.fromtimestamp(now, timezone.utc).date()

    def _is_stock(self, ss: SymState) -> bool:
        return ss.cfg.kind == "stock"

    def _is_fx(self, ss: SymState) -> bool:
        return ss.cfg.kind == "fx"

    def _fx_open(self, ss: SymState, now: float) -> bool:
        return fx_hours.is_open(self._local(ss, now), ss.cfg.metal)

    def _active(self, ss: SymState, now: float) -> bool:
        """Is it worth polling this symbol right now?"""
        if self._is_fx(ss):
            return self._fx_open(ss, now)
        if not self._is_stock(ss):
            return True
        return cal.active_window(self._local(ss, now), 5, 30 if ss.eng.pos else 3)

    def _entries_allowed(self, ss: SymState, now: float) -> bool:
        if os.path.exists(self.stop_file) or self.flatten_req or ss.cfg.exit_only:
            return False
        if not self._trading(ss):
            return False
        if max(ss.errs.values()) >= self.p.max_error_streak:
            return False
        block = self.guard.entry_block(ss.cfg.symbol, now)
        if block:
            self.n.send(f"{ss.cfg.symbol}: entry blocked by guardian - {block}",
                        key=f"gblock-{ss.cfg.symbol}", every=900, level=logging.WARNING)
            return False
        if self._is_stock(ss):
            local = self._local(ss, now)
            sess = cal.session(local.date())
            if sess is None:
                return False
            if ss.cfg.flatten_eod and local >= sess[1] - timedelta(minutes=ss.flat_min + 3):
                return False
        if self._is_fx(ss):
            local = self._local(ss, now)
            if not self._fx_open(ss, now):
                return False
            if ss.cfg.flatten_eod and local.strftime("%H:%M") >= self._minus3(ss.cfg.eod_time):
                return False
        return True

    def _trading(self, ss: SymState) -> bool:
        """May this stream open trades automatically? (qualified by its own backtest, not switched off)"""
        if ss.disabled or ss.cfg.exit_only or ss.auto_off:
            return False
        return (not self.gate) or bool(ss.qual and ss.qual.get("status") == "ok")

    def _in_watch_hours(self, ss: SymState, now: float) -> bool:
        """Bars are only worth their credits around the entry windows (or while a trade is on)."""
        if ss.eng.pos is not None or ss.pending is not None or ss.last_eval is None or ss.needs_full:
            return True
        if not ss.cfg.windows or ss.cfg.provider == "binance":
            return True
        hhmm = self._local(ss, now).strftime("%H:%M")
        m = int(hhmm[:2]) * 60 + int(hhmm[3:])
        pad = 2 * ss.cfg.interval_minutes
        for a, b in ss.cfg.windows:
            lo = int(a[:2]) * 60 + int(a[3:]) - pad
            hi = int(b[:2]) * 60 + int(b[3:]) + ss.cfg.interval_minutes
            if lo <= m <= hi:
                return True
        return False

    FW_MIN_TRADES = 20
    FW_MIN_AVG_R = -0.05

    def _forward_check(self, ss: SymState, R: float, pnl: float) -> None:
        """Forward kill-switch: the backtest said this stream should make money. If the REAL paper results
        disagree (>= 20 trades averaging below -0.05R) the stream stops auto-trading until the owner
        re-enables it from the dashboard. Only ever reduces risk."""
        f = ss.fw
        f["n"] += 1
        f["w"] += int(R > 0)
        f["sumR"] = round(f["sumR"] + R, 4)
        f["pnl"] = round(f["pnl"] + pnl, 2)
        self.dirty = True
        if not ss.auto_off and f["n"] >= self.FW_MIN_TRADES and f["sumR"] / f["n"] < self.FW_MIN_AVG_R:
            ss.auto_off = (f"النتائج الفعلية أسوأ من الاختبار: {f['n']} صفقة بمتوسط "
                           f"{f['sumR'] / f['n']:+.2f}R - أُوقف تلقائياً")
            self.n.send(f"{ss.cfg.symbol}: FORWARD KILL-SWITCH - {f['n']} trades avg {f['sumR'] / f['n']:+.3f}R "
                        f"-> auto-trading OFF (re-enable from the dashboard)", level=logging.WARNING)

    def _schedule_qual(self, ss: SymState, now: float) -> None:
        if not self.gate or ss.qual_future is not None or now < ss.next_qual_try or ss.disabled:
            return
        q = ss.qual
        fresh = q and q.get("status") in ("ok", "fail") and now - float(q.get("at", 0)) < qualify.MAX_AGE_H * 3600
        if fresh:
            return
        b = ss.bar_prov.budget
        if b is not None and not b.try_acquire(1.0, "qual"):
            return
        ss.next_qual_try = now + 1800.0                  # retry an error after 30 min
        p, cfg, prov = self.p, ss.cfg, ss.bar_prov

        def job():
            df = prov.bars(cfg, qualify.HISTORY_BARS)
            return qualify.qualify(df, cfg, p)
        ss.qual_future = self.ex.submit(_timed, job)
        ss.qual_started = now

    def _harvest_qual(self, now: float) -> None:
        for ss in self.syms:
            fut = ss.qual_future
            if fut is None:
                continue
            if not fut.done():
                if now - ss.qual_started > 120:
                    ss.qual_future = None
                continue
            ss.qual_future = None
            try:
                res, _ = fut.result()
            except Exception as e:
                ss.qual = {"status": "error", "at": now, "reason": redact(f"{type(e).__name__}: {e}")[:160]}
                continue
            old = (ss.qual or {}).get("status")
            ss.qual = res
            self.dirty = True
            if res.get("status") != old:
                verdict = "AUTO-TRADE ON" if res["status"] == "ok" else "watch only"
                self.n.send(f"{ss.cfg.symbol}: qualification {res['status']} ({verdict}) - n={res.get('n')} "
                            f"win={res.get('wr')}% avgR={res.get('avgR')} | {res.get('reason')}")

    @staticmethod
    def _minus3(hhmm: str) -> str:
        m = int(hhmm[:2]) * 60 + int(hhmm[3:]) - 3
        return f"{m // 60:02d}:{m % 60:02d}"

    # ------------------------------------------------------------ scheduling (never blocks)
    def _submit(self, ss: SymState, kind: str, prov, fn, *args) -> None:
        setattr(ss, f"{kind}_future", self.ex.submit(_timed, fn, *args))
        setattr(ss, f"{kind}_started", self.clock())

    def _schedule_quote(self, ss: SymState, now: float) -> None:
        if ss.quote_future is not None or now < ss.next_quote_try:
            return
        if ss.eng.pos is None and ss.pending is None:
            return
        if not self._active(ss, now):
            return
        b = ss.quote_prov.budget
        if b is not None and not b.try_acquire(1.0):
            return
        ss.next_quote_try = now + (self.p.entry_poll_sec if ss.pending else ss.poll)
        self._submit(ss, "quote", ss.quote_prov, ss.quote_prov.quote, ss.cfg)

    def _schedule_bars(self, ss: SymState, now: float) -> None:
        if ss.bar_future is not None or now < ss.next_bar_try or not self._active(ss, now):
            return
        if not self._in_watch_hours(ss, now):
            return
        local = self._local(ss, now)
        if self._is_stock(ss) and not ss.store.df.empty:
            sess = cal.session(local.date())               # no new bar can exist before the first one closes
            if sess is None or local < sess[0] + ss.itv + timedelta(seconds=self.p.bar_settle_sec):
                return
        L = ss.last_eval
        due_at = None if L is None else L + 2 * ss.itv + timedelta(seconds=self.p.bar_settle_sec)
        if L is not None and self._is_fx(ss):
            nxt = L + ss.itv                               # skip bars that cannot exist (break / weekend)
            for _ in range(24 * 60):
                if fx_hours.bar_is_open(nxt.to_pydatetime(), ss.cfg.metal):
                    break
                nxt += ss.itv
            due_at = nxt + ss.itv + timedelta(seconds=self.p.bar_settle_sec)
        if not (L is None or ss.needs_full or local >= due_at):
            return
        b = ss.bar_prov.budget
        tier = "bars" if (self._trading(ss) or ss.eng.pos is not None) else "scan"
        if b is not None and not b.try_acquire(1.0, tier):
            return
        overdue = 0.0 if due_at is None else (local - due_at).total_seconds()
        ss.next_bar_try = now + (self.p.bar_retry_sec if overdue < 60 else 5.0 if overdue < 300 else 15.0)
        n = ss.store.keep if (ss.needs_full or ss.store.df.empty) else 6
        self._submit(ss, "bar", ss.bar_prov, ss.bar_prov.bars, ss.cfg, n)
        if overdue > 3 * ss.itv.total_seconds() and L is not None:
            self.n.send(f"{ss.cfg.symbol}: no new bar for {overdue/60:.0f} min (data stalled?)",
                        key=f"stale-{ss.cfg.symbol}", every=300, level=logging.WARNING)

    def _on_err(self, ss: SymState, kind: str, e: ProviderError, now: float) -> None:
        ss.errs[kind] += 1
        delay = min(30.0, 2.0 ** min(ss.errs[kind] - 1, 5))
        if kind == "quote" and ss.eng.pos is not None:
            delay = min(delay, 5.0)                           # keep hammering while money is at risk
        delay = max(delay, e.retry_after)
        if e.kind == "auth":
            delay = max(delay, 60.0)
            self.n.send(f"{ss.cfg.symbol}: DATA API AUTH ERROR - check TWELVE_DATA_API_KEY / plan ({e})",
                        key="auth", every=600, level=logging.ERROR)
        if kind == "quote":
            ss.next_quote_try = max(ss.next_quote_try, now + delay)
        else:
            ss.next_bar_try = max(ss.next_bar_try, now + delay)
        self.n.send(f"{ss.cfg.symbol}: {kind} error #{ss.errs[kind]} -> {e}",
                    key=f"err-{ss.cfg.symbol}-{kind}", every=60, level=logging.WARNING)

    def _harvest(self, now: float) -> None:
        for kind in ("quote", "bar"):
            for ss in self.syms:
                fut = getattr(ss, f"{kind}_future")
                if fut is None:
                    continue
                if not fut.done():
                    if now - getattr(ss, f"{kind}_started") > 15.0:      # abandon a stuck request
                        setattr(ss, f"{kind}_future", None)
                        self._on_err(ss, kind, ProviderError("network", "request stuck > 15 s"), now)
                    continue
                setattr(ss, f"{kind}_future", None)
                try:
                    res, lat = fut.result()
                except ProviderError as e:
                    self._on_err(ss, kind, e, now)
                    continue
                except Exception as e:                                     # a provider bug must not kill the loop
                    self._on_err(ss, kind, ProviderError("server", redact(f"{type(e).__name__}: {e}")), now)
                    continue
                ss.lat[kind] = lat if ss.lat[kind] is None else 0.8 * ss.lat[kind] + 0.2 * lat
                if kind == "quote":
                    self._apply_quote(ss, res, now)
                else:
                    self._apply_bars(ss, res, now)

    # ------------------------------------------------------------ live prices -> exits / entries
    def _apply_quote(self, ss: SymState, q: Quote, now: float) -> None:
        q = ss.filter.accept(q)
        if q is None:
            self.n.send(f"{ss.cfg.symbol}: tick rejected by sanity filter", key=f"tick-{ss.cfg.symbol}",
                        every=120, level=logging.WARNING)
            return
        ss.errs["quote"] = 0
        ss.quote = q
        pos = ss.eng.pos
        shock = self.guard.on_price(ss.cfg.symbol, q.mid, ss.atr, pos.side if pos else 0, now)
        if self.guard.changed:
            self.guard.changed, self.dirty = False, True
        if shock and pos is not None:
            px = q.bid if pos.side == 1 else q.ask
            self._do_exit(ss, "guardian_shock", px, now, latency_from=q.ts)
        if ss.eng.pos is not None:
            self._manage_position(ss, q, now)
        if ss.pending is not None:
            self._try_fill_pending(ss, q, now)

    def _manage_position(self, ss: SymState, q: Quote, now: float) -> None:
        pos = ss.eng.pos
        px = q.bid if pos.side == 1 else q.ask
        reason: Optional[str] = None
        ref = px
        if self.flatten_req:
            reason = "manual_flatten"
        elif self._is_stock(ss) and ss.cfg.flatten_eod:
            local = self._local(ss, now)
            sess = cal.session(local.date())
            if pos.entry_ts.date() < local.date():
                if sess is not None and local >= sess[0]:
                    reason = "overnight"                       # never carry stock overnight
            elif sess is not None and local >= sess[1] - timedelta(minutes=ss.flat_min):
                reason = "eod"
        elif self._is_fx(ss) and ss.cfg.flatten_eod:
            local = self._local(ss, now)
            if pos.entry_ts.date() < local.date():
                reason = "overnight"                           # bot was down at 16:45 -> close at once
            elif local.strftime("%H:%M") >= ss.cfg.eod_time:
                reason = "eod"                                 # daily flat: no overnight / weekend risk
        if reason is None:
            sig = ss.eng.check_tick(q.bid, q.ask, q.last)
            if sig:
                reason, ref = sig
        if reason:
            self._do_exit(ss, reason, ref, now, latency_from=q.ts)

    def _do_exit(self, ss: SymState, reason: str, ref: float, now: float, latency_from: float = 0.0,
                 ts: Optional[pd.Timestamp] = None) -> None:
        pos = ss.eng.pos
        if pos is None or now < ss.exit_retry_at:
            return
        cid = f"{ss.cfg.symbol}-x-{int(now)}"
        try:
            fill = self.broker.market_exit(ss.cfg.symbol, pos.side, pos.qty, ref, cid, ticker=ss.cfg.ticker)
        except Exception as e:                                # a broker bug must never lose the position
            fill = Fill(False, reason=f"{type(e).__name__}: {e}")
        if not fill.ok:
            ss.exit_retry_at = now + 1.0
            self.n.send(f"{ss.cfg.symbol}: EXIT REJECTED ({reason}): {fill.reason} - retrying",
                        key=f"rej-{ss.cfg.symbol}", every=30, level=logging.ERROR)
            return
        tr = ss.eng.close_position(ts if ts is not None else pd.Timestamp(self._local(ss, now)),
                                   fill.price, reason)
        self._save(now, force=True)                           # state first, alerts after
        if tr is None:
            return
        self._log_trade(tr)
        self.recent_trades = (self.recent_trades + [{k: str(v) for k, v in tr.items()}])[-20:]
        try:
            self.guard.on_trade(tr["symbol"], float(tr["R"]), self.acct.equity, now)
        except Exception:
            log.exception("guardian on_trade failed")
        try:
            self._forward_check(ss, float(tr["R"]), float(tr["pnl"]))
        except Exception:
            log.exception("forward check failed")
        self.dirty = True
        lat = f" latency={now - latency_from:.1f}s" if latency_from else ""
        self.n.send(f"EXIT {tr['symbol']} {tr['side']} {tr['reason']} pnl={tr['pnl']:.2f} "
                    f"({tr['R']:.2f}R) equity={self.acct.equity:.2f}{lat}")

    # ------------------------------------------------------------ bars -> signals -> pending entry
    def _apply_bars(self, ss: SymState, df: pd.DataFrame, now: float) -> None:
        ss.errs["bar"] = 0
        if self._is_fx(ss) and df is not None and not df.empty:
            keep = [fx_hours.bar_is_open(t.to_pydatetime(), ss.cfg.metal) for t in df.index]
            df = df[keep]                                      # drop the provider's closed-market filler bars
        res = ss.store.merge(df)
        if res == "gap":                                       # missed bars / new session -> full download
            ss.needs_full, ss.next_bar_try = True, now
            return
        if res == "empty":
            return
        closed = ss.store.closed(self._local(ss, now), self.p.bar_settle_sec)
        if len(closed) < 150:
            ss.needs_full = True
            ss.next_bar_try = now + 30.0
            self.n.send(f"{ss.cfg.symbol}: only {len(closed)} bars available (need 150)",
                        key=f"few-{ss.cfg.symbol}", every=600, level=logging.WARNING)
            return
        ss.needs_full = False
        last_ts = closed.index[-1]
        prev = ss.last_eval
        eng, cfg = ss.eng, ss.cfg

        # 1) backstop exits on newly closed bars (the tick loop normally exits long before this)
        if eng.pos is not None:
            base = prev if prev is not None else eng.pos.entry_ts - ss.itv
            for ts, row in closed[closed.index > base].iterrows():
                pos = eng.pos
                if pos is None:
                    break
                last_of_day = bool(cfg.flatten_eod and (ts + ss.itv).strftime("%H:%M") >= cfg.eod_time)
                # Wicks of the entry bar include prices from before our fill, and bars from a different
                # source than the live prices are not comparable -> only trust wicks when safe.
                wicks = (ss.same_source and ts > pos.entry_ts) or pos.monitor_gap
                sig = eng.bar_exit_signal(float(row.high), float(row.low), float(row.close), last_of_day, wicks)
                if sig:
                    self._do_exit(ss, sig[0] + ("" if sig[0] == "eod" else "_bar"), sig[1], now, ts=ts)

        # 2) new closed bar -> evaluate the signal
        if prev is not None and last_ts <= prev:
            return
        ss.last_eval = last_ts
        self.dirty = True
        d = signals(add_indicators(closed, self.p), self.p, cfg)
        try:
            ss.plan = plan_levels(d, self.p, cfg)
        except Exception as e:                                 # a plan bug must never block trading
            ss.plan = {"note": f"plan error: {type(e).__name__}"}
        side, atr, close = int(d["side"].iloc[-1]), float(d["atr"].iloc[-1]), float(d["close"].iloc[-1])
        ss.atr = atr if math.isfinite(atr) and atr > 0 else ss.atr
        if side != 0:
            self._create_pending(ss, side, atr, close, last_ts, now)

    def _create_pending(self, ss: SymState, side: int, atr: float, close: float,
                        signal_ts: pd.Timestamp, now: float) -> None:
        if ss.last_signal_ts is not None and signal_ts <= ss.last_signal_ts:
            return                                             # already handled (restart / duplicate fetch)
        ss.last_signal_ts = signal_ts                          # at most ONE attempt per signal bar
        self.dirty = True
        sym = ss.cfg.symbol
        entry_ts = signal_ts + ss.itv
        nominal = epoch_of(entry_ts.to_pydatetime(), ss.cfg.tz)
        late = now - nominal
        if late > self.p.max_entry_delay_sec:
            self.n.send(f"{sym}: signal skipped - too late ({late:.0f}s after the bar open)")
            return
        if ss.eng.pos is not None:
            return
        ss.pending = Pending(side, atr, close, signal_ts, entry_ts, nominal, nominal + self.p.max_entry_delay_sec)
        ss.next_quote_try = now                                # fetch a live price immediately
        q = ss.quote
        if q is not None:
            self._try_fill_pending(ss, q, now)

    def _try_fill_pending(self, ss: SymState, q: Quote, now: float) -> None:
        pd_, sym = ss.pending, ss.cfg.symbol
        if pd_ is None:
            return
        if ss.eng.pos is not None:
            ss.pending = None
            return
        if now > pd_.deadline:
            ss.pending = None
            self.n.send(f"{sym}: entry MISSED - no fresh price within {self.p.max_entry_delay_sec:.0f}s of the open")
            return
        if q.ts < pd_.nominal - 0.5 or now - q.ts > 3.0:
            return                                             # wait for a price taken AFTER the bar opened
        px = q.ask if pd_.side == 1 else q.bid
        if not (math.isfinite(px) and px > 0):
            return
        if abs(px - pd_.ref_close) > self.p.max_entry_slip_atr * pd_.atr:
            ss.pending = None
            self.n.send(f"{sym}: entry skipped - price moved {abs(px - pd_.ref_close):.4f} "
                        f"(> {self.p.max_entry_slip_atr} ATR) from the signal close")
            return
        if not self._entries_allowed(ss, now):
            ss.pending = None
            log.info("%s: entry cancelled (STOP/FLATTEN file, data errors, or end of session)", sym)
            return
        plan = ss.eng.plan_entry(pd_.entry_ts, px, pd_.side, pd_.atr, self._utc_day(now))
        if plan is None:
            ss.pending = None
            log.info("%s: entry refused by rules (window / daily limits / shorts off / size 0)", sym)
            return
        cid = f"{sym}-{pd_.signal_ts:%Y%m%d%H%M}"
        try:
            fill = self.broker.market_entry(sym, plan.side, plan.qty, px, cid, stop_dist=plan.stop_dist,
                                            target_dist=plan.target_dist, ticker=ss.cfg.ticker,
                                            conv=ss.cfg.usd_per_quote(px))
        except Exception as e:
            fill = Fill(False, reason=f"{type(e).__name__}: {e}")
        ss.pending = None
        if not fill.ok:
            self.n.send(f"{sym}: ENTRY REJECTED: {fill.reason}", level=logging.ERROR)
            return
        if fill.qty > 0 and abs(fill.qty - plan.qty) > 1e-9:          # broker rounded to its lot step / min lot
            conv = ss.cfg.usd_per_quote(fill.price)
            plan = dc_replace(plan, qty=fill.qty, risk_amt=fill.qty * plan.stop_dist * conv)
            if fill.reason:
                self.n.send(f"{sym}: {fill.reason}")
        pos = ss.eng.open_position(plan, fill.price, opened_epoch=now, client_id=cid)
        self._save(now, force=True)                            # position on disk BEFORE anything else
        self.n.send(f"ENTRY {sym} {'LONG' if pos.side == 1 else 'SHORT'} qty={pos.qty} @ {pos.entry:.4f} "
                    f"stop={pos.stop:.4f} target={pos.target:.4f} (filled {now - pd_.nominal:.1f}s after bar open)")

    # ------------------------------------------------------------ guards + housekeeping
    def _guards(self, now: float) -> None:
        for ss in self.syms:
            pos = ss.eng.pos
            if pos is None:
                continue
            sym = ss.cfg.symbol
            if not self._active(ss, now):
                if self._is_fx(ss):
                    self.n.send(f"{sym}: position open while the market is closed - will close at the next price",
                                key=f"afterclose-{sym}", every=600, level=logging.WARNING)
                if self._is_stock(ss):
                    local = self._local(ss, now)
                    sess = cal.session(local.date())
                    if sess and local > sess[1]:
                        self.n.send(f"{sym}: position still open after the close - will flatten at next open",
                                    key=f"afterclose-{sym}", every=600, level=logging.WARNING)
                continue
            last = ss.quote.ts if (ss.quote and ss.quote.ts >= pos.opened_epoch) else pos.opened_epoch
            last = max(last, self.t0)                      # grace period right after a (re)start
            if now - last > self.p.max_quote_age_sec:
                if not pos.monitor_gap:
                    pos.monitor_gap = True
                    self.dirty = True
                self.n.send(f"{sym}: NO LIVE PRICE for {now - last:.0f}s while a position is open!",
                            key=f"blind-{sym}", every=120, level=logging.ERROR)
        for prov in self.providers.values():
            if abs(getattr(prov, "skew", 0.0)) > 3.0:
                self.n.send(f"{prov.name}: local clock is off by {prov.skew:+.1f}s - fix NTP (chrony / timesyncd)",
                            key=f"skew-{prov.name}", every=600, level=logging.WARNING)

    def _heartbeat(self, now: float) -> None:
        parts = []
        for ss in self.syms:
            q = ss.quote
            age = f"{now - q.ts:.1f}s" if q else "-"
            pos = "flat"
            if ss.eng.pos:
                p_ = ss.eng.pos
                pos = f"{'L' if p_.side == 1 else 'S'} {p_.qty}@{p_.entry:.2f}"
            lat = " ".join(f"{k}={v * 1000:.0f}ms" for k, v in ss.lat.items() if v is not None)
            parts.append(f"{ss.cfg.symbol}[{pos} quote_age={age} {lat} err={ss.errs['quote']}/{ss.errs['bar']}]")
        log.info("hb equity=%.2f trades=%d losses_in_row=%d | %s", self.acct.equity,
                 self.acct.trades_today, self.acct.consec_losses, " ".join(parts))

    def _housekeeping(self, now: float) -> None:
        self._save(now)
        if now - self.last_wd >= 5:
            self.last_wd = now
            sd_notify("WATCHDOG=1")
        if now - self.last_hb >= 60:
            self.last_hb = now
            self._heartbeat(now)
        if now - self.last_status >= 15:
            self.last_status = now
            self._write_status(now)

    def status(self, now: float) -> dict:
        syms = {}
        for ss in self.syms:
            p_ = ss.eng.pos
            q = ss.quote
            syms[ss.cfg.symbol] = {
                "strategy": ss.cfg.strategy, "ticker": ss.cfg.ticker, "interval": ss.cfg.interval,
                "class": self.classes.get(ss.cfg.symbol, ss.cfg.kind),
                "trading": self._trading(ss), "disabled": ss.disabled, "qual": ss.qual, "plan": ss.plan,
                "fw": ss.fw, "auto_off": ss.auto_off,
                "windows": ss.cfg.windows, "win_bg": self._win_bg(ss, now), "allow_short": ss.cfg.allow_short,
                "position": None if p_ is None else {
                    "side": "long" if p_.side == 1 else "short", "qty": p_.qty, "entry": p_.entry,
                    "stop": p_.stop, "target": p_.target,
                    "unrealized": None if q is None else round(
                        p_.side * (q.mid - p_.entry) * p_.qty * ss.cfg.usd_per_quote(q.mid), 2)},
                "last_price": None if q is None else q.mid,
                "quote_age_s": None if q is None else round(now - q.ts, 1),
                "errors": dict(ss.errs)}
        a = self.acct
        market = None
        stock = next((s for s in self.syms if self._is_stock(s)), None)
        if stock is not None:
            ny = self._local(stock, now)
            sess = cal.session(ny.date())
            is_open = bool(sess and sess[0] <= ny < sess[1])
            if is_open:
                text = f"QQQ: السوق الأمريكي مفتوح حتى {sess[1]:%H:%M} نيويورك"
            else:
                nxt = ny.date() if (sess and ny < sess[0]) else ny.date() + timedelta(days=1)
                while cal.session(nxt) is None:
                    nxt += timedelta(days=1)
                o = cal.session(nxt)[0].replace(tzinfo=ZoneInfo("America/New_York")).astimezone(ZoneInfo("Asia/Baghdad"))
                days = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
                text = f"QQQ: السوق الأمريكي مغلق - يفتح {days[o.weekday()]} {o:%H:%M} بتوقيت بغداد"
            market = {"open": is_open, "text": text, "ny_time": ny.strftime("%H:%M")}
        fx_market = None
        fxs = next((s for s in self.syms if self._is_fx(s)), None)
        if fxs is not None:
            ny = self._local(fxs, now)
            if fx_hours.is_open(ny, False):
                w = fxs.cfg.windows[0] if fxs.cfg.windows else ("00:00", "23:59")
                bg = [datetime.combine(ny.date(), datetime.strptime(x, "%H:%M").time()).replace(
                    tzinfo=ZoneInfo("America/New_York")).astimezone(ZoneInfo("Asia/Baghdad")) for x in w]
                fx_market = {"open": True,
                             "text": f"الذهب والعملات: السوق مفتوح (الدخول {bg[0]:%H:%M}-{bg[1]:%H:%M} بغداد)"}
            else:
                o = fx_hours.next_open(ny, False).replace(tzinfo=ZoneInfo("America/New_York")).astimezone(
                    ZoneInfo("Asia/Baghdad"))
                days = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
                fx_market = {"open": False, "text": f"الذهب والعملات: مغلق - يفتح {days[o.weekday()]} {o:%H:%M} بغداد"}
        return {"time_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
                "market": market, "fx_market": fx_market, "mode": self.mode, "credits": self._credits(),
                "execution": self.broker.describe() if hasattr(self.broker, "describe") else {"mode": "paper"},
                "equity": round(a.equity, 2), "day_start_equity": round(a.day_start_equity, 2),
                "trades_today": a.trades_today, "consec_losses": a.consec_losses,
                "stop_file": os.path.exists(self.stop_file), "flatten_file": os.path.exists(self.flatten_file),
                "guardian": self.guard.status(now), "symbols": syms, "recent_trades": self.recent_trades[-10:],
                "news": self.news.snapshot() if self.news is not None else None}

    def _win_bg(self, ss: SymState, now: float) -> str:
        """Entry windows shown in Baghdad time (follows US/EU daylight-saving changes automatically)."""
        if not ss.cfg.windows:
            return "24 ساعة"
        day = self._local(ss, now).date()
        out = []
        for a, b in ss.cfg.windows:
            conv = [datetime.combine(day, datetime.strptime(x, "%H:%M").time()).replace(
                tzinfo=ZoneInfo(ss.cfg.tz)).astimezone(ZoneInfo("Asia/Baghdad")).strftime("%H:%M") for x in (a, b)]
            out.append(f"{conv[0]}-{conv[1]}")
        return "، ".join(out)

    def _credits(self) -> Optional[dict]:
        for prov in self.providers.values():
            b = getattr(prov, "budget", None)
            if b is not None and getattr(b, "per_day", 0):
                return {"used": int(b.used_day), "cap": int(b.per_day)}
        return None

    def _write_status(self, now: float) -> None:
        try:
            import json
            tmp = self.status_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.status(now), f, indent=2, default=str)
            os.replace(tmp, self.status_file)
        except Exception as e:
            log.warning("status write failed: %s", e)

    # ------------------------------------------------------------ main step / loop
    def step(self, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        self.last_alive = time.time()
        self.flatten_req = os.path.exists(self.flatten_file)
        for fn in (self._schedule_quote, self._schedule_bars):     # money-at-risk polling first
            for ss in self.syms:
                try:
                    fn(ss, now)
                except Exception:
                    log.exception("%s: scheduling error", ss.cfg.symbol)
        for ss in self.syms:
            try:
                self._schedule_qual(ss, now)
            except Exception:
                log.exception("%s: qualification scheduling error", ss.cfg.symbol)
        try:
            self._harvest_qual(now)
        except Exception:
            log.exception("qualification harvest error")
        try:
            self._harvest(now)
        except Exception:
            log.exception("harvest error")
            self.n.send("internal error while processing market data (see bot.log)",
                        key="harvest", every=60, level=logging.ERROR)
        try:
            self._guards(now)
        except Exception:
            log.exception("guard error")
        self._housekeeping(now)

    def run(self) -> None:
        sd_notify("READY=1")
        while not self.stopping.is_set():
            t0 = time.time()
            try:
                self.step()
                self.step_errors = 0
            except Exception:
                self.step_errors += 1
                log.exception("unexpected error in main loop (#%d)", self.step_errors)
                self.n.send("BOT LOOP ERROR - see bot.log", key="loop", every=60, level=logging.ERROR)
                if self.step_errors >= 50:
                    self.n.send("too many consecutive loop errors - exiting so the service restarts",
                                level=logging.CRITICAL)
                    self._save(time.time(), force=True)
                    raise SystemExit(1)
            self.stopping.wait(max(0.0, self.p.loop_sec - (time.time() - t0)))
        self._save(time.time(), force=True)
        log.info("stopped cleanly; state saved")


# ---------------------------------------------------------------- entry point
LOG_TAIL = None


def main() -> None:
    global LOG_TAIL
    load_env()
    setup_logging()
    from dashboard import LogTail
    LOG_TAIL = LogTail()
    logging.getLogger().addHandler(LOG_TAIL)
    if os.getenv("MODE", "paper").lower() != "paper":
        raise SystemExit("Only MODE=paper is implemented. Live trading is intentionally not included.")
    p = get_params()
    try:
        items = wl.load()
        symbols = wl.streams(items)
        apply_env_overrides(symbols)
        validate_config(p, symbols)
        mode = "watchlist"
    except ConfigError as e:
        raise SystemExit(f"CONFIG ERROR: {e}")

    lock = SingleInstance("bot.lock")
    if not lock.acquire():
        raise SystemExit("Another bot instance is already running in this folder - refusing to start.")

    notifier = Notifier()
    store = StateStore(STATE_FILE)
    st = store.load()
    # A saved open position on a strategy that is not part of the current mode (e.g. TRADING_MODE was
    # changed by hand) is still managed until it closes - exits only, no new entries.
    for sym, d in ((st or {}).get("symbols") or {}).items():
        if d.get("position") and sym not in {s.symbol for s in symbols}:
            known = {s.symbol: s for s in ALL_KNOWN + wl.streams([wl.validate_entry(e) for e in wl.DEFAULT])}
            if sym in known:
                symbols = symbols + [dc_replace(known[sym], exit_only=True)]
                apply_env_overrides(symbols[-1:])
                notifier.send(f"{sym}: open position on a symbol no longer watched - managed EXIT-ONLY until closed",
                              level=logging.WARNING)
    providers = make_providers(p, symbols)
    executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="io")
    import ctrader
    _eq = {"rt": None}
    broker = ctrader.from_env(lambda: _eq["rt"].acct.equity if _eq["rt"] else 0.0) or PaperBroker()
    rt = Runtime(p, symbols, providers, broker, store, notifier, executor, mode=mode, gate=True,
                 classes=wl.class_of(items))

    for note in store.notes:
        notifier.send(f"STATE WARNING: {note}", level=logging.ERROR)
    if st is not None:
        try:
            rt.restore(st)
        except Exception as e:
            notifier.send(f"STATE WARNING: could not apply saved state ({type(e).__name__}: {e}) - "
                          f"starting fresh; CHECK for open positions manually", level=logging.ERROR)
            rt = Runtime(p, symbols, providers, broker, store, notifier, executor, mode=mode, gate=True,
                 classes=wl.class_of(items))
    _eq["rt"] = rt
    if hasattr(broker, "live"):                               # cTrader demo: compare with the broker at startup
        try:
            at_broker = broker.sync_positions()
            if at_broker is not None:
                mine = {ss.cfg.symbol for ss in rt.syms if ss.eng.pos and ss.cfg.ticker in broker.live_tickers}
                for lab in set(at_broker) - mine:
                    notifier.send(f"cTrader: position {lab} exists at the broker but not in the bot - check it in "
                                  f"the cTrader app (it keeps its server-side stop/target)", level=logging.WARNING)
                for lab in mine - set(at_broker):
                    notifier.send(f"cTrader: {lab} is open in the bot but already closed at the broker "
                                  f"(server stop/target hit while the bot was down) - will be booked on next price",
                                  level=logging.WARNING)
            else:
                notifier.send(f"cTrader: not reachable at startup ({broker.live.c.last_error}) - retrying every 30 s",
                              level=logging.WARNING)
        except Exception as e:
            notifier.send(f"cTrader reconcile error: {type(e).__name__}: {e}", level=logging.WARNING)
    off = wl.disabled_ids(items)
    for ss in rt.syms:
        ss.disabled = ss.cfg.symbol in off
    if any(s.kind == "stock" for s in symbols) and not cal.covers(datetime.now(timezone.utc).date()):
        notifier.send("WARNING: NYSE calendar in market_calendar.py does not cover today - "
                      "holidays/early closes are NOT known. Update it.", level=logging.WARNING)

    from telegram_ctl import TelegramControl
    def _watch(action: str, data: dict) -> str:
        ok, msg, restart = rt.watchlist_action(action, data)
        if ok and restart:
            notifier.send(f"watchlist {action} {data.get('symbol', '')} (restarting)", level=logging.WARNING)
            threading.Timer(2.0, rt.stopping.set).start()
        return msg

    ctl = TelegramControl(lambda: rt.status(time.time()),
                          {"stop": STOP_FILE, "flatten": FLATTEN_FILE, "halt": "HALT"})
    ctl.start()
    if os.getenv("PORT"):                                     # web dashboard (Render / any host with a port)
        from dashboard import Dashboard
        pw, tok = os.getenv("DASHBOARD_PASSWORD", ""), os.getenv("MONITOR_TOKEN", "")
        if not pw:
            notifier.send("DASHBOARD_PASSWORD is not set - the dashboard pages are locked (only /healthz works)",
                          level=logging.ERROR)
        Dashboard(lambda: rt.status(time.time()), ctl.handle, lambda: rt.last_alive, pw, tok,
                  tail=LOG_TAIL, port=int(os.getenv("PORT")), restart_fn=lambda: rt.stopping.set(),
                  watch_fn=_watch).start()

    if os.getenv("NEWS_WATCH", "1") != "0":                  # Truth Social window - alerts only, never trades
        try:
            from news import NewsWatcher
            rt.news = NewsWatcher(lambda m: notifier.send(m), float(os.getenv("NEWS_POLL_SEC", "300")))
            rt.news.start(rt.stopping)
        except Exception as e:                                # news must never stop the bot
            log.warning("news watcher not started: %s", e)

    from remote import RemoteLink
    RemoteLink(os.getenv("NTFY_TOPIC", ""), os.getenv("MONITOR_TOKEN", ""), lambda: rt.status(time.time()),
               ctl.handle, lambda: rt.stopping.set()).start()

    def _bye(*_):
        rt.stopping.set()
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is not None:
            signal.signal(sig, _bye)

    notifier.send(f"Bot started ({getattr(broker, 'name', 'paper')}). watchlist streams={[s.symbol for s in symbols]} equity={rt.acct.equity:.2f} "
                  f"providers={{{', '.join(f'{s.symbol}: bars={s.provider} quotes={s.quote_src}' for s in symbols)}}}")
    try:
        rt.run()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
