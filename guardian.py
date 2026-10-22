"""Guardian: automatic emergency intervention (runs inside the bot, reacts in seconds).

The engine's daily limits stop a bad DAY. The guardian stops the things that daily limits miss:

  1. Account drawdown from the equity PEAK      -> HALT all new entries (needs a human: delete HALT file)
  2. Strategy decay: last N trades of a symbol average below a threshold -> pause that symbol for hours
  3. Volatility shock: the live price jumps more than k x ATR within a short window
       - no position: block entries on that symbol for a cool-down
       - position open and the shock is AGAINST it: exit immediately (do not wait for the stop)
  4. Per-symbol daily loss cap (in R)            -> that symbol is done for the day
  5. Loss streak across the whole account in a rolling window -> global cool-down

Everything is persisted in state.json (via snapshot()/restore()), so a restart never "forgets" a halt.
The guardian only ever REDUCES risk: it can block entries or close positions, never open one.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

log = logging.getLogger("guardian")
HALT_FILE = "HALT"


@dataclass
class GuardCfg:
    max_drawdown_pct: float = 6.0        # from equity peak -> HALT (manual reset)
    decay_window: int = 12               # last N trades of a symbol ...
    decay_min_avg_R: float = -0.35       # ... averaging below this -> pause the symbol
    decay_pause_h: float = 12.0
    shock_atr: float = 3.0               # live move > k*ATR within shock_window_sec = shock
    shock_window_sec: float = 120.0
    shock_cooldown_min: float = 30.0
    sym_daily_loss_R: float = 3.0        # per symbol per UTC day
    streak_window_h: float = 6.0         # account-wide: N losses within H hours -> cool-down
    streak_losses: int = 5
    streak_cooldown_h: float = 3.0

    @classmethod
    def from_env(cls) -> "GuardCfg":
        g = cls()
        for k in g.__dataclass_fields__:
            raw = os.getenv("GUARD_" + k.upper())
            if raw not in (None, ""):
                try:
                    setattr(g, k, type(getattr(g, k))(float(raw)))
                except ValueError:
                    raise ValueError(f"GUARD_{k.upper()}={raw!r} is not a number") from None
        g.validate()
        return g

    def validate(self) -> None:
        bad = [n for n, ok in (
            ("MAX_DRAWDOWN_PCT", 0 < self.max_drawdown_pct <= 50), ("DECAY_WINDOW", self.decay_window >= 1),
            ("DECAY_PAUSE_H", self.decay_pause_h >= 0), ("SHOCK_ATR", self.shock_atr > 0),
            ("SHOCK_WINDOW_SEC", self.shock_window_sec > 0), ("SHOCK_COOLDOWN_MIN", self.shock_cooldown_min >= 0),
            ("SYM_DAILY_LOSS_R", self.sym_daily_loss_R > 0), ("STREAK_LOSSES", self.streak_losses >= 1),
            ("STREAK_WINDOW_H", self.streak_window_h > 0), ("STREAK_COOLDOWN_H", self.streak_cooldown_h >= 0)) if not ok]
        if bad:
            raise ValueError("invalid guardian setting(s): " + ", ".join("GUARD_" + b for b in bad))


class Guardian:
    def __init__(self, cfg: GuardCfg, notify, clock=time.time, halt_file: str = HALT_FILE):
        self.cfg, self.notify, self.clock, self.halt_file = cfg, notify, clock, halt_file
        self.peak: Optional[float] = None
        self.halted_reason: str = ""
        self.paused_until: Dict[str, float] = {}
        self.pause_reason: Dict[str, str] = {}
        self.recent_R: Dict[str, Deque[float]] = {}
        self.day_R: Dict[str, float] = {}
        self.day: Optional[str] = None
        self.loss_times: Deque[float] = deque(maxlen=50)
        self.global_pause_until = 0.0
        self.ticks: Dict[str, Deque[Tuple[float, float]]] = {}
        self.events: List[str] = []            # last interventions (for status reports)
        self.changed = False                   # set on every intervention -> the bot saves state at once
        self._halt_file_ok = True

    # ---------------------------------------------------------------- persistence
    def snapshot(self) -> dict:
        return {"peak": self.peak, "halted_reason": self.halted_reason,
                "paused_until": self.paused_until, "pause_reason": self.pause_reason,
                "recent_R": {k: list(v) for k, v in self.recent_R.items()},
                "day_R": self.day_R, "day": self.day, "loss_times": list(self.loss_times),
                "global_pause_until": self.global_pause_until, "events": self.events[-20:]}

    def restore(self, d: Optional[dict]) -> None:
        if not d:
            return
        self.peak = d.get("peak")
        self.halted_reason = d.get("halted_reason", "")
        self.paused_until = {k: float(v) for k, v in (d.get("paused_until") or {}).items()}
        self.pause_reason = dict(d.get("pause_reason") or {})
        self.recent_R = {k: deque(v, maxlen=self.cfg.decay_window) for k, v in (d.get("recent_R") or {}).items()}
        self.day_R = dict(d.get("day_R") or {})
        self.day = d.get("day")
        self.loss_times = deque(d.get("loss_times") or [], maxlen=50)
        self.global_pause_until = float(d.get("global_pause_until") or 0.0)
        self.events = list(d.get("events") or [])

    # ---------------------------------------------------------------- helpers
    def _event(self, msg: str, level=logging.ERROR) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.clock()))
        self.events.append(f"{stamp}Z {msg}")
        self.events = self.events[-20:]
        self.changed = True
        self.notify(f"GUARDIAN: {msg}", level)

    def _roll_day(self, now: float) -> None:
        d = time.strftime("%Y-%m-%d", time.gmtime(now))
        if d != self.day:
            self.day, self.day_R = d, {}

    def halted(self) -> bool:
        exists = os.path.exists(self.halt_file)
        if exists and not self.halted_reason:          # HALT written by hand / Telegram /halt
            try:
                with open(self.halt_file, encoding="utf-8") as f:
                    why = f.readline().strip()
            except OSError:
                why = ""
            self.halted_reason = why or "manual HALT file"
            self._halt_file_ok = True
            self.changed = True
            return True
        if self.halted_reason and not exists and self._halt_file_ok:
            # the human deleted the HALT file -> resume, and restart drawdown tracking from here
            self.notify("GUARDIAN: HALT file removed - trading resumes, drawdown peak reset", logging.WARNING)
            self.halted_reason, self.peak = "", None
            return False
        return bool(self.halted_reason)

    def _halt(self, reason: str) -> None:
        self.halted_reason = reason
        try:
            with open(self.halt_file, "w", encoding="utf-8") as f:
                f.write(reason + "\nDelete this file to resume trading.\n")
            self._halt_file_ok = True
        except OSError:
            self._halt_file_ok = False             # stay halted; only a restart with the file removed resets
        self._event(f"HALT - {reason}. No new entries until the HALT file is deleted.", logging.CRITICAL)

    # ---------------------------------------------------------------- public API used by the bot
    def entry_block(self, symbol: str, now: Optional[float] = None) -> Optional[str]:
        """Reason why a NEW entry on `symbol` is blocked, or None."""
        now = self.clock() if now is None else now
        self._roll_day(now)
        if self.halted():
            return f"halted ({self.halted_reason})"
        if now < self.global_pause_until:
            return f"account cool-down ({(self.global_pause_until - now) / 60:.0f} min left)"
        until = self.paused_until.get(symbol, 0.0)
        if now < until:
            return f"{self.pause_reason.get(symbol, 'paused')} ({(until - now) / 60:.0f} min left)"
        if self.day_R.get(symbol, 0.0) <= -self.cfg.sym_daily_loss_R:
            return f"symbol daily loss {self.day_R[symbol]:.1f}R"
        return None

    def on_equity(self, equity: float) -> None:
        if self.peak is None or equity > self.peak:
            self.peak = equity
        dd = (1 - equity / self.peak) * 100 if self.peak else 0.0
        if dd >= self.cfg.max_drawdown_pct and not self.halted_reason:
            self._halt(f"drawdown {dd:.1f}% from peak {self.peak:.2f}")

    def on_trade(self, symbol: str, R: float, equity: float, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        self._roll_day(now)
        c = self.cfg
        self.on_equity(equity)                     # drawdown first: it must never be skipped
        dq = self.recent_R.setdefault(symbol, deque(maxlen=c.decay_window))
        dq.append(float(R))
        self.day_R[symbol] = self.day_R.get(symbol, 0.0) + float(R)
        if len(dq) >= c.decay_window and sum(dq) / len(dq) < c.decay_min_avg_R:
            self._pause(symbol, c.decay_pause_h * 3600, now,
                        f"strategy decay: last {len(dq)} trades avg {sum(dq) / len(dq):.2f}R")
            dq.clear()
        if self.day_R[symbol] <= -c.sym_daily_loss_R:
            self._event(f"{symbol}: daily loss {self.day_R[symbol]:.1f}R - done for today", logging.WARNING)
        if R < 0:
            self.loss_times.append(now)
            recent = [t for t in self.loss_times if now - t <= c.streak_window_h * 3600]
            if len(recent) >= c.streak_losses and now >= self.global_pause_until:
                self.global_pause_until = now + c.streak_cooldown_h * 3600
                self.loss_times.clear()
                self._event(f"{len(recent)} losses within {c.streak_window_h:.0f}h - all entries paused "
                            f"{c.streak_cooldown_h:.0f}h", logging.WARNING)

    def _pause(self, symbol: str, sec: float, now: float, why: str) -> None:
        self.paused_until[symbol] = max(self.paused_until.get(symbol, 0.0), now + sec)
        self.pause_reason[symbol] = why
        self._event(f"{symbol}: paused {sec / 3600:.1f}h - {why}", logging.WARNING)

    def on_price(self, symbol: str, px: float, atr: Optional[float], pos_side: int = 0,
                 now: Optional[float] = None) -> bool:
        """Feed every accepted live price. Returns True if an open position must be closed NOW
        (violent move against it)."""
        now = self.clock() if now is None else now
        dq = self.ticks.setdefault(symbol, deque(maxlen=400))
        dq.append((now, px))
        while dq and now - dq[0][0] > self.cfg.shock_window_sec:
            dq.popleft()
        if not atr or atr <= 0 or len(dq) < 2:
            return False
        hi = max(p for _, p in dq)
        lo = min(p for _, p in dq)
        if hi - lo <= self.cfg.shock_atr * atr:
            return False
        move_dir = 1 if px >= (hi + lo) / 2 else -1        # where are we now inside the spike?
        if self.paused_until.get(symbol, 0.0) <= now:
            self._pause(symbol, self.cfg.shock_cooldown_min * 60, now,
                        f"volatility shock {(hi - lo) / atr:.1f}xATR in {self.cfg.shock_window_sec:.0f}s")
        return pos_side != 0 and move_dir == -pos_side

    def status(self, now: Optional[float] = None) -> dict:
        now = self.clock() if now is None else now
        return {"halted": self.halted_reason or None, "peak": self.peak,
                "global_pause_min": max(0, round((self.global_pause_until - now) / 60)),
                "paused": {k: round((v - now) / 60) for k, v in self.paused_until.items() if v > now},
                "day_R": {k: round(v, 2) for k, v in self.day_R.items()},
                "recent_avg_R": {k: round(sum(v) / len(v), 2) for k, v in self.recent_R.items() if v},
                "last_events": self.events[-5:]}
