"""Trade engine shared by the backtest and the paper-live bot,
so both behave identically (sizing, stops, costs, daily limits).

Two ways to use it:
  * backtest / tests : `try_enter()` and `on_bar_close()` (decide + record in one call)
  * live bot         : `plan_entry()` -> broker fills -> `open_position()`;
                       `check_tick()` / `bar_exit_signal()` -> broker fills -> `close_position()`
                       (the engine never assumes an order was filled before the broker says so)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple

import pandas as pd

from config import Params, SymbolCfg


@dataclass
class Position:
    side: int            # +1 long, -1 short
    qty: float
    entry: float
    stop: float
    target: float
    entry_ts: pd.Timestamp   # start of the bar in which the position was opened
    risk_amt: float      # money at risk if the stop is hit (before costs)
    opened_epoch: float = 0.0    # wall-clock time of the fill (live bot)
    monitor_gap: bool = False    # True if live prices were unavailable for a while (enables wick backstop)
    client_id: str = ""
    stop0: float = 0.0           # initial stop (v11: the live stop may trail behind confirmed swings)
    planned: float = 0.0         # price the strategy planned to enter at (signal close) -> entry slippage
    trail_moves: int = 0         # how many times the stop was moved behind a new swing
    order_px: float = 0.0        # live price when the order was sent -> broker fill slippage

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_ts"] = pd.Timestamp(self.entry_ts).isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in names}          # tolerate older/newer state files
        d["entry_ts"] = pd.Timestamp(d["entry_ts"])
        d["side"] = int(d["side"])
        for k in ("qty", "entry", "stop", "target", "risk_amt"):
            d[k] = float(d[k])
        pos = cls(**d)
        if not pos.stop0:
            pos.stop0 = pos.stop
        return pos


@dataclass
class EntryPlan:
    side: int
    qty: float
    stop_dist: float
    target_dist: float
    risk_amt: float
    ts: pd.Timestamp
    ref_price: float


class Account:
    """Equity + daily risk limits. In live mode ONE account is shared by all symbols."""

    def __init__(self, equity: float, p: Params):
        self.equity = float(equity)
        self.p = p
        self.day = None
        self.day_start_equity = self.equity
        self.trades_today = 0
        self.consec_losses = 0

    def roll_day(self, day) -> None:
        if day != self.day:
            self.day = day
            self.day_start_equity = self.equity
            self.trades_today = 0
            self.consec_losses = 0

    def can_trade(self) -> bool:
        p = self.p
        if self.trades_today >= p.max_trades_per_day:
            return False
        if self.consec_losses >= p.max_consec_losses:
            return False
        if self.equity <= self.day_start_equity * (1.0 - p.daily_loss_cap):
            return False
        return True


NO_TARGET_MULT = 20.0


def _ok(x) -> bool:
    return x is not None and math.isfinite(x) and x > 0


class SymbolEngine:
    def __init__(self, scfg: SymbolCfg, p: Params, account: Account):
        self.scfg = scfg
        self.p = p
        self.acct = account
        self.pos: Optional[Position] = None
        self.slip_bps = 0.0          # BACKTEST ONLY: adverse slippage on market fills (entries, stops, eod)

    # ---- helpers -------------------------------------------------------
    def in_window(self, ts: pd.Timestamp) -> bool:
        w = self.scfg.windows
        if not w:
            return True
        hhmm = ts.strftime("%H:%M")
        return any(a <= hhmm <= b for a, b in w)

    # ---- entry ---------------------------------------------------------
    def plan_entry(self, ts: pd.Timestamp, price: float, side: int, atr: float, day) -> Optional[EntryPlan]:
        """All entry rules + position sizing. Does not open anything."""
        self.acct.roll_day(day)
        if self.pos is not None or side == 0 or not _ok(atr) or not _ok(price):
            return None
        if side == -1 and not self.scfg.allow_short:
            return None
        if not self.in_window(ts) or not self.acct.can_trade():
            return None

        p = self.p
        stop_atr = self.scfg.stop_atr or p.stop_atr
        target_atr = self.scfg.target_atr or p.target_atr
        stop_dist = stop_atr * atr
        conv = self.scfg.usd_per_quote(price)          # quote currency -> USD (USD/JPY: 1/price)
        qty = self.acct.equity * p.risk_pct / (stop_dist * conv)
        qty = min(qty, self.acct.equity * p.max_leverage / (price * conv))
        if self.scfg.kind == "stock":
            qty = math.floor(qty)
        elif self.scfg.kind == "fx":
            qty = math.floor(qty * 100) / 100          # 0.01 oz / 0.01 currency unit
        else:
            qty = math.floor(qty * 1e6) / 1e6
        if not _ok(qty):
            return None
        tgt = target_atr * atr
        if self.scfg.trail == "swing" and self.scfg.trail_no_target:
            tgt *= NO_TARGET_MULT        # "no target": a far safety target, the trailing stop takes the profit
        return EntryPlan(side, qty, stop_dist, tgt, qty * stop_dist * conv, ts, price)

    def open_position(self, plan: EntryPlan, fill_price: float, opened_epoch: float = 0.0,
                      client_id: str = "") -> Position:
        """Record a position AFTER the broker confirmed the fill. Stop/target are anchored to the real
        fill price, so the risk:reward stays exactly as designed even if the fill moved."""
        if not _ok(fill_price):
            raise ValueError(f"invalid fill price {fill_price}")
        self.pos = Position(
            side=plan.side, qty=plan.qty, entry=fill_price,
            stop=fill_price - plan.side * plan.stop_dist,
            target=fill_price + plan.side * plan.target_dist,
            entry_ts=plan.ts, risk_amt=plan.risk_amt,
            opened_epoch=opened_epoch, client_id=client_id,
            stop0=fill_price - plan.side * plan.stop_dist, planned=plan.ref_price,
        )
        self.acct.trades_today += 1
        return self.pos

    def try_enter(self, ts: pd.Timestamp, price: float, side: int, atr: float, day,
                  planned: float = 0.0) -> Optional[Position]:
        """Backtest path: called at the START of a bar (price = that bar's open) with the previous bar's signal.
        A market entry is filled `slip_bps` WORSE than the open (realistic, see RESEARCH.md section 15)."""
        plan = self.plan_entry(ts, price, side, atr, day)
        if not plan:
            return None
        pos = self.open_position(plan, price * (1.0 + side * self.slip_bps / 1e4))
        pos.planned = planned or price
        return pos

    # ---- trailing stop behind confirmed swings (v11) --------------------
    def trail_update(self, swing_lo: float, swing_hi: float, atr: float, close: float) -> bool:
        """After a bar CLOSES: move the stop behind the last confirmed swing low (long) / high (short),
        only in the profitable direction and only once the trade is `trail_after_r` R in profit.
        Returns True if the stop moved. Never widens the stop."""
        pos, cfg = self.pos, self.scfg
        if pos is None or cfg.trail != "swing" or not _ok(atr) or not _ok(close):
            return False
        risk = abs(pos.entry - (pos.stop0 or pos.stop))
        if risk <= 0 or pos.side * (close - pos.entry) < cfg.trail_after_r * risk:
            return False
        buf = cfg.trail_buf_atr * atr
        if pos.side == 1:
            if not (swing_lo is not None and math.isfinite(swing_lo)):
                return False
            new = swing_lo - buf
            if new > pos.stop and new < close:
                pos.stop, pos.trail_moves = new, pos.trail_moves + 1
                return True
        else:
            if not (swing_hi is not None and math.isfinite(swing_hi)):
                return False
            new = swing_hi + buf
            if new < pos.stop and new > close:
                pos.stop, pos.trail_moves = new, pos.trail_moves + 1
                return True
        return False

    # ---- exit signals (pure: they do not change state) ------------------
    def check_tick(self, bid: float, ask: float, last: Optional[float] = None) -> Optional[Tuple[str, float]]:
        """Live-price exit check. Longs exit on the BID, shorts on the ASK.
        stop  -> filled at the market price seen (a gap through the stop costs more than 1R - realistic)
        target-> filled at the target (limit order semantics)"""
        pos = self.pos
        if pos is None:
            return None
        px = bid if pos.side == 1 else ask
        if not _ok(px):
            px = last
        if not _ok(px):
            return None
        if pos.side == 1:
            if px <= pos.stop:
                return "stop", px
            if px >= pos.target:
                return "target", pos.target
        else:
            if px >= pos.stop:
                return "stop", px
            if px <= pos.target:
                return "target", pos.target
        return None

    def bar_exit_signal(self, h: float, l: float, c: float, last_of_day: bool,
                        wicks: bool = True, o: Optional[float] = None) -> Optional[Tuple[str, float]]:
        """Bar-based exit check (backtest, and the live backstop). If stop and target are both touched
        inside one bar the STOP is assumed to hit first (conservative)."""
        pos = self.pos
        if pos is None:
            return None
        if wicks and math.isfinite(h) and math.isfinite(l):
            if pos.side == 1:
                hit_stop, hit_tgt = l <= pos.stop, h >= pos.target
            else:
                hit_stop, hit_tgt = h >= pos.stop, l <= pos.target
            if hit_stop:
                # a bar that OPENS beyond the stop (gap) is filled at the open, not at the stop (realistic)
                if o is not None and math.isfinite(o) and pos.side * (o - pos.stop) < 0:
                    return "stop", o
                return "stop", pos.stop
            if hit_tgt:
                return "target", pos.target
        if last_of_day and self.scfg.flatten_eod and _ok(c):
            return "eod", c
        return None

    # ---- exit (records the trade) ---------------------------------------
    def on_bar_close(self, ts, o, h, l, c, day, last_of_day: bool, wicks: bool = True) -> Optional[dict]:
        """Backtest path: check exits using the bar that just closed and record the trade."""
        self.acct.roll_day(day)
        gap_open = o if (self.pos is not None and ts > self.pos.entry_ts) else None
        sig = self.bar_exit_signal(h, l, c, last_of_day, wicks, gap_open)
        if not sig:
            return None
        px = sig[1]
        if sig[0] != "target" and self.slip_bps and self.pos is not None:
            px = px * (1.0 - self.pos.side * self.slip_bps / 1e4)      # market exit: filled a bit worse
        return self._close(ts, px, sig[0])

    def force_close(self, ts, price, reason="forced") -> Optional[dict]:
        return self._close(ts, price, reason) if self.pos else None

    def close_position(self, ts, exit_price: float, reason: str) -> Optional[dict]:
        """Record a close AFTER the broker confirmed the exit fill."""
        return self._close(ts, exit_price, reason) if self.pos else None

    def _close(self, ts, exit_price: float, reason: str) -> dict:
        pos = self.pos
        if not _ok(exit_price):
            raise ValueError(f"invalid exit price {exit_price}")
        conv = self.scfg.usd_per_quote(exit_price)    # P&L is in the quote currency -> USD
        gross = pos.side * (exit_price - pos.entry) * pos.qty * conv
        cost = (pos.entry + exit_price) * pos.qty * self.scfg.cost_bps / 1e4 * conv
        pnl = gross - cost
        # slippage in bps, positive = worse than planned (entry: vs the signal close; exit: vs the stop/target)
        want_x = pos.stop if reason.startswith(("stop", "trail")) else pos.target if reason.startswith("target") else exit_price
        slip_in = pos.side * (pos.entry - pos.planned) / pos.planned * 1e4 if pos.planned else 0.0
        slip_out = pos.side * (want_x - exit_price) / want_x * 1e4 if want_x else 0.0
        slip_fill = pos.side * (pos.entry - pos.order_px) / pos.order_px * 1e4 if pos.order_px else 0.0
        if reason.startswith("stop") and pos.trail_moves:
            reason = "trail" + reason[4:]                 # the stop that hit had been moved behind a swing
        self.acct.equity += pnl
        self.acct.consec_losses = self.acct.consec_losses + 1 if pnl < 0 else 0
        trade = {
            "symbol": self.scfg.symbol,
            "side": "long" if pos.side == 1 else "short",
            "entry_ts": pos.entry_ts, "exit_ts": ts,
            "entry": pos.entry, "exit": exit_price, "qty": pos.qty,
            "pnl_gross": gross, "cost": cost, "pnl": pnl,
            "R": pnl / pos.risk_amt if pos.risk_amt else 0.0,
            "reason": reason, "equity_after": self.acct.equity,
            "slip_in_bps": round(slip_in, 2), "slip_out_bps": round(slip_out, 2), "slip_fill_bps": round(slip_fill, 2),
            "trail_moves": pos.trail_moves, "stop0": pos.stop0,
        }
        self.pos = None
        return trade
