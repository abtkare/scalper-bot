"""Execution layer. The engine DECIDES (signals, sizing, stops); a Broker EXECUTES.

`PaperBroker` simulates fills. To trade real money you would write a `LiveBroker(Broker)` for the
broker you choose - it is intentionally NOT included. A live adapter MUST:

  1. Use `client_id` as the broker's client-order-id (idempotency: a retry after a timeout must not
     open a second position).
  2. Place a PROTECTIVE STOP (bracket/OCO) on the broker's side at entry, so the position is
     protected even if this bot, the VPS or the internet dies. The tick loop then only handles
     targets / manual flatten / EOD.
  3. Implement `sync_positions()` and reconcile at startup (broker is the source of truth).
  4. Return `Fill(ok=False, ...)` on ANY rejection/timeout/ambiguity - never raise into the loop.
     The bot keeps the position and retries the exit; it never "forgets" an open position.
  5. Be tested on the broker's paper/testnet account for weeks before a cent of real money.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Fill:
    ok: bool
    price: float = 0.0
    qty: float = 0.0
    order_id: str = ""
    reason: str = ""


class Broker(ABC):
    name = "abstract"

    @abstractmethod
    def market_entry(self, symbol: str, side: int, qty: float, ref_price: float, client_id: str) -> Fill: ...

    @abstractmethod
    def market_exit(self, symbol: str, side: int, qty: float, ref_price: float, client_id: str) -> Fill: ...

    def sync_positions(self) -> Optional[Dict[str, float]]:
        """{symbol: signed qty} as the broker sees it, or None if the broker cannot tell."""
        return None


class PaperBroker(Broker):
    """Fills at the reference price handed in by the bot (the live quote, side-adjusted).
    Trading costs are charged by the engine (`cost_bps`), so no extra slippage is added here."""

    name = "paper"

    def _fill(self, symbol: str, qty: float, ref_price: float, client_id: str) -> Fill:
        if not (math.isfinite(ref_price) and ref_price > 0 and math.isfinite(qty) and qty > 0):
            return Fill(False, reason=f"invalid order qty={qty} price={ref_price}")
        return Fill(True, price=float(ref_price), qty=float(qty), order_id=client_id)

    def market_entry(self, symbol, side, qty, ref_price, client_id):
        return self._fill(symbol, qty, ref_price, client_id)

    def market_exit(self, symbol, side, qty, ref_price, client_id):
        return self._fill(symbol, qty, ref_price, client_id)
