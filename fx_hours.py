"""Trading hours for spot gold / FX (times are America/New_York, naive).

  FX    : Sunday 17:00 -> Friday 17:00, continuous
  Metals: Sunday 18:00 -> Friday 17:00, with a daily break 17:00-18:00
  Closed: Dec 25 and Jan 1 (thin, unreliable prices)

Twelve Data keeps publishing tiny "filler" bars while these markets are closed (weekends, gold's daily
break). Those bars are not real trading and must never reach the indicators - `is_open()` filters them.
"""
from __future__ import annotations

from datetime import datetime, timedelta


def is_open(ny: datetime, metal: bool) -> bool:
    d, h = ny.weekday(), ny.hour            # Mon=0 .. Sun=6
    if (ny.month, ny.day) in ((12, 25), (1, 1)):
        return False
    if d == 5:
        return False
    if d == 6:
        return h >= (18 if metal else 17)
    if d == 4 and h >= 17:
        return False
    if metal and h == 17:
        return False
    return True


def bar_is_open(bar_start_ny: datetime, metal: bool) -> bool:
    return is_open(bar_start_ny, metal)


def next_open(ny: datetime, metal: bool) -> datetime:
    t = ny.replace(second=0, microsecond=0)
    for _ in range(60 * 24 * 4):
        if is_open(t, metal):
            return t
        t += timedelta(minutes=1)
    return t
