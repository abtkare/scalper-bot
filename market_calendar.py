"""NYSE trading calendar (regular sessions only).

Source: NYSE Group "2025, 2026 and 2027 Holiday and Early Closings Calendar" (ICE press release).
Outside 2026-2027 the calendar falls back to "weekdays only" and `covers()` returns False so the
bot can warn you to extend the tables below.

Why this matters: without it the bot would (a) keep polling on holidays and (b) NEVER flatten a
stock position on an early-close day (13:00 ET), because its fixed 15:55 flatten time never
arrives -> the position would be carried overnight.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Optional, Tuple

HOLIDAYS = {
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31),
    date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24),
}
EARLY_CLOSE = {date(2026, 11, 27), date(2026, 12, 24), date(2027, 11, 26)}   # 13:00 ET

COVERED_FROM, COVERED_TO = date(2026, 1, 1), date(2027, 12, 31)
OPEN, CLOSE, EARLY = time(9, 30), time(16, 0), time(13, 0)


def covers(day: date) -> bool:
    return COVERED_FROM <= day <= COVERED_TO


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in HOLIDAYS


def session(day: date) -> Optional[Tuple[datetime, datetime]]:
    """(open, close) as naive New-York datetimes, or None if the market is closed that day."""
    if not is_trading_day(day):
        return None
    close = EARLY if day in EARLY_CLOSE else CLOSE
    return datetime.combine(day, OPEN), datetime.combine(day, close)


def active_window(now_ny: datetime, pre_min: int = 5, post_min: int = 3) -> bool:
    """True while it makes sense to poll (a few minutes around the session)."""
    s = session(now_ny.date())
    if s is None:
        return False
    return s[0] - timedelta(minutes=pre_min) <= now_ny <= s[1] + timedelta(minutes=post_min)


def flatten_time(day: date, minutes_before_close: int) -> Optional[datetime]:
    s = session(day)
    return None if s is None else s[1] - timedelta(minutes=minutes_before_close)
