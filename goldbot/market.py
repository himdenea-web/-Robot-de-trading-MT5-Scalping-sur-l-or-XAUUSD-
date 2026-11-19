"""Market data interface used by strategies, plus session-time helpers.

A strategy never touches MT5 or a DataFrame directly; it asks a Market for what it needs.
Live and simulated markets implement the same methods, so strategies behave identically.
"""
from datetime import date, time
from typing import Protocol

import pandas as pd


class Market(Protocol):
    def now(self) -> pd.Timestamp:
        """Current time (UTC, naive). In simulation: the open time of the bar being processed."""

    def hilo(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float, int]:
        """High, low and bar count of closed M1 bars with start <= time < end."""

    def daily(self, day: date) -> dict:
        """Values known at the start of `day`, from prior completed days only:
        {"trend": +1/-1/0 (close vs EMA20), "atr": ATR14 in USD, "prev_close": float}."""

    def spread(self) -> float:
        """Current (or modelled) spread in USD."""

    def last_price(self) -> float:
        """Last bid close."""


def local_to_utc(day: date, t: time, tz: str) -> pd.Timestamp:
    """Wall-clock time in `tz` on `day` -> naive UTC timestamp (handles daylight saving)."""
    return (pd.Timestamp.combine(day, t).tz_localize(tz).tz_convert("UTC").tz_localize(None))


def utc_to_local_date(ts: pd.Timestamp, tz: str) -> date:
    return pd.Timestamp(ts).tz_localize("UTC").tz_convert(tz).date()


def trading_day(ts: pd.Timestamp, tz: str = "America/New_York", reset_hour: int = 17) -> date:
    """Prop-firm trading day: the day rolls over at `reset_hour` local time (17:00 New York
    by default), so Sunday 18:00 NY belongs to Monday's trading day."""
    loc = pd.Timestamp(ts).tz_localize("UTC").tz_convert(tz)
    if loc.hour >= reset_hour:
        loc = loc + pd.Timedelta(days=1)
    return loc.date()


def trading_week(day: date) -> date:
    """Monday of the trading week containing `day`."""
    d = pd.Timestamp(day)
    return (d - pd.Timedelta(days=d.weekday())).date()
