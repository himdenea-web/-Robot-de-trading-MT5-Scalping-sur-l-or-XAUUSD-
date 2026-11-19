"""Plain data objects shared by strategies, risk, brokers and the engine.

All timestamps are naive pandas Timestamps in UTC.
"""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Setup:
    """A proposed entry. Setups with the same `group` are one-cancels-other (OCO)."""
    strategy: str
    group: str
    direction: int                 # +1 long, -1 short
    entry: float                   # stop-order level (long: ask >= entry, short: bid <= entry)
    sl: float                      # stop level, on the side the position exits on
    tp: float | None               # take-profit level, or None for time exit only
    created: pd.Timestamp
    expires: pd.Timestamp          # pending order is cancelled after this time
    exit_by: pd.Timestamp          # open position is closed at this time
    context: dict = field(default_factory=dict)
    # cancel the pending order if price reaches this level first
    # (long: bid <= invalidate, short: ask >= invalidate); None = never
    invalidate: float | None = None

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.sl)

    @property
    def side(self) -> str:
        return "BUY" if self.direction > 0 else "SELL"


@dataclass
class Order:
    id: str
    setup: Setup
    lots: float
    planned_risk: float


@dataclass
class Position:
    id: str
    setup: Setup
    lots: float
    entry_price: float
    entry_time: pd.Timestamp
    planned_risk: float


@dataclass
class Fill:
    """A closed trade."""
    position_id: str
    setup: Setup
    lots: float
    entry_price: float
    entry_time: pd.Timestamp
    exit_price: float
    exit_time: pd.Timestamp
    reason: str                    # "sl" | "tp" | "time" | "manual" | "end"
    pnl: float
    planned_risk: float

    @property
    def r(self) -> float:
        return self.pnl / self.planned_risk if self.planned_risk else 0.0
