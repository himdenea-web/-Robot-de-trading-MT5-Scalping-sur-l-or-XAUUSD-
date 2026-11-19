"""Broker interface used by the engine (live MT5 or simulated)."""
from typing import Protocol

import pandas as pd

from goldbot.models import Fill, Order, Position, Setup


class Broker(Protocol):
    def account(self) -> dict:
        """{"equity": float, "balance": float, "is_demo": bool}"""

    def place(self, setup: Setup, lots: float, planned_risk: float) -> Order | None:
        """Place a stop order for the setup. Returns the Order, or None if refused."""

    def cancel(self, order: Order) -> bool: ...

    def pending(self) -> list[Order]: ...

    def positions(self) -> list[Position]: ...

    def close(self, position: Position, reason: str) -> Fill | None: ...

    def fills_since(self, start: pd.Timestamp) -> list[Fill]:
        """Bot fills closed at or after `start` (UTC)."""

    def realized_since(self, start: pd.Timestamp) -> list[tuple[pd.Timestamp, float]]:
        """(close time UTC, net pnl) of every closed deal on the account since `start`
        (all deals, or bot deals only, depending on config)."""

    def external_risk(self) -> tuple[float, bool]:
        """(risk to stop of non-bot positions, any non-bot position without a stop)."""
