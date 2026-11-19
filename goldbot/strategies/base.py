"""Strategy plug-in interface.

A strategy reads market data through the Market interface and returns Setups. It never
places orders, sizes positions or checks account limits; the engine and risk manager do.
"""
from abc import ABC, abstractmethod

from goldbot.market import Market
from goldbot.models import Setup


class Strategy(ABC):
    name = "base"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def on_bar(self, market: Market) -> list[Setup]:
        """Called once per closed M1 bar. Return new setups (possibly an OCO pair), or []."""

    def describe(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
