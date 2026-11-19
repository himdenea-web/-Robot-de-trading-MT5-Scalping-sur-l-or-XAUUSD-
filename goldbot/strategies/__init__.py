"""Strategy registry. Add a plug-in by subclassing Strategy and listing it here."""
from goldbot.strategies.base import Strategy  # noqa: F401
from goldbot.strategies.orb import OpeningRangeBreakout

STRATEGIES = {
    OpeningRangeBreakout.name: OpeningRangeBreakout,
}


def make_strategy(name: str, **params) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](**params)
