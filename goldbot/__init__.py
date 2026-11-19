"""goldbot: a strategy-agnostic trading framework for MT5 with funded-account risk rules
and an optional multi-agent LLM committee that can veto setups.

The same Engine runs live (MT5 broker) and in backtests (simulated broker on M1 bars),
so a backtest exercises exactly the code that trades.
"""
__version__ = "0.1.0"
