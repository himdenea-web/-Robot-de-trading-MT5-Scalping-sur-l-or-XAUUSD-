"""Backtest the real engine (strategy + risk + optional committee) on M1 bars.

    from goldbot.backtest import run_backtest
    res = run_backtest(cfg, m1, spread=0.30, slip=0.10)
    res["trades"]   # DataFrame of closed trades with USD pnl and R

The committee is off unless a provider is passed explicitly: running an LLM over years of
setups is slow and billed. Use backtest/llm_replay.py to evaluate it on a sample.
"""
import copy

import pandas as pd

from goldbot.brokers.sim import SimBroker, SimMarket
from goldbot.engine import Engine
from goldbot.market import trading_day, trading_week
from goldbot.strategies import make_strategy


def run_backtest(cfg, m1: pd.DataFrame, spread=0.30, slip=0.10, start_equity=None,
                 committee=None, start=None, end=None, progress=False):
    cfg = copy.deepcopy(cfg)
    cfg.mode = "trade"                     # the simulated broker must receive orders
    if committee is None:
        cfg.judge.enabled = False
    # daily statistics (trend, ATR) use the full history so a sliced run is warmed up;
    # trading only happens between start and end
    market = SimMarket(m1, spread=spread)
    i0 = 0 if start is None else int(market.t.searchsorted(pd.Timestamp(start).to_datetime64()))
    i1 = len(market.t) if end is None else int(market.t.searchsorted(pd.Timestamp(end).to_datetime64()))
    if start_equity is None:
        start_equity = cfg.risk.start_balance
    if cfg.news.source in ("forexfactory", "file"):
        cfg.news.source = "off"            # no historical calendar; use fixed:HH:MM to test the rule
    broker = SimBroker(market, contract_size=cfg.risk.contract_size, slip=slip, equity=start_equity)
    strategy = make_strategy(cfg.strategy, **cfg.strategy_params)
    engine = Engine(cfg, strategy, market, broker, committee=committee)
    step = max(1, (i1 - i0) // 20)
    for i in range(i0, i1):
        market.i = i
        engine.step()
        broker.process_bar()
        if progress and (i - i0) % step == 0:
            print(f"  {(i - i0) / (i1 - i0):5.0%}  {pd.Timestamp(market.t[i]):%Y-%m-%d}  fills {len(broker.fills)}")
    # close anything still open at the last bar
    market.i = max(i0, i1 - 1)
    for p in broker.positions():
        broker.close(p, "end")
    return {"trades": fills_frame(broker.fills, cfg), "engine": engine, "broker": broker,
            "decisions": pd.DataFrame(engine.decisions)}


def fills_frame(fills, cfg) -> pd.DataFrame:
    r = cfg.risk
    rows = []
    for f in fills:
        d = trading_day(f.exit_time, r.day_reset_tz, r.day_reset_hour)
        rows.append({"group": f.setup.group, "side": f.setup.side, "entry_time": f.entry_time,
                     "exit_time": f.exit_time, "entry": f.entry_price, "exit": f.exit_price,
                     "sl": f.setup.sl, "tp": f.setup.tp, "lots": f.lots, "planned_risk": f.planned_risk,
                     "pnl": f.pnl, "R": f.r, "reason": f.reason, "day": d, "week": trading_week(d),
                     "session": f.setup.context.get("session", "")})
    return pd.DataFrame(rows)
