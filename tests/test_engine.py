"""Engine tests on synthetic bars, plus a parity test against the research engine when the
HistData cache is available locally."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

from goldbot.backtest import run_backtest
from goldbot.config import Config, JudgeConfig, RiskConfig
from goldbot.judge import Committee
from goldbot.judge.providers import make_provider

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "backtest", "data", "xauusd_m1.parquet")


def synthetic_m1(days=40, seed=3):
    """Weekday M1 bars with a drift, so the D1 trend filter has something to agree with."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05", periods=days * 1440, freq="min")
    idx = idx[idx.weekday < 5]
    steps = rng.normal(0.01, 0.35, len(idx))
    close = 2000 + np.cumsum(steps)
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) + rng.uniform(0, 0.3, len(idx))
    lo = np.minimum(open_, close) - rng.uniform(0, 0.3, len(idx))
    return pd.DataFrame({"time": idx, "open": open_, "high": hi, "low": lo, "close": close})


def base_cfg(**risk):
    return Config(strategy_params=dict(sessions=("ny_comex",), range_min=60, window_min=120,
                                       stop="mid", target_r=3.0, trend_filter=True),
                  risk=RiskConfig(**{"risk_per_trade_usd": 90, "volume_max": 5.0, **risk}))


def test_one_trade_per_session_and_limits_hold():
    m1 = synthetic_m1()
    res = run_backtest(base_cfg(), m1, spread=0.3, slip=0.1)
    tr = res["trades"]
    assert len(tr) > 5
    assert tr.groupby("group").size().max() == 1                 # OCO: one fill per session
    assert (tr.groupby("day")["pnl"].sum() > -200 - 1e-6).all()  # daily loss limit never broken
    assert (tr["planned_risk"] <= 90 + 1e-9).all()


def test_weekly_target_stops_new_trades():
    m1 = synthetic_m1(days=60)
    res = run_backtest(base_cfg(weekly_profit_target=50.0), m1)
    tr = res["trades"]
    for _, w in tr.groupby("week"):
        cum = w.sort_values("entry_time")["pnl"].cumsum().to_numpy()
        hit = np.flatnonzero(cum >= 50.0)
        if len(hit):
            assert hit[0] == len(cum) - 1      # nothing opened after the target was reached


def test_committee_veto_only_reduces_trades():
    m1 = synthetic_m1()
    cfg = base_cfg()
    cfg.judge = JudgeConfig(enabled=True, provider="fake", use_chair=False, min_approvals=3)
    base = run_backtest(base_cfg(), m1)["trades"]
    with_c = run_backtest(cfg, m1, committee=Committee(cfg.judge, provider=make_provider("fake")))["trades"]
    assert set(with_c["group"]) <= set(base["group"])


@pytest.mark.skipif(not os.path.isfile(CACHE), reason="HistData cache not available")
def test_parity_with_research_engine_2024():
    sys.path.insert(0, os.path.join(ROOT, "backtest"))
    import orb as O
    m1 = pd.read_parquet(CACHE)
    cfg = base_cfg(daily_loss_limit=1e9, first_week_loss_limit=1e9, weekly_profit_target=1e9)
    tr = run_backtest(cfg, m1, start="2024-01-01", end="2025-01-01")["trades"]
    ref = O.run_orb(O.M1(m1), O.OrbParams(session="ny_comex", range_min=60, stop="mid",
                                          target_r=3.0, trend_filter=True))
    ref = ref[(pd.to_datetime(ref.entry_time) >= "2024-01-01") & (pd.to_datetime(ref.entry_time) < "2025-01-01")]
    a = tr.assign(d=tr.entry_time.dt.date).set_index("d")
    b = ref.assign(d=pd.to_datetime(ref.entry_time).dt.date).set_index("d")
    assert set(a.index) == set(b.index)
    j = a.join(b, rsuffix="_ref")
    assert (np.where(j.side == "BUY", 1, -1) == j.dir).all()
    assert np.allclose(j.entry, j.entry_ref)
    assert j.R.corr(j.R_ref) > 0.97
