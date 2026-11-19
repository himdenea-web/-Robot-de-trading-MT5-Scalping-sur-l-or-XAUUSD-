from datetime import date

import pandas as pd
import pytest

from goldbot.config import RiskConfig
from goldbot.market import local_to_utc, trading_day
from goldbot.models import Setup
from goldbot.risk import AccountState, FundedRiskManager

T = pd.Timestamp("2026-03-10 13:20")


def setup(entry=2000.0, sl=1995.0, d=1):
    return Setup("orb", "g1", d, entry, sl, entry + 3 * (entry - sl) * d, T, T, T)


def rm(**kw):
    return FundedRiskManager(RiskConfig(**kw))


def test_size_rounds_down_to_risk():
    lots, risk = rm(risk_per_trade_usd=90).size(setup(sl=1995.0))   # $5 stop -> $500 per lot
    assert lots == 0.18 and risk == pytest.approx(90.0)
    lots, risk = rm(risk_per_trade_usd=90).size(setup(sl=1993.0))   # $7 stop -> $700 per lot
    assert lots == 0.12 and risk <= 90.0


def test_rejects_when_stop_too_wide_for_min_lot():
    v = rm(risk_per_trade_usd=90).check(setup(sl=1900.0), AccountState())  # $100 stop
    assert not v.ok and "too wide" in v.reason


def test_daily_limit_counts_open_risk_and_buffer():
    r = rm(risk_per_trade_usd=90, daily_loss_limit=200, loss_buffer_usd=15)
    assert r.check(setup(), AccountState(day_realized=-90)).ok             # -90-90-15 = -195
    v = r.check(setup(), AccountState(day_realized=-90, open_risk=10))     # -205
    assert not v.ok and "daily" in v.reason
    assert not r.check(setup(), AccountState(open_risk=100)).ok            # -205 worst case


def test_profit_today_gives_room():
    r = rm(risk_per_trade_usd=90, daily_loss_limit=200, loss_buffer_usd=15)
    assert r.check(setup(), AccountState(day_realized=+50, open_risk=140)).ok   # 50-140-90-15 = -195
    assert not r.check(setup(), AccountState(day_realized=0, open_risk=140)).ok  # -245


def test_first_week_limit():
    r = rm(risk_per_trade_usd=90, first_week_loss_limit=400, loss_buffer_usd=15)
    st = AccountState(since_start_realized=-300, in_first_week=True)
    v = r.check(setup(), st)
    assert not v.ok and "first-week" in v.reason
    st.in_first_week = False
    assert r.check(setup(), st).ok


def test_weekly_target_stops_trading():
    v = rm(weekly_profit_target=600).check(setup(), AccountState(week_realized=612))
    assert not v.ok and "weekly target" in v.reason


def test_max_trades_per_day():
    v = rm(max_trades_per_day=2).check(setup(), AccountState(trades_today=2))
    assert not v.ok


def test_unprotected_external_position_blocks():
    assert not rm().check(setup(), AccountState(external_unprotected=True)).ok


def test_trading_day_rolls_at_17_new_york():
    # 21:30 UTC in March 2026 (after US DST starts on 8 Mar) = 17:30 NY -> next trading day
    assert trading_day(pd.Timestamp("2026-03-10 21:30")) == date(2026, 3, 11)
    assert trading_day(pd.Timestamp("2026-03-10 20:30")) == date(2026, 3, 10)


def test_session_open_follows_daylight_saving():
    assert local_to_utc(date(2026, 1, 15), pd.Timestamp("08:20").time(), "America/New_York") == pd.Timestamp("2026-01-15 13:20")
    assert local_to_utc(date(2026, 7, 15), pd.Timestamp("08:20").time(), "America/New_York") == pd.Timestamp("2026-07-15 12:20")
