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
    """Old-style limits only (daily, per trade) unless a test switches the new ones on."""
    base = {"risk_per_trade_usd": 90, "max_total_loss": 0, "profit_target": 0, "max_open_risk": 0,
            "commission_rate": 0.0, "leverage": 0.0}
    return FundedRiskManager(RiskConfig(**{**base, **kw}))


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
    v = r.check(setup(), AccountState(day_realized=-90))                   # room 95 >= 90
    assert v.ok and v.planned_risk == pytest.approx(90)
    v = r.check(setup(), AccountState(day_realized=-90, open_risk=10))     # room 85: size reduced
    assert v.ok and v.planned_risk <= 85 and "daily" in v.reason
    v = r.check(setup(), AccountState(day_realized=-160))                  # room 25 >= min 20
    assert v.ok and v.planned_risk <= 25
    v = r.check(setup(), AccountState(day_realized=-170))                  # room 15 < min 20
    assert not v.ok and "daily" in v.reason


def test_profit_today_gives_room():
    r = rm(risk_per_trade_usd=90, daily_loss_limit=200, loss_buffer_usd=15)
    v = r.check(setup(), AccountState(day_realized=+50, open_risk=140))    # room 95
    assert v.ok and v.planned_risk == pytest.approx(90)
    v = r.check(setup(), AccountState(day_realized=0, open_risk=140))      # room 45
    assert v.ok and v.planned_risk <= 45


def test_static_floor_shrinks_then_blocks():
    r = rm(risk_per_trade_usd=50, start_balance=5000, max_total_loss=400, loss_buffer_usd=15)
    assert r.check(setup(), AccountState(balance=5000)).planned_risk == pytest.approx(50)
    v = r.check(setup(), AccountState(balance=4650))                        # room 35
    assert v.ok and v.planned_risk <= 35 and "floor" in v.reason
    v = r.check(setup(), AccountState(balance=4630))                        # room 15
    assert not v.ok and "floor" in v.reason
    v = r.check(setup(), AccountState(balance=4700, open_risk=50))          # room 35
    assert v.ok and v.planned_risk <= 35


def test_max_open_risk():
    r = rm(risk_per_trade_usd=90, max_open_risk=150)
    v = r.check(setup(), AccountState(open_risk=90))                        # 60 left
    assert v.ok and v.planned_risk <= 60 and "exposure" in v.reason
    v = r.check(setup(), AccountState(open_risk=140))
    assert not v.ok and "exposure" in v.reason


def test_profit_target_stops_trading():
    r = rm(start_balance=5000, profit_target=400)
    assert r.check(setup(), AccountState(balance=5399)).ok
    v = r.check(setup(), AccountState(balance=5400))
    assert not v.ok and "profit target" in v.reason


def test_default_rules_are_the_5k_two_phase_account():
    c = RiskConfig()
    assert (c.start_balance, c.max_total_loss, c.profit_target, c.daily_loss_limit, c.max_open_risk) == \
        (5000, 400, 400, 200, 150)
    assert c.weekly_profit_target == 0 and c.first_week_loss_limit == 0


def test_first_week_limit():
    r = rm(risk_per_trade_usd=90, first_week_loss_limit=400, loss_buffer_usd=15)
    st = AccountState(since_start_realized=-300, in_first_week=True)   # room 85
    v = r.check(setup(), st)
    assert v.ok and v.planned_risk <= 85 and "first-week" in v.reason
    st.since_start_realized = -370                                       # room 15
    assert not r.check(setup(), st).ok
    st.since_start_realized = -300
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


def test_broker_contract_specs_drive_sizing():
    from types import SimpleNamespace
    from goldbot.brokers.mt5 import apply_symbol_specs
    c = RiskConfig(risk_per_trade_usd=50, volume_max=1.0, commission_rate=0.0, leverage=0.0)
    info = SimpleNamespace(trade_contract_size=10.0, volume_min=0.01, volume_step=0.01, volume_max=50.0)
    changed = apply_symbol_specs(c, info)
    assert c.contract_size == 10.0 and "contract_size" in changed and c.volume_max == 1.0
    lots, risk = FundedRiskManager(c).size(setup(sl=1995.0))     # $5 stop x 10 oz = $50 per lot
    assert lots == 1.0 and risk == pytest.approx(50.0)
    c100 = RiskConfig(risk_per_trade_usd=50, commission_rate=0.0, leverage=0.0)
    apply_symbol_specs(c100, SimpleNamespace(trade_contract_size=100.0, volume_min=0.01, volume_step=0.01, volume_max=0.5))
    assert FundedRiskManager(c100).size(setup(sl=1995.0)) == (0.1, pytest.approx(50.0)) and c100.volume_max == 0.5


def test_commission_is_part_of_the_planned_loss():
    # $5 stop at 2000: 500 per lot + commission 2 x 0.0018% x 2000 x 100 = 7.2 per lot
    r = rm(risk_per_trade_usd=50, commission_rate=0.000018)
    assert r.loss_per_lot(setup(sl=1995.0)) == pytest.approx(507.2)
    lots, risk = r.size(setup(sl=1995.0))
    assert lots == 0.09 and risk <= 50


def test_margin_caps_the_lot_size():
    # 1:15 on 2000 x 100 oz = 13,333 margin per lot; half of 5,000 equity allows 0.18 lots
    r = rm(risk_per_trade_usd=150, max_open_risk=150, leverage=15.0, max_margin_use=0.5)
    lots, _ = r.size(setup(sl=1999.0), equity=5000.0)          # $1 stop would want 1.5 lots
    assert lots == 0.18
    v = r.check(setup(sl=1999.0), AccountState(equity=5000.0))
    assert v.ok and v.lots == 0.18
    v = r.check(setup(sl=1999.0), AccountState(equity=50.0))
    assert not v.ok and "margin" in v.reason


def test_min_trading_days_allows_one_min_lot_trade_after_target():
    r = rm(start_balance=5000, profit_target=200, min_trading_days=2, commission_rate=0.0)
    st = AccountState(balance=5250, trading_days=1, trades_today=0)
    v = r.check(setup(), st)
    assert v.ok and v.lots == 0.01 and "trading days" in v.reason
    st.trading_days = 2
    assert not r.check(setup(), st).ok
    st.trading_days, st.trades_today = 1, 1
    assert not r.check(setup(), st).ok
