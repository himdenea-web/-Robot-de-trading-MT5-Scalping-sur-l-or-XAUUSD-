"""MT5 broker against an in-memory fake terminal: order flow, restart recovery, reconnect."""
import pandas as pd
import pytest

import goldbot.brokers.mt5 as M
from goldbot.config import RiskConfig
from goldbot.models import Setup
from goldbot.state import StateStore
from tests import fake_mt5 as F


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    F.reset()
    monkeypatch.setattr(M, "mt5", F)
    yield F


def make_setup(direction=1):
    now = pd.Timestamp.now("UTC").tz_localize(None).floor("min")
    entry = 2005.0 if direction > 0 else 1995.0
    return Setup("orb", f"orb-ny_comex-{now:%Y-%m-%d}", direction, entry,
                 entry - 5.0 * direction, entry + 15.0 * direction, now,
                 now + pd.Timedelta(hours=2), now + pd.Timedelta(hours=3), {"session": "ny_comex"})


def connect(**kw):
    return M.MT5Connection("XAUUSD", require_demo=True, log=lambda m: None, **kw)


def test_login_parameters_are_passed_to_the_terminal():
    connect(path=r"C:\MT5\terminal64.exe", login=123, password="pw", server="Srv")
    assert F.S.init_kwargs["login"] == 123 and F.S.init_kwargs["path"].endswith("terminal64.exe")


def test_restart_adopts_orders_and_positions_with_their_context(tmp_path):
    state = StateStore(str(tmp_path / "s.json"))
    c = connect()
    b = M.MT5Broker(c, 777, state=state, log=lambda m: None)
    s = make_setup()
    o = b.place(s, 0.1, 50.0)
    assert o is not None and len(F.S.orders) == 1
    # restart: a new broker reads the state file and keeps the order
    b2 = M.MT5Broker(connect(), 777, state=StateStore(str(tmp_path / "s.json")), log=lambda m: None)
    assert [x.setup.group for x in b2.pending()] == [s.group]
    assert b2.pending()[0].setup.exit_by == s.exit_by
    # the order fills, then the bot restarts again: the position keeps its time exit and risk
    F.trigger_order(int(o.id))
    b3 = M.MT5Broker(connect(), 777, state=StateStore(str(tmp_path / "s.json")), log=lambda m: None)
    pos = b3.positions()
    assert len(pos) == 1 and pos[0].setup.exit_by == s.exit_by and pos[0].planned_risk == 50.0
    # stopped out while the bot was down: the next start rebuilds the closed trade from history
    F.hit_stop(int(pos[0].id))
    b4 = M.MT5Broker(connect(), 777, state=StateStore(str(tmp_path / "s.json")), log=lambda m: None)
    fills = b4.fills_since(pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=1))
    assert len(fills) == 1 and fills[0].setup.group == s.group and fills[0].reason == "sl"
    assert fills[0].pnl < 0 and b4.positions() == []


def test_unknown_leftover_order_is_cancelled(tmp_path):
    c = connect()
    M.MT5Broker(c, 777, state=StateStore(str(tmp_path / "a.json")), log=lambda m: None).place(make_setup(), 0.1, 50)
    assert len(F.S.orders) == 1
    M.MT5Broker(connect(), 777, state=StateStore(str(tmp_path / "other.json")), log=lambda m: None)
    assert F.S.orders == {}


def test_margin_check_refuses_oversized_orders(tmp_path):
    b = M.MT5Broker(connect(), 777, state=StateStore(None), log=lambda m: None)
    assert b.place(make_setup(), 5.0, 50.0) is None             # 5 lots need 40,100 margin
    assert "margin" in b.last_error


def test_health_and_reconnect():
    c = connect()
    assert c.healthy()[0]
    F.S.connected = False
    ok, why = c.healthy()
    assert not ok and "trade server" in why
    F.S.connected = True
    F.shutdown()
    assert not c.healthy()[0]
    c.reconnect()
    assert c.healthy()[0]


def test_trading_allowed_reports_algo_button():
    c = connect()
    F.S.trade_allowed = False
    ok, why = c.trading_allowed()
    assert not ok and "Algo Trading" in why


def test_symbol_specs_include_effective_leverage():
    c = connect()
    r = RiskConfig()
    margin, price = c.margin_per_lot()
    M.apply_symbol_specs(r, c.info, margin, price)
    assert r.leverage == pytest.approx(25.0)


def test_live_runner_check_mode(tmp_path):
    from goldbot.config import Config
    from goldbot.live import run
    cfg = Config(state_file=str(tmp_path / "state.json"), log_file=str(tmp_path / "logs" / "g.log"),
                 journal_path=str(tmp_path / "j.csv"), lock_port=47999)
    assert run(cfg, check=True) == 0
    log = (tmp_path / "logs" / "g.log").read_text(encoding="utf-8")
    assert "100 oz per lot" in log and "leverage 1:25" in log and "goldbot status" in log


def test_second_copy_is_refused():
    from goldbot.live import FatalError, SingleInstance
    first = SingleInstance(47998)  # noqa: F841
    with pytest.raises(FatalError) as e:
        SingleInstance(47998)
    assert e.value.code == 3
