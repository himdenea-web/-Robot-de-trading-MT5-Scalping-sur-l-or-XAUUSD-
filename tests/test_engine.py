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
                  risk=RiskConfig(**{"risk_per_trade_usd": 90, "volume_max": 5.0, "max_total_loss": 0,
                                     "profit_target": 0, "commission_rate": 0.0, "leverage": 0.0, **risk}))


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
    cfg = base_cfg(daily_loss_limit=1e9, max_open_risk=0)
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


def test_static_floor_and_profit_target_in_backtest():
    m1 = synthetic_m1(days=80)
    res = run_backtest(base_cfg(max_total_loss=400, profit_target=150), m1)
    tr = res["trades"].sort_values("entry_time")
    eq = tr["pnl"].cumsum().to_numpy()
    assert (eq > -400).all()                                  # floor never broken
    hit = np.flatnonzero(eq >= 150)
    if len(hit):
        assert hit[0] == len(eq) - 1                          # nothing opened after the target


def _ny_minutes(s):
    t = pd.to_datetime(s).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    return t.dt.hour * 60 + t.dt.minute


def test_news_blackout_and_close_winners():
    from goldbot.config import NewsConfig
    m1 = synthetic_m1(days=60)
    cfg = base_cfg()
    cfg.news = NewsConfig(source="fixed:10:00", before_min=6, after_min=5, close_winners=True)
    res = run_backtest(cfg, m1)
    tr = res["trades"]
    news_exits = tr[tr.reason == "news"]
    assert len(news_exits) > 0
    assert (news_exits.pnl > 0).all()
    assert (_ny_minutes(news_exits.exit_time) < 600 - 5).all()   # banked before the window
    for d in res["engine"].decisions:                              # no setup accepted in a blackout
        if d["stage"] == "news":
            m = _ny_minutes(pd.Series([d["time"]])).iloc[0]
            assert 600 - 6 <= m <= 600 + 5


def test_news_calendar_sources(tmp_path):
    from goldbot.config import NewsConfig
    from goldbot.news import NewsCalendar
    cal = NewsCalendar(NewsConfig(source="fixed:08:30,10:00"))
    ev = cal.events_between(pd.Timestamp("2026-07-15 00:00"), pd.Timestamp("2026-07-16 00:00"))
    assert [t for t, _ in ev] == [pd.Timestamp("2026-07-15 12:30"), pd.Timestamp("2026-07-15 14:00")]
    assert cal.blackout(pd.Timestamp("2026-07-15 13:56")) is not None   # 4 min before 10:00 NY
    assert cal.blackout(pd.Timestamp("2026-07-15 14:06")) is None
    rows = [{"title": "CPI m/m", "country": "USD", "impact": "High", "date": "2026-11-12T08:30:00-05:00"},
            {"title": "ECB", "country": "EUR", "impact": "High", "date": "2026-11-12T08:15:00-05:00"},
            {"title": "Retail", "country": "USD", "impact": "Medium", "date": "2026-11-12T10:00:00-05:00"}]
    ff = NewsCalendar(NewsConfig(source="forexfactory"), fetch=lambda: rows, log=lambda m: None)
    ff.refresh(force=True)
    assert ff.events_between("2026-11-12", "2026-11-13") == [(pd.Timestamp("2026-11-12 13:30"), "CPI m/m")]
    f = tmp_path / "news.csv"
    f.write_text("time,title\n2026-11-12 08:30,CPI\n")
    fc = NewsCalendar(NewsConfig(source="file", file=str(f)), log=lambda m: None)
    fc.refresh(force=True)
    assert fc.events_between("2026-11-12", "2026-11-13")[0][0] == pd.Timestamp("2026-11-12 13:30")


def test_env_inline_comments(tmp_path, monkeypatch):
    from goldbot.config import load_env
    f = tmp_path / ".env"
    f.write_text("GB_TEST_MODE=alert      # a comment\nGB_TEST_Q='a # b'\n")
    monkeypatch.delenv("GB_TEST_MODE", raising=False)
    monkeypatch.delenv("GB_TEST_Q", raising=False)
    load_env(str(f))
    assert os.environ["GB_TEST_MODE"] == "alert" and os.environ["GB_TEST_Q"] == "a # b"


def test_news_pause_keeps_entries_out_of_the_window():
    from goldbot.config import NewsConfig
    m1 = synthetic_m1(days=60)
    cfg = base_cfg()
    cfg.news = NewsConfig(source="fixed:10:00", pause_pending=True)
    res = run_backtest(cfg, m1)
    tr = res["trades"]
    m = _ny_minutes(tr.entry_time)
    assert not m.between(600 - 5, 600 + 5).any()
    assert len(tr) > 5


def test_state_store_roundtrip(tmp_path):
    from goldbot.state import StateStore
    from goldbot.models import Setup
    T = pd.Timestamp("2026-11-10 14:20")
    s = Setup("orb", "orb-ny_comex-2026-11-10", 1, 2000.0, 1995.0, 2015.0, T, T + pd.Timedelta(hours=2),
              T + pd.Timedelta(hours=3), {"session": "ny_comex", "x": [1]}, invalidate=1990.0)
    st = StateStore(str(tmp_path / "state.json"))
    st.remember(123, s, 0.1, 50.0)
    back = StateStore(str(tmp_path / "state.json")).get(123)
    assert back[0].group == s.group and back[0].exit_by == s.exit_by and back[0].invalidate == 1990.0
    assert back[1:] == (0.1, 50.0)
    st.prune({999})
    assert StateStore(str(tmp_path / "state.json")).get(123) is None


def test_env_example_parses_with_defaults():
    from goldbot.config import Config
    saved = dict(os.environ)
    try:
        for k in list(os.environ):
            if k.startswith("GB_") or k in ("SYMBOL", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"):
                del os.environ[k]
        cfg = Config.from_env(os.path.join(ROOT, ".env.example"))
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert cfg.mode == "alert" and cfg.risk.account_start is None and cfg.judge.model == ""
    assert cfg.mt5_path == "" and cfg.mt5_login == 0
    assert cfg.risk.risk_per_trade_usd == 50 and cfg.news.pause_pending and cfg.news.source == "forexfactory"
