"""Configuration, read from environment variables (and an optional .env file).

Every value has a safe default: alert-only mode, demo account required, LLM committee off.
"""
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any


def load_env(path=".env"):
    """Minimal .env reader: KEY=VALUE lines, existing environment variables win."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if not v.startswith(("'", '"')):
                v = "" if v.startswith("#") else re.split(r"\s#", v, maxsplit=1)[0].strip()   # inline comment
            os.environ.setdefault(k.strip(), v.strip('"').strip("'"))


def _get(name, default, cast: Any = str) -> Any:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw)


@dataclass
class RiskConfig:
    """Defaults follow a 5,000 USD two-phase account (phase 1 +400, phase 2 +200, daily loss
    200, static floor at 4,600, at most 150 USD at risk at once, no time limit). A limit set
    to 0 is switched off."""
    risk_per_trade_usd: float = 50.0      # planned loss if the stop is hit
    max_trades_per_day: int = 2
    daily_loss_limit: float = 200.0       # never lose more than this in a trading day
    start_balance: float = 5000.0         # balance the floor and the profit target refer to
    max_total_loss: float = 400.0         # static floor = start_balance - max_total_loss
    profit_target: float = 400.0          # stop opening trades at start_balance + target (phase 1: 400, phase 2: 200)
    max_open_risk: float = 150.0          # combined loss at the stops of all open and pending trades
    min_risk_usd: float = 20.0            # skip the trade when less room than this is left
    min_trading_days: int = 2             # per phase; needs account_start (the phase start date)
    first_week_loss_limit: float = 0.0    # optional: max loss during the first week
    weekly_profit_target: float = 0.0     # optional: stop opening trades once the week reaches this
    loss_buffer_usd: float = 15.0         # safety margin for slippage and gaps
    account_start: date | None = None     # first day of the account (first-week rule)
    contract_size: float = 100.0          # oz per lot on XAUUSD (replaced by the broker's value live)
    commission_rate: float = 0.000018     # per side, share of the trade value (FundedNext Stellar Lite metals: 0.0018%)
    leverage: float = 15.0                # metals leverage (Stellar Lite: 1:25 challenge, 1:15 funded)
    max_margin_use: float = 0.5           # never use more than this share of equity as margin
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 1.0
    day_reset_tz: str = "America/New_York"  # most prop firms reset the day at 17:00 New York
    day_reset_hour: int = 17
    count_all_account_pnl: bool = True    # limits apply to the whole account, not only bot trades


@dataclass
class NewsConfig:
    """Major announcements. Some funded accounts count only 40% of gains realized within 5
    minutes of a major release (losses count 100%): block new setups around releases, and
    optionally close winning positions just before them."""
    source: str = "off"                   # off | forexfactory | file | fixed:08:30,10:00 (New York times, every weekday)
    file: str = "news.csv"                # for source=file: columns time (ISO, New York time if no offset), title
    currencies: tuple = ("USD",)
    before_min: float = 6.0               # blackout starts this long before the release
    after_min: float = 5.0                # and ends this long after it
    close_winners: bool = False           # close positions in profit before the window (backtest: cuts 3R winners, worse than the 40% haircut)
    rule_window_min: float = 5.0          # the firm's window: closing inside it does not help
    pause_pending: bool = True            # cancel waiting orders before a release, re-place them after (backtest: about free)
    refresh_hours: float = 6.0


@dataclass
class JudgeConfig:
    """One LLM provider serves every agent of the committee."""
    enabled: bool = False
    provider: str = "anthropic"           # any name registered in goldbot.judge.providers
    api_key: str = ""                     # key for that provider
    model: str = ""                       # empty = provider default (if it has one)
    base_url: str = ""                    # for "openai_compatible" (OpenRouter, Groq, Ollama, ...)
    fail_mode: str = "open"               # "open": trade the tested strategy if the committee fails
    timeout_s: float = 45.0
    min_approvals: int = 2                # analysts that must approve (out of 3) before the chair
    use_chair: bool = True                # final LLM chair; False = vote count only


@dataclass
class Config:
    symbol: str = "XAUUSD"
    mode: str = "alert"                   # "alert" = notify only, "trade" = place orders
    require_demo: bool = True
    magic: int = 20261119
    max_spread: float = 0.50              # USD; skip setups when the live spread is wider
    strategy: str = "orb"
    strategy_params: dict = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    telegram_token: str = ""
    telegram_chat_id: str = ""
    journal_path: str = "goldbot_journal.csv"
    poll_seconds: float = 2.0
    # unattended operation (VPS or always-on laptop)
    mt5_path: str = ""                    # terminal64.exe; empty = the terminal already running
    mt5_login: int = 0                    # with password and server, the bot logs the terminal in
    mt5_password: str = ""
    mt5_server: str = ""
    state_file: str = "goldbot_state.json"
    log_file: str = "logs/goldbot.log"
    lock_port: int = 47653                # local port used to refuse a second running copy
    heartbeat_ny: str = "17:05"           # daily Telegram status (New York time), empty = off

    @classmethod
    def from_env(cls, env_file=".env"):
        load_env(env_file)
        start = _get("GB_ACCOUNT_START", "")
        risk = RiskConfig(
            risk_per_trade_usd=_get("GB_RISK_PER_TRADE", 50.0, float),
            max_trades_per_day=_get("GB_MAX_TRADES_PER_DAY", 2, int),
            daily_loss_limit=_get("GB_DAILY_LOSS_LIMIT", 200.0, float),
            start_balance=_get("GB_START_BALANCE", 5000.0, float),
            max_total_loss=_get("GB_MAX_TOTAL_LOSS", 400.0, float),
            profit_target=_get("GB_PROFIT_TARGET", 400.0, float),
            max_open_risk=_get("GB_MAX_OPEN_RISK", 150.0, float),
            min_risk_usd=_get("GB_MIN_RISK", 20.0, float),
            min_trading_days=_get("GB_MIN_TRADING_DAYS", 2, int),
            first_week_loss_limit=_get("GB_FIRST_WEEK_LOSS_LIMIT", 0.0, float),
            weekly_profit_target=_get("GB_WEEKLY_PROFIT_TARGET", 0.0, float),
            loss_buffer_usd=_get("GB_LOSS_BUFFER", 15.0, float),
            account_start=date.fromisoformat(start) if start else None,
            volume_max=_get("GB_MAX_LOTS", 1.0, float),
            commission_rate=_get("GB_COMMISSION_RATE", 0.000018, float),
            leverage=_get("GB_LEVERAGE", 15.0, float),
            max_margin_use=_get("GB_MAX_MARGIN_USE", 0.5, float),
            count_all_account_pnl=_get("GB_COUNT_ALL_ACCOUNT_PNL", True, bool),
        )
        judge = JudgeConfig(
            enabled=_get("GB_LLM_ENABLED", False, bool),
            provider=_get("GB_LLM_PROVIDER", "anthropic"),
            api_key=_get("GB_LLM_API_KEY", ""),
            model=_get("GB_LLM_MODEL", ""),
            base_url=_get("GB_LLM_BASE_URL", ""),
            fail_mode=_get("GB_LLM_FAIL_MODE", "open"),
            timeout_s=_get("GB_LLM_TIMEOUT", 45.0, float),
            min_approvals=_get("GB_LLM_MIN_APPROVALS", 2, int),
            use_chair=_get("GB_LLM_USE_CHAIR", True, bool),
        )
        news = NewsConfig(
            source=_get("GB_NEWS_SOURCE", "off"),
            file=_get("GB_NEWS_FILE", "news.csv"),
            currencies=tuple(c.strip().upper() for c in _get("GB_NEWS_CURRENCIES", "USD").split(",") if c.strip()),
            before_min=_get("GB_NEWS_BEFORE_MIN", 6.0, float),
            after_min=_get("GB_NEWS_AFTER_MIN", 5.0, float),
            close_winners=_get("GB_NEWS_CLOSE_WINNERS", False, bool),
            pause_pending=_get("GB_NEWS_PAUSE_PENDING", True, bool),
        )
        sessions = _get("GB_ORB_SESSIONS", "ny_comex")
        params = {
            "sessions": tuple(s.strip() for s in sessions.split(",") if s.strip()),
            "range_min": _get("GB_ORB_RANGE_MIN", 60, int),
            "window_min": _get("GB_ORB_WINDOW_MIN", 120, int),
            "stop": _get("GB_ORB_STOP", "mid"),
            "target_r": _get("GB_ORB_TARGET_R", 3.0, float),
            "trend_filter": _get("GB_ORB_TREND_FILTER", True, bool),
        }
        cfg = cls(
            symbol=_get("SYMBOL", "XAUUSD"),
            mode=_get("GB_MODE", "alert"),
            require_demo=_get("GB_REQUIRE_DEMO", True, bool),
            max_spread=_get("GB_MAX_SPREAD", 0.50, float),
            strategy=_get("GB_STRATEGY", "orb"),
            strategy_params=params,
            risk=risk,
            news=news,
            judge=judge,
            telegram_token=_get("TELEGRAM_TOKEN", ""),
            telegram_chat_id=_get("TELEGRAM_CHAT_ID", ""),
            journal_path=_get("GB_JOURNAL", "goldbot_journal.csv"),
            mt5_path=_get("GB_MT5_PATH", ""),
            mt5_login=_get("GB_MT5_LOGIN", 0, int),
            mt5_password=_get("GB_MT5_PASSWORD", ""),
            mt5_server=_get("GB_MT5_SERVER", ""),
            state_file=_get("GB_STATE_FILE", "goldbot_state.json"),
            log_file=_get("GB_LOG_FILE", "logs/goldbot.log"),
            lock_port=_get("GB_LOCK_PORT", 47653, int),
            heartbeat_ny=_get("GB_HEARTBEAT_NY", "17:05"),
        )
        cfg.validate()
        return cfg

    def validate(self):
        if self.mode not in ("alert", "trade"):
            raise ValueError(f"GB_MODE must be 'alert' or 'trade', got {self.mode!r}")
        if self.judge.fail_mode not in ("open", "closed"):
            raise ValueError("GB_LLM_FAIL_MODE must be 'open' or 'closed'")
        if self.judge.enabled:
            from goldbot.judge.providers import available_providers
            if self.judge.provider not in available_providers():
                raise ValueError(f"GB_LLM_PROVIDER {self.judge.provider!r} is not registered; "
                                 f"choose one of {sorted(available_providers())}")
        r = self.risk
        if r.risk_per_trade_usd + r.loss_buffer_usd > r.daily_loss_limit:
            raise ValueError("risk per trade plus buffer must stay below the daily loss limit")
        if r.max_open_risk and r.risk_per_trade_usd > r.max_open_risk:
            raise ValueError("risk per trade is above the maximum open risk (GB_MAX_OPEN_RISK)")
        src = self.news.source
        if not (src in ("off", "forexfactory", "file") or src.startswith("fixed:")):
            raise ValueError("GB_NEWS_SOURCE must be off, forexfactory, file or fixed:HH:MM,...")
