"""Configuration, read from environment variables (and an optional .env file).

Every value has a safe default: alert-only mode, demo account required, LLM committee off.
"""
import os
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
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _get(name, default, cast: Any = str) -> Any:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw)


@dataclass
class RiskConfig:
    risk_per_trade_usd: float = 90.0      # planned loss if the stop is hit
    max_trades_per_day: int = 2
    daily_loss_limit: float = 200.0       # funded rule: never lose more than this in a day
    first_week_loss_limit: float = 400.0  # funded rule: max loss during the first week
    weekly_profit_target: float = 600.0   # stop opening trades once the week reaches this
    loss_buffer_usd: float = 15.0         # safety margin for slippage and gaps
    account_start: date | None = None     # first day of the funded account (first-week rule)
    contract_size: float = 100.0          # oz per lot on XAUUSD
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 1.0
    day_reset_tz: str = "America/New_York"  # most prop firms reset the day at 17:00 New York
    day_reset_hour: int = 17
    count_all_account_pnl: bool = True    # limits apply to the whole account, not only bot trades


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
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    telegram_token: str = ""
    telegram_chat_id: str = ""
    journal_path: str = "goldbot_journal.csv"
    poll_seconds: float = 2.0

    @classmethod
    def from_env(cls, env_file=".env"):
        load_env(env_file)
        start = _get("GB_ACCOUNT_START", "")
        risk = RiskConfig(
            risk_per_trade_usd=_get("GB_RISK_PER_TRADE", 90.0, float),
            max_trades_per_day=_get("GB_MAX_TRADES_PER_DAY", 2, int),
            daily_loss_limit=_get("GB_DAILY_LOSS_LIMIT", 200.0, float),
            first_week_loss_limit=_get("GB_FIRST_WEEK_LOSS_LIMIT", 400.0, float),
            weekly_profit_target=_get("GB_WEEKLY_PROFIT_TARGET", 600.0, float),
            loss_buffer_usd=_get("GB_LOSS_BUFFER", 15.0, float),
            account_start=date.fromisoformat(start) if start else None,
            volume_max=_get("GB_MAX_LOTS", 1.0, float),
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
            judge=judge,
            telegram_token=_get("TELEGRAM_TOKEN", ""),
            telegram_chat_id=_get("TELEGRAM_CHAT_ID", ""),
            journal_path=_get("GB_JOURNAL", "goldbot_journal.csv"),
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
