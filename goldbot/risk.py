"""Funded-account risk manager: position sizing and hard limits.

All limits are checked worst case before an order is placed:
    realized PnL (today / this week / since account start)
  - risk of everything already open or pending (loss if every stop is hit)
  - risk of the new trade
  - a buffer for slippage and gaps
must stay above the limit. Limits apply to the whole account by default, because funded
rules count every trade, not only the bot's.
"""
import math
from dataclasses import dataclass

from goldbot.config import RiskConfig
from goldbot.models import Setup


@dataclass
class AccountState:
    day_realized: float = 0.0
    week_realized: float = 0.0
    since_start_realized: float = 0.0
    open_risk: float = 0.0          # bot positions + pending groups + external positions
    trades_today: int = 0           # filled entries today + pending groups reserved today
    in_first_week: bool = False
    external_unprotected: bool = False   # a non-bot position without a stop is open
    equity: float = 0.0

    def as_dict(self, rc: RiskConfig) -> dict:
        return {
            "day_realized_usd": round(self.day_realized, 2),
            "week_realized_usd": round(self.week_realized, 2),
            "open_risk_usd": round(self.open_risk, 2),
            "trades_today": self.trades_today,
            "daily_loss_limit_usd": rc.daily_loss_limit,
            "room_today_usd": round(rc.daily_loss_limit + self.day_realized - self.open_risk, 2),
            "in_first_week": self.in_first_week,
            "first_week_room_usd": round(rc.first_week_loss_limit + self.since_start_realized
                                         - self.open_risk, 2) if self.in_first_week else None,
            "weekly_target_usd": rc.weekly_profit_target,
            "to_weekly_target_usd": round(rc.weekly_profit_target - self.week_realized, 2),
        }


@dataclass
class RiskVerdict:
    ok: bool
    reason: str
    lots: float = 0.0
    planned_risk: float = 0.0


class FundedRiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def size(self, setup: Setup) -> tuple[float, float]:
        """Lots so that hitting the stop loses at most risk_per_trade_usd (rounded down)."""
        c = self.cfg
        per_lot = setup.risk_per_unit * c.contract_size
        if per_lot <= 0:
            return 0.0, 0.0
        steps = math.floor(c.risk_per_trade_usd / per_lot / c.volume_step + 1e-9)
        lots = round(min(steps * c.volume_step, c.volume_max), 2)
        return lots, lots * per_lot

    def check(self, setup: Setup, st: AccountState) -> RiskVerdict:
        c = self.cfg
        if st.external_unprotected:
            return RiskVerdict(False, "a non-bot position without a stop is open; risk unknown")
        if st.week_realized >= c.weekly_profit_target:
            return RiskVerdict(False, f"weekly target reached ({st.week_realized:.2f} >= "
                                      f"{c.weekly_profit_target}); no new trades this week")
        if st.trades_today >= c.max_trades_per_day:
            return RiskVerdict(False, f"{c.max_trades_per_day} trades already taken or pending today")
        lots, risk = self.size(setup)
        if lots < c.volume_min:
            return RiskVerdict(False, f"stop too wide: minimum lot would risk "
                                      f"{c.volume_min * setup.risk_per_unit * c.contract_size:.2f} "
                                      f"> {c.risk_per_trade_usd}")
        worst_day = st.day_realized - st.open_risk - risk - c.loss_buffer_usd
        if worst_day < -c.daily_loss_limit:
            return RiskVerdict(False, f"daily limit: worst case {worst_day:.2f} < "
                                      f"-{c.daily_loss_limit}", lots, risk)
        if st.in_first_week:
            worst_start = st.since_start_realized - st.open_risk - risk - c.loss_buffer_usd
            if worst_start < -c.first_week_loss_limit:
                return RiskVerdict(False, f"first-week limit: worst case {worst_start:.2f} < "
                                          f"-{c.first_week_loss_limit}", lots, risk)
        return RiskVerdict(True, "ok", lots, risk)
