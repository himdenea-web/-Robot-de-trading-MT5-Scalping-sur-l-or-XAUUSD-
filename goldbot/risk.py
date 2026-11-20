"""Funded-account risk manager: position sizing and hard limits.

Every limit is checked worst case before an order is placed. For each limit the room left is
    what the limit allows - realized losses - risk of everything already open or pending - buffer
and the new trade's risk is the smallest of: the configured risk per trade and every room.
If that is below min_risk_usd the trade is skipped. Limits apply to the whole account by
default, because funded rules count every trade, not only the bot's.

Limits (0 switches one off):
    daily loss limit            realized today - open risk - new risk >= -daily_loss_limit
    static floor                balance - open risk - new risk >= start_balance - max_total_loss
    maximum exposure            open risk + new risk <= max_open_risk
    first-week loss (optional)  realized since start - open risk - new risk >= -first_week_loss_limit
    profit target               no new trade once balance >= start_balance + profit_target
    weekly target (optional)    no new trade once the week made weekly_profit_target
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
    balance: float | None = None    # account balance (None: unknown, floor and target unchecked)
    trading_days: int | None = None # days with at least one trade since the phase start (None: unknown)

    def as_dict(self, rc: RiskConfig) -> dict:
        d = {
            "day_realized_usd": round(self.day_realized, 2),
            "open_risk_usd": round(self.open_risk, 2),
            "trades_today": self.trades_today,
            "daily_loss_limit_usd": rc.daily_loss_limit,
            "room_today_usd": round(rc.daily_loss_limit + self.day_realized - self.open_risk, 2),
        }
        if self.balance is not None and rc.max_total_loss:
            d["floor_usd"] = rc.start_balance - rc.max_total_loss
            d["room_to_floor_usd"] = round(self.balance - d["floor_usd"] - self.open_risk, 2)
        if self.balance is not None and rc.profit_target:
            d["profit_target_usd"] = rc.profit_target
            d["to_profit_target_usd"] = round(rc.start_balance + rc.profit_target - self.balance, 2)
        if rc.max_open_risk:
            d["max_open_risk_usd"] = rc.max_open_risk
        if rc.first_week_loss_limit and self.in_first_week:
            d["first_week_room_usd"] = round(rc.first_week_loss_limit + self.since_start_realized
                                             - self.open_risk, 2)
        if rc.weekly_profit_target:
            d["week_realized_usd"] = round(self.week_realized, 2)
            d["weekly_target_usd"] = rc.weekly_profit_target
        return d


@dataclass
class RiskVerdict:
    ok: bool
    reason: str
    lots: float = 0.0
    planned_risk: float = 0.0


class FundedRiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def loss_per_lot(self, setup: Setup) -> float:
        """USD lost per lot if the stop is hit, including the commission on both sides."""
        c = self.cfg
        return (setup.risk_per_unit + 2 * c.commission_rate * setup.entry) * c.contract_size

    def max_lots_for_margin(self, setup: Setup, equity: float) -> float:
        c = self.cfg
        if not (c.leverage and equity and equity > 0):
            return c.volume_max
        margin_per_lot = setup.entry * c.contract_size / c.leverage
        return c.max_margin_use * equity / margin_per_lot

    def size(self, setup: Setup, budget: float | None = None, equity: float = 0.0) -> tuple[float, float]:
        """Lots so that hitting the stop (plus commission) loses at most budget (default: risk
        per trade), capped by margin, rounded down to the lot step."""
        c = self.cfg
        budget = c.risk_per_trade_usd if budget is None else budget
        per_lot = self.loss_per_lot(setup)
        if per_lot <= 0 or budget <= 0:
            return 0.0, 0.0
        lots = min(budget / per_lot, c.volume_max, self.max_lots_for_margin(setup, equity))
        steps = math.floor(lots / c.volume_step + 1e-9)
        lots = round(steps * c.volume_step, 2)
        return lots, lots * per_lot

    def rooms(self, st: AccountState) -> list[tuple[str, float]]:
        """(limit name, USD the new trade may risk under that limit)."""
        c = self.cfg
        out = [("risk per trade", c.risk_per_trade_usd),
               ("daily limit", c.daily_loss_limit + st.day_realized - st.open_risk - c.loss_buffer_usd)]
        if c.max_total_loss and st.balance is not None:
            floor = c.start_balance - c.max_total_loss
            out.append(("total loss floor", st.balance - floor - st.open_risk - c.loss_buffer_usd))
        if c.max_open_risk:
            out.append(("maximum exposure", c.max_open_risk - st.open_risk))
        if c.first_week_loss_limit and st.in_first_week:
            out.append(("first-week limit", c.first_week_loss_limit + st.since_start_realized
                        - st.open_risk - c.loss_buffer_usd))
        return out

    def _min_lot_trade(self, setup: Setup, st: AccountState) -> RiskVerdict:
        """Target reached but not enough trading days: allow one minimum-lot trade today."""
        c = self.cfg
        risk = c.volume_min * self.loss_per_lot(setup)
        name, budget = min(self.rooms(st), key=lambda x: x[1])
        if risk > budget:
            return RiskVerdict(False, f"minimum trading days: minimum lot risk {risk:.2f} > {budget:.2f} ({name})")
        return RiskVerdict(True, f"ok, minimum lot to complete {c.min_trading_days} trading days",
                           c.volume_min, risk)

    def check(self, setup: Setup, st: AccountState) -> RiskVerdict:
        c = self.cfg
        if st.external_unprotected:
            return RiskVerdict(False, "a non-bot position without a stop is open; risk unknown")
        if c.profit_target and st.balance is not None and st.balance >= c.start_balance + c.profit_target:
            if (c.min_trading_days and st.trading_days is not None
                    and st.trading_days < c.min_trading_days and st.trades_today == 0):
                return self._min_lot_trade(setup, st)
            return RiskVerdict(False, f"profit target reached (balance {st.balance:.2f} >= "
                                      f"{c.start_balance + c.profit_target:.2f}); no new trades")
        if c.weekly_profit_target and st.week_realized >= c.weekly_profit_target:
            return RiskVerdict(False, f"weekly target reached ({st.week_realized:.2f} >= "
                                      f"{c.weekly_profit_target}); no new trades this week")
        if st.trades_today >= c.max_trades_per_day:
            return RiskVerdict(False, f"{c.max_trades_per_day} trades already taken or pending today")
        name, budget = min(self.rooms(st), key=lambda x: x[1])
        if budget < c.min_risk_usd:
            return RiskVerdict(False, f"{name}: only {budget:.2f} USD of room left "
                                      f"(minimum {c.min_risk_usd})")
        lots, risk = self.size(setup, budget, st.equity)
        if lots < c.volume_min:
            if self.max_lots_for_margin(setup, st.equity) < c.volume_min:
                return RiskVerdict(False, "not enough margin for the minimum lot")
            return RiskVerdict(False, f"stop too wide: minimum lot would risk "
                                      f"{c.volume_min * self.loss_per_lot(setup):.2f} "
                                      f"> {budget:.2f} allowed ({name})")
        reason = "ok" if name == "risk per trade" else f"ok, size reduced by the {name}"
        return RiskVerdict(True, reason, lots, risk)
