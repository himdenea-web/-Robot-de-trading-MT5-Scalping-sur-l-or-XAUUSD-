"""The engine: one step per closed M1 bar, identical live and in backtests.

    manage (expire pending orders, time exits, OCO)  ->  account state  ->  strategy setups
    ->  risk check  ->  committee (optional, veto only)  ->  alert or order  ->  journal
"""
import pandas as pd

from goldbot.config import Config
from goldbot.market import trading_day, trading_week
from goldbot.news import NewsCalendar
from goldbot.risk import AccountState, FundedRiskManager


class Engine:
    def __init__(self, cfg: Config, strategy, market, broker, committee=None,
                 notifier=None, journal=None, news=None):
        self.cfg = cfg
        self.strategy = strategy
        self.market = market
        self.broker = broker
        self.risk = FundedRiskManager(cfg.risk)
        self.committee = committee if cfg.judge.enabled else None
        self.notifier = notifier
        self.journal = journal
        self.alerts: list[dict] = []          # alert mode: what would have been placed
        self.decisions: list[dict] = []
        self._known_positions: set[str] = set()
        self.news = news if news is not None else NewsCalendar(cfg.news, log=lambda m: self._log("news", msg=m))

    # ------------------------------------------------------------------ helpers
    def _day(self, ts):
        r = self.cfg.risk
        return trading_day(ts, r.day_reset_tz, r.day_reset_hour)

    def _log(self, event, **kw):
        if self.journal is not None:
            self.journal.write(event, **kw)

    def _notify(self, text):
        if self.notifier is not None:
            self.notifier.send(text)

    def account_state(self, now) -> AccountState:
        r = self.cfg.risk
        today = self._day(now)
        week = trading_week(today)
        start = r.account_start
        lookback_start = pd.Timestamp(min(week, start) if start else week) - pd.Timedelta(days=2)
        st = AccountState()
        for t, pnl in self.broker.realized_since(lookback_start):
            d = self._day(t)
            if d == today:
                st.day_realized += pnl
            if trading_week(d) == week:
                st.week_realized += pnl
            if start and d >= start:
                st.since_start_realized += pnl
        st.in_first_week = bool(start) and start <= today < (pd.Timestamp(start) + pd.Timedelta(days=7)).date()
        positions, pending = self.broker.positions(), self.broker.pending()
        groups_pending = {}
        for o in pending:
            groups_pending[o.setup.group] = max(groups_pending.get(o.setup.group, 0), o.planned_risk)
        ext_risk, ext_naked = self.broker.external_risk()
        st.open_risk = sum(p.planned_risk for p in positions) + sum(groups_pending.values()) + ext_risk
        st.external_unprotected = ext_naked
        # a pending OCO group reserves one trade slot; a filled group uses one
        st.trades_today = len(self._groups_entered_on(today, lookback_start) | set(groups_pending))
        acc = self.broker.account()
        st.equity = acc.get("equity", 0.0)
        st.balance = acc.get("balance")
        return st

    def _groups_entered_on(self, day, since) -> set:
        """Groups with an entry on `day`: open positions plus closed fills (a trade can open
        and close inside one bar, so positions alone would miss it)."""
        groups = {p.setup.group for p in self.broker.positions() if self._day(p.entry_time) == day}
        groups |= {f.setup.group for f in self.broker.fills_since(since) if self._day(f.entry_time) == day}
        return groups

    # ------------------------------------------------------------------ step
    def step(self):
        now = self.market.now()
        self._manage(now)
        setups = self.strategy.on_bar(self.market)
        if not setups:
            return
        by_group = {}
        for s in setups:
            by_group.setdefault(s.group, []).append(s)
        for group, legs in by_group.items():
            self._handle_group(now, group, legs)

    def _close_winners_before_news(self, now, positions):
        """Gains realized around a major release may count only partly: bank them before."""
        if not (self.news.enabled and self.cfg.news.close_winners and positions):
            return positions
        ev = self.news.next_within(now, 1.0)
        if ev is None or self.news.blackout(now + pd.Timedelta(minutes=1)) is None:
            return positions
        if ev[0] - now <= pd.Timedelta(minutes=self.cfg.news.rule_window_min):
            return positions                   # already inside the firm's window
        bid = self.market.last_price()
        ask = bid + self.market.spread()
        kept = []
        for p in positions:
            exit_px = bid if p.setup.direction > 0 else ask
            if (exit_px - p.entry_price) * p.setup.direction > 0:
                f = self.broker.close(p, "news")
                if f is not None:
                    self._log("exit", group=p.setup.group, side=p.setup.side, price=f.exit_price,
                              pnl=f.pnl, r=f.r, reason=f"news: {ev[1]} at {ev[0]:%H:%M} UTC")
                    self._notify(f"Closed {p.setup.side} in profit ({f.pnl:+.2f}) before {ev[1]}")
                    continue
            kept.append(p)
        return kept

    def _manage(self, now):
        self.news.refresh()
        positions = self.broker.positions()
        for p in positions:
            if p.id not in self._known_positions:
                self._known_positions.add(p.id)
                self._log("fill", group=p.setup.group, side=p.setup.side, price=p.entry_price,
                          lots=p.lots, risk=p.planned_risk)
                self._notify(f"Filled {p.setup.side} {self.cfg.symbol} {p.lots} lots at "
                             f"{p.entry_price:.2f} (SL {p.setup.sl:.2f}, TP {p.setup.tp})")
        # OCO: once any leg of a group has filled, cancel the other legs; also expire orders
        pending = self.broker.pending()
        if pending:
            since = now - pd.Timedelta(days=2)
            entered = {p.setup.group for p in positions} | {f.setup.group for f in self.broker.fills_since(since)}
            for o in pending:
                s = o.setup
                reason = "oco" if s.group in entered else ("expired" if now >= s.expires else "")
                if not reason and s.invalidate is not None:
                    # live brokers have no native "cancel if the other side breaks first";
                    # check closed bars since the setup (the simulator also does it intrabar)
                    hi, lo, n = self.market.hilo(s.created, now)
                    if n and ((s.direction > 0 and lo <= s.invalidate) or
                              (s.direction < 0 and hi + self.market.spread() >= s.invalidate)):
                        reason = "invalidated"
                if reason:
                    self.broker.cancel(o)
                    self._log("cancel", group=s.group, side=s.side, reason=reason)
        positions = self._close_winners_before_news(now, self.broker.positions())
        for p in positions:
            if now >= p.setup.exit_by:
                f = self.broker.close(p, "time")
                if f is not None:
                    self._log("exit", group=p.setup.group, side=p.setup.side, price=f.exit_price,
                              pnl=f.pnl, r=f.r, reason="time")

    def _handle_group(self, now, group, legs):
        st = self.account_state(now)
        spread = self.market.spread()
        if spread > self.cfg.max_spread:
            self._log("skip", group=group, reason=f"spread {spread:.2f} > {self.cfg.max_spread}")
            return
        ev = self.news.blackout(now) if self.news.enabled else None
        if ev is not None:
            reason = f"news blackout: {ev[1]} at {ev[0]:%H:%M} UTC"
            self._log("skip", group=group, reason=reason)
            self.decisions.append({"time": now, "group": group, "side": "", "stage": "news",
                                   "approved": False, "reason": reason})
            return
        approved = []
        for s in legs:
            v = self.risk.check(s, st)
            if not v.ok:
                self._log("risk_reject", group=group, side=s.side, reason=v.reason)
                self.decisions.append({"time": now, "group": group, "side": s.side,
                                       "stage": "risk", "approved": False, "reason": v.reason})
                continue
            if self.committee is not None:
                snap = {"spread": round(spread, 2), "last_price": round(self.market.last_price(), 2)}
                dec = self.committee.evaluate(s, st.as_dict(self.cfg.risk), snap)
                self._log("committee", group=group, side=s.side, reason=dec.summary(),
                          latency=round(dec.latency_s, 1))
                self.decisions.append({"time": now, "group": group, "side": s.side,
                                       "stage": "committee", "approved": dec.approved,
                                       "reason": dec.summary()})
                if not dec.approved:
                    continue
            approved.append((s, v))
        if not approved:
            return
        for s, v in approved:
            msg = (f"{s.strategy} {s.side} STOP {self.cfg.symbol} @ {s.entry:.2f} | SL {s.sl:.2f} | "
                   f"TP {'-' if s.tp is None else f'{s.tp:.2f}'} | {v.lots} lots | risk {v.planned_risk:.2f} | "
                   f"valid until {s.expires:%H:%M} UTC, exit by {s.exit_by:%H:%M} UTC")
            if self.cfg.mode == "alert":
                self.alerts.append({"time": now, "group": group, "setup": s, "lots": v.lots,
                                    "risk": v.planned_risk})
                self._log("alert", group=group, side=s.side, price=s.entry, sl=s.sl, tp=s.tp,
                          lots=v.lots, risk=v.planned_risk)
                self._notify("ALERT (not placed): " + msg)
            else:
                o = self.broker.place(s, v.lots, v.planned_risk)
                self._log("order" if o else "order_refused", group=group, side=s.side, price=s.entry,
                          sl=s.sl, tp=s.tp, lots=v.lots, risk=v.planned_risk)
                if o:
                    self._notify("ORDER placed: " + msg)
