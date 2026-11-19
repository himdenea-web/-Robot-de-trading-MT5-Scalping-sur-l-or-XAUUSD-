"""Opening Range Breakout strategy plug-in.

For each configured session: when the opening range (first `range_min` minutes after the
session open, local time with daylight saving) has closed, emit an OCO pair of stop orders
at the range high (buy) and low (sell). Optional daily-trend filter keeps only the side that
agrees with the D1 trend. Pending orders expire at the end of the entry window; positions are
closed at the session exit time.

Research results for this family are in backtest/ (it is not a proven edge).
"""
from datetime import time

import pandas as pd

from goldbot.market import Market, local_to_utc, utc_to_local_date
from goldbot.models import Setup
from goldbot.strategies.base import Strategy

SESSIONS = {
    # name: (timezone, open, exit)
    "london": ("Europe/London", time(8, 0), time(12, 0)),
    "ny_comex": ("America/New_York", time(8, 20), time(12, 0)),
    "ny_equity": ("America/New_York", time(9, 30), time(13, 0)),
}


class OpeningRangeBreakout(Strategy):
    name = "orb"

    def __init__(self, sessions=("ny_comex",), range_min=60, window_min=120, stop="mid",
                 target_r=3.0, trend_filter=True, min_range_atr=0.0, max_range_atr=9.9,
                 first_break_only=True):
        super().__init__(sessions=tuple(sessions), range_min=range_min, window_min=window_min,
                         stop=stop, target_r=target_r, trend_filter=trend_filter,
                         min_range_atr=min_range_atr, max_range_atr=max_range_atr,
                         first_break_only=first_break_only)
        self.first_break_only = first_break_only
        unknown = set(sessions) - set(SESSIONS)
        if unknown:
            raise ValueError(f"unknown sessions {unknown}; choose from {sorted(SESSIONS)}")
        if stop not in ("range", "mid"):
            raise ValueError("stop must be 'range' or 'mid'")
        self.sessions = tuple(sessions)
        self.range_min, self.window_min = range_min, window_min
        self.stop, self.target_r, self.trend_filter = stop, target_r, trend_filter
        self.min_range_atr, self.max_range_atr = min_range_atr, max_range_atr
        self._times = {}
        self._done = set()

    def session_times(self, session, day):
        key = (session, day)
        if key not in self._times:
            tz, t_open, t_exit = SESSIONS[session]
            o = local_to_utc(day, t_open, tz)
            r = o + pd.Timedelta(minutes=self.range_min)
            x = local_to_utc(day, t_exit, tz)
            w = min(r + pd.Timedelta(minutes=self.window_min), x)
            self._times[key] = (o, r, w, x)
        return self._times[key]

    def _next_event(self, now):
        """Earliest range close after `now` among the next few days (cheap fast path)."""
        best = None
        base = utc_to_local_date(now, "UTC")
        for k in range(0, 5):
            day = base + pd.Timedelta(days=k)
            day = day.date() if hasattr(day, "date") else day
            for s in self.sessions:
                if day.weekday() >= 5 or (s, day) in self._done:
                    continue
                t_range = self.session_times(s, day)[1]
                if t_range > now and (best is None or t_range < best):
                    best = t_range
            if best is not None and k >= 1:
                break
        return best

    def on_bar(self, market: Market) -> list[Setup]:
        now = market.now()
        nxt = getattr(self, "_next", None)
        if nxt is not None and now < nxt:
            return []
        out = self._on_bar_full(market, now)
        self._next = self._next_event(now)
        return out

    def _on_bar_full(self, market, now):
        out = []
        for s in self.sessions:
            tz = SESSIONS[s][0]
            day = utc_to_local_date(now, tz)
            if day.weekday() >= 5 or (s, day) in self._done:
                continue
            t_open, t_range, t_window, t_exit = self.session_times(s, day)
            if now < t_range:
                continue
            self._done.add((s, day))
            if now >= t_window:       # started late (e.g. bot restarted): skip this session
                continue
            out += self._build(market, s, day, t_open, t_range, t_window, t_exit)
        return out

    def _build(self, market, session, day, t_open, t_range, t_window, t_exit):
        hi, lo, n = market.hilo(t_open, t_range)
        if n < 0.8 * self.range_min or not hi > lo:
            return []
        d1 = market.daily(t_open.date())
        atr, trend = d1.get("atr"), d1.get("trend", 0)
        if not atr or atr != atr:     # missing or NaN
            return []
        width = hi - lo
        if width < self.min_range_atr * atr or width > self.max_range_atr * atr:
            return []
        spread = market.spread()
        mid = (hi + lo) / 2
        ctx = {"session": session, "range_high": round(hi, 2), "range_low": round(lo, 2),
               "range_width": round(width, 2), "d1_atr": round(atr, 2),
               "range_atr_ratio": round(width / atr, 3), "trend_d1": int(trend),
               "prev_close": d1.get("prev_close"), "spread": round(spread, 2),
               "minutes_left_in_window": int((t_window - t_range).total_seconds() // 60)}
        group = f"orb-{session}-{day.isoformat()}"
        setups = []
        # long: buy stop at the range high (ask), exits on bid
        sl_long = lo if self.stop == "range" else mid
        risk_l = hi - sl_long
        # short: sell stop at the range low (bid), exits on ask, so the stop carries the spread
        sl_short = (hi if self.stop == "range" else mid) + spread
        risk_s = sl_short - lo
        for d, entry, sl, risk in ((1, hi, sl_long, risk_l), (-1, lo, sl_short, risk_s)):
            if risk <= 0 or (self.trend_filter and d != trend):
                continue
            tp = entry + d * self.target_r * risk if self.target_r > 0 else None
            # with the trend filter only one leg is placed; if the other side of the range
            # breaks first, the day is void (this is the rule that was researched)
            inv = (lo if d > 0 else hi) if (self.trend_filter and self.first_break_only) else None
            setups.append(Setup(strategy=self.name, group=group, direction=d, entry=entry, sl=sl,
                                tp=tp, created=t_range, expires=t_window, exit_by=t_exit,
                                context=dict(ctx), invalidate=inv))
        return setups
