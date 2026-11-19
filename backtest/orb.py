"""Opening Range Breakout (ORB) research engine for XAUUSD on M1 bid data.

Per session and day:
  1. range = high/low of the first `range_min` minutes after the session open (local time, DST aware)
  2. entry = first break of the range within `window_min` minutes after the range closes
     (long triggers when ask >= range high, short when bid <= range low; one trade per session, OCO)
  3. stop  = opposite side of the range ("range") or the range midpoint ("mid")
  4. exit  = target of `target_r` x risk (0 = no target), stop, or time exit at `exit_time` local

Prices are bid. Longs fill at ask (+ slippage) and exit on bid; shorts fill at bid (- slippage)
and exit on ask. If the stop is touched on the entry bar the trade is counted as stopped
(conservative); a target is only checked from the next bar.
"""
from dataclasses import dataclass
from datetime import time as dtime

import numpy as np
import pandas as pd

SESSIONS = {
    # name: (timezone, open, exit)
    "london": ("Europe/London", dtime(8, 0), dtime(12, 0)),
    "ny_comex": ("America/New_York", dtime(8, 20), dtime(12, 0)),
    "ny_equity": ("America/New_York", dtime(9, 30), dtime(13, 0)),
}


@dataclass(frozen=True)
class OrbParams:
    session: str = "ny_comex"
    range_min: int = 30
    window_min: int = 120
    stop: str = "range"         # "range" | "mid"
    target_r: float = 2.0       # 0 = time exit only
    trend_filter: bool = False  # trade only in the direction of D1 close vs D1 EMA20 (prior day)
    min_range_atr: float = 0.0  # skip if range < k * D1 ATR(14)
    max_range_atr: float = 9.9  # skip if range > k * D1 ATR(14)


class M1:
    def __init__(self, m1):
        self.t = m1["time"].to_numpy()
        self.o, self.h, self.l, self.c = (m1[k].to_numpy() for k in ("open", "high", "low", "close"))
        d1 = (m1.set_index("time").resample("1D")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna())
        tr = pd.concat([d1.high - d1.low, (d1.high - d1.close.shift()).abs(),
                        (d1.low - d1.close.shift()).abs()], axis=1).max(axis=1)
        # values known at the start of each day (computed on prior days only)
        self.d1 = pd.DataFrame({
            "trend": np.sign(d1.close - d1.close.ewm(span=20, adjust=False).mean()).shift(1),
            "atr": tr.ewm(alpha=1 / 14, adjust=False).mean().shift(1),
        })
        self.d1.index = self.d1.index.date
        self._tables = {}

    def idx(self, ts_utc):
        return np.searchsorted(self.t, np.asarray(ts_utc, dtype="datetime64[ns]"))

    def table(self, session, range_min, window_min):
        key = (session, range_min, window_min)
        if key in self._tables:
            return self._tables[key]
        tz, t_open, t_exit = SESSIONS[session]
        days = pd.bdate_range(pd.Timestamp(self.t[0]).normalize(), pd.Timestamp(self.t[-1]).normalize())
        loc_open = pd.DatetimeIndex([pd.Timestamp.combine(d.date(), t_open) for d in days]).tz_localize(tz)
        loc_exit = pd.DatetimeIndex([pd.Timestamp.combine(d.date(), t_exit) for d in days]).tz_localize(tz)
        o_utc = loc_open.tz_convert("UTC").tz_localize(None)
        x_utc = loc_exit.tz_convert("UTC").tz_localize(None)
        j0 = self.idx(o_utc)
        j1 = self.idx(o_utc + pd.Timedelta(minutes=range_min))
        j2 = self.idx(o_utc + pd.Timedelta(minutes=range_min + window_min))
        j3 = self.idx(x_utc)
        rows = []
        for k, d in enumerate(days):
            a, b = j0[k], j1[k]
            if b - a < range_min * 0.8 or j3[k] <= b:   # missing data / holiday
                continue
            rows.append((d.date(), a, b, min(j2[k], j3[k]), j3[k],
                         self.h[a:b].max(), self.l[a:b].min()))
        tb = pd.DataFrame(rows, columns=["day", "j_open", "j_range", "j_window", "j_exit", "hi", "lo"])
        tb = tb.join(self.d1, on="day")
        self._tables[key] = tb
        return tb


def run_orb(m: M1, p: OrbParams, spread=0.30, slip=0.10):
    tb = m.table(p.session, p.range_min, p.window_min)
    h, l, o, c, t = m.h, m.l, m.o, m.c, m.t
    out = []
    for r in tb.itertuples(index=False):
        width = r.hi - r.lo
        if not np.isfinite(r.atr) or width <= 0:
            continue
        if width < p.min_range_atr * r.atr or width > p.max_range_atr * r.atr:
            continue
        a, b = r.j_range, r.j_window
        if b <= a:
            continue
        up = np.flatnonzero(h[a:b] + spread >= r.hi)
        dn = np.flatnonzero(l[a:b] <= r.lo)
        ku = up[0] if len(up) else 10**9
        kd = dn[0] if len(dn) else 10**9
        if ku == kd == 10**9:
            continue
        if ku == kd:   # both in one M1 bar: take the side closer to that bar's open
            j = a + ku
            d = 1 if (r.hi - (o[j] + spread)) < (o[j] - r.lo) else -1
        else:
            d = 1 if ku < kd else -1
        j = a + (ku if d > 0 else kd)
        if p.trend_filter and d != r.trend:
            continue
        if d > 0:
            entry = max(r.hi, o[j] + spread) + slip
            sl = r.lo if p.stop == "range" else (r.hi + r.lo) / 2
        else:
            entry = min(r.lo, o[j]) - slip
            sl = (r.hi + spread) if p.stop == "range" else (r.hi + r.lo) / 2 + spread
        risk = (entry - sl) * d
        if risk <= 0:
            continue
        tp = entry + d * p.target_r * risk if p.target_r > 0 else None
        # scan for exits until the time exit
        end = r.j_exit
        adj = spread if d < 0 else 0.0           # shorts exit on ask
        hh, ll, oo = h[j:end] + adj, l[j:end] + adj, o[j:end] + adj
        hit_sl = (ll <= sl) if d > 0 else (hh >= sl)
        if tp is not None:
            hit_tp = (hh >= tp) if d > 0 else (ll <= tp)
            hit_tp[0] = False
        else:
            hit_tp = np.zeros_like(hit_sl)
        any_hit = hit_sl | hit_tp
        if any_hit.any():
            k = int(np.argmax(any_hit))
            if hit_sl[k]:
                gap = k > 0 and ((oo[k] <= sl) if d > 0 else (oo[k] >= sl))
                px, why = ((oo[k] if gap else sl) - d * slip), "sl"
            else:
                gap = (oo[k] >= tp) if d > 0 else (oo[k] <= tp)
                px, why = (oo[k] if gap else tp), "tp"
            xj = j + k
        else:
            xj = end - 1 if end > j else j
            px, why = c[xj] + adj, "time"
        out.append((r.day, t[j], t[xj], d, entry, sl, px, why, risk, width, r.atr,
                    (px - entry) * d / risk))
    return pd.DataFrame(out, columns=["day", "entry_time", "exit_time", "dir", "entry", "sl", "exit",
                                      "why", "risk", "range", "d1_atr", "R"])


def combine(*trade_sets):
    """Merge several sessions' trades (e.g. London + NY = up to 2 trades a day)."""
    tr = pd.concat(trade_sets, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return tr


def funded_sim(tr, risk_usd=90.0, daily_loss=200.0, weekly_target=600.0, max_per_day=2,
               start_balance=6000.0):
    """Applies funded-account rules to an R stream. A trade is skipped if it could push the
    day below -daily_loss, or if the week already reached the weekly target.
    Returns the trades taken with a USD pnl column and a weekly table."""
    tr = tr.sort_values("entry_time").reset_index(drop=True).copy()
    tr["week"] = pd.to_datetime(tr["day"]) - pd.to_timedelta(pd.to_datetime(tr["day"]).dt.weekday, unit="D")
    taken, day_pnl, week_pnl, day_n = [], {}, {}, {}
    for r in tr.itertuples():
        dp, wp, n = day_pnl.get(r.day, 0.0), week_pnl.get(r.week, 0.0), day_n.get(r.day, 0)
        if n >= max_per_day or dp - risk_usd < -daily_loss or wp >= weekly_target:
            continue
        pnl = r.R * risk_usd
        day_pnl[r.day], week_pnl[r.week], day_n[r.day] = dp + pnl, wp + pnl, n + 1
        taken.append(r.Index)
    t = tr.loc[taken].copy()
    t["pnl"] = t["R"] * risk_usd
    wk = t.groupby("week")["pnl"].agg(["sum", "size"]).rename(columns={"sum": "pnl", "size": "trades"})
    return t, wk


def challenge_stats(t, target=600.0, max_loss=400.0, horizon_weeks=None):
    """For every possible start week: does cumulative pnl reach +target before -max_loss
    (optionally within horizon_weeks)? Path-dependent, trade by trade."""
    weeks = sorted(t["week"].unique())
    res = []
    for w0 in weeks:
        s = t[t["week"] >= w0]
        if horizon_weeks:
            s = s[s["week"] < w0 + pd.Timedelta(weeks=horizon_weeks)]
        eq = s["pnl"].cumsum().to_numpy()
        hit = np.flatnonzero(eq >= target)
        bust = np.flatnonzero(eq <= -max_loss)
        h, b = (hit[0] if len(hit) else 10**9), (bust[0] if len(bust) else 10**9)
        res.append("pass" if h < b else ("fail" if b < h else "open"))
    return pd.Series(res, index=weeks).value_counts(normalize=True)
