"""Simulated market and broker on M1 bid bars, for backtests of the real engine.

Fill model (same as backtest/orb.py):
  * a buy stop triggers when ask (bid high + spread) reaches the level; it fills at the
    level, or at the bar open if price gapped through it, plus slippage
  * a sell stop triggers when bid low reaches the level; fills at the level or the open,
    minus slippage
  * longs exit on bid, shorts exit on ask (bid + spread)
  * on the entry bar only the stop is checked (conservative); targets from the next bar
  * a gap through a stop fills at the bar open, minus slippage
  * time exits close at the last bar's close
"""
import bisect
import itertools

import numpy as np
import pandas as pd

from goldbot.models import Fill, Order, Position, Setup


class SimMarket:
    def __init__(self, m1: pd.DataFrame, spread=0.30):
        self.t = m1["time"].to_numpy()
        self.o, self.h, self.l, self.c = (m1[k].to_numpy() for k in ("open", "high", "low", "close"))
        self._spread = spread
        self.i = 0
        d1 = (m1.set_index("time").resample("1D")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna())
        tr = pd.concat([d1.high - d1.low, (d1.high - d1.close.shift()).abs(),
                        (d1.low - d1.close.shift()).abs()], axis=1).max(axis=1)
        known = pd.DataFrame({
            "trend": np.sign(d1.close - d1.close.ewm(span=20, adjust=False).mean()).shift(1),
            "atr": tr.ewm(alpha=1 / 14, adjust=False).mean().shift(1),
            "prev_close": d1.close.shift(1),
        })
        self._daily = {d.date(): row for d, row in zip(known.index, known.to_dict("records"))}

    def now(self) -> pd.Timestamp:
        return pd.Timestamp(self.t[self.i])

    def hilo(self, start, end):
        a = int(np.searchsorted(self.t, np.datetime64(start)))
        b = int(np.searchsorted(self.t, np.datetime64(end)))
        b = min(b, self.i)          # never look at bars that are not closed yet
        if b <= a:
            return float("nan"), float("nan"), 0
        return float(self.h[a:b].max()), float(self.l[a:b].min()), b - a

    def daily(self, day):
        r = self._daily.get(day)
        if r is None:
            return {"trend": 0, "atr": float("nan"), "prev_close": None}
        tr = r["trend"]
        return {"trend": 0 if tr != tr else int(tr), "atr": float(r["atr"]),
                "prev_close": None if r["prev_close"] != r["prev_close"] else round(r["prev_close"], 2)}

    def spread(self):
        return self._spread

    def last_price(self):
        return float(self.c[self.i - 1]) if self.i > 0 else float(self.o[0])


class SimBroker:
    def __init__(self, market: SimMarket, contract_size=100.0, slip=0.10, equity=6000.0,
                 commission_rate=0.0):
        self.m = market
        self.contract = contract_size
        self.commission_rate = commission_rate       # per side, share of the trade value
        self.slip = slip
        self.balance = equity
        self._ids = itertools.count(1)
        self._orders: dict[str, Order] = {}
        self._pos: dict[str, tuple[Position, int]] = {}   # id -> (position, entry bar index)
        self.fills: list[Fill] = []
        self._fill_times: list[int] = []

    # ---- interface
    def account(self):
        return {"equity": self.balance, "balance": self.balance, "is_demo": True}

    def place(self, setup: Setup, lots: float, planned_risk: float):
        o = Order(id=f"o{next(self._ids)}", setup=setup, lots=lots, planned_risk=planned_risk)
        self._orders[o.id] = o
        return o

    def cancel(self, order):
        return self._orders.pop(order.id, None) is not None

    def pending(self):
        return list(self._orders.values())

    def positions(self):
        return [p for p, _ in self._pos.values()]

    def close(self, position, reason):
        if position.id not in self._pos:
            return None
        adj = self.m._spread if position.setup.direction < 0 else 0.0
        return self._exit(position, self.m.last_price() + adj, self.m.i - 1, reason)

    def fills_since(self, start):
        # fills are appended in time order, so a bisect on exit times is enough
        k = bisect.bisect_left(self._fill_times, pd.Timestamp(start).value)
        return self.fills[k:]

    def realized_since(self, start):
        return [(f.exit_time, f.pnl) for f in self.fills_since(start)]

    def external_risk(self):
        return 0.0, False

    # ---- simulation step: process bar i (called after the engine acted at its open)
    def process_bar(self):
        m, i, sp = self.m, self.m.i, self.m._spread
        o, h, l = m.o[i], m.h[i], m.l[i]
        # pending stop orders (one fill per OCO group per bar; pick the side closer to the open)
        trig = []
        for od in list(self._orders.values()):
            s = od.setup
            if s.direction > 0:
                hit, dist = h + sp >= s.entry, s.entry - (o + sp)
                inv = s.invalidate is not None and l <= s.invalidate
                inv_dist = (o - s.invalidate) if inv else None
            else:
                hit, dist = l <= s.entry, o - s.entry
                inv = s.invalidate is not None and h + sp >= s.invalidate
                inv_dist = (s.invalidate - (o + sp)) if inv else None
            if inv and (not hit or inv_dist <= dist):
                self._orders.pop(od.id, None)     # the other side broke first: setup void
                continue
            if hit:
                trig.append((dist, od))
        seen = set()
        for _, od in sorted(trig, key=lambda x: x[0]):
            s = od.setup
            if s.group in seen or od.id not in self._orders:
                continue
            seen.add(s.group)
            for other in [x for x in self._orders.values() if x.setup.group == s.group]:
                self._orders.pop(other.id, None)          # OCO: fill this one, drop the rest
            if s.direction > 0:
                px = max(s.entry, o + sp) + self.slip
            else:
                px = min(s.entry, o) - self.slip
            pos = Position(id=f"p{od.id[1:]}", setup=s, lots=od.lots, entry_price=px,
                           entry_time=pd.Timestamp(m.t[i]), planned_risk=od.planned_risk)
            self._pos[pos.id] = (pos, i)
        # exits
        for pid, (p, j0) in list(self._pos.items()):
            s, d = p.setup, p.setup.direction
            adj = sp if d < 0 else 0.0
            hh, ll, oo = h + adj, l + adj, o + adj
            hit_sl = ll <= s.sl if d > 0 else hh >= s.sl
            hit_tp = False if (i == j0 or s.tp is None) else (hh >= s.tp if d > 0 else ll <= s.tp)
            if hit_sl:
                gap = i > j0 and (oo <= s.sl if d > 0 else oo >= s.sl)
                self._exit(p, (oo if gap else s.sl) - d * self.slip, i, "sl")
            elif hit_tp:
                gap = oo >= s.tp if d > 0 else oo <= s.tp
                self._exit(p, oo if gap else s.tp, i, "tp")

    def _exit(self, p: Position, px: float, i: int, reason: str):
        self._pos.pop(p.id, None)
        pnl = (px - p.entry_price) * p.setup.direction * p.lots * self.contract
        pnl -= self.commission_rate * (p.entry_price + px) * p.lots * self.contract
        self.balance += pnl
        f = Fill(position_id=p.id, setup=p.setup, lots=p.lots, entry_price=p.entry_price,
                 entry_time=p.entry_time, exit_price=px, exit_time=pd.Timestamp(self.m.t[max(i, 0)]),
                 reason=reason, pnl=pnl, planned_risk=p.planned_risk)
        self.fills.append(f)
        self._fill_times.append(f.exit_time.value)
        return f
