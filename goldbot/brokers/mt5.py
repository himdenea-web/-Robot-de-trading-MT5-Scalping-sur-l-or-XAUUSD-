"""MetaTrader 5 market feed and broker (Windows, MetaTrader5 Python package).

Times: MT5 returns bar and deal times as *server* time expressed as epoch seconds. The broker
measures the server offset from the latest tick and converts everything to UTC, so the rest
of the framework only ever sees UTC.

Safety and unattended operation:
  * refuses to run on a real account when require_demo is set
  * only touches orders and positions carrying this bot's magic number
  * can start and log in the terminal itself (path, login, password, server), and
    reconnects when the terminal or its link to the server drops
  * on start, adopts its own leftover orders and positions using the state file (so time
    exits, expiries and planned risk survive a restart) and rebuilds recent closed trades
    from the deal history (so the daily trade count survives too); orders it cannot explain
    are cancelled, positions it cannot explain get a conservative time exit
  * a stop order whose level is already crossed is sent as a market order only if price is
    still within `max_chase` of the level; otherwise the setup is skipped
  * before sending, the order's margin is checked against free margin
"""
import time

import numpy as np
import pandas as pd

from goldbot.models import Fill, Order, Position, Setup
from goldbot.state import StateStore

try:
    import MetaTrader5 as mt5
except ImportError:  # allows importing the module on machines without MT5 (tests, backtests)
    mt5 = None


class MT5Error(RuntimeError):
    pass


def apply_symbol_specs(risk_cfg, info, margin_per_lot: float | None = None, price: float | None = None) -> dict:
    """Use the broker's real contract specification for sizing (ounces per lot, lot min,
    step, max, effective leverage) instead of the config defaults. XAUUSD is often 100 oz
    per lot but some brokers use 10 or 1; sizing with the wrong value would multiply the risk.
    Returns what changed."""
    specs = {"contract_size": float(info.trade_contract_size), "volume_min": float(info.volume_min),
             "volume_step": float(info.volume_step)}
    if not all(v > 0 for v in specs.values()):
        raise MT5Error(f"invalid contract specification from the broker: {specs}")
    if margin_per_lot and price:
        specs["leverage"] = round(price * specs["contract_size"] / margin_per_lot, 2)
    changed = {k: (getattr(risk_cfg, k), v) for k, v in specs.items() if getattr(risk_cfg, k) != v}
    for k, v in specs.items():
        setattr(risk_cfg, k, v)
    vmax = float(info.volume_max)
    if vmax > 0 and vmax < risk_cfg.volume_max:
        changed["volume_max"] = (risk_cfg.volume_max, vmax)
        risk_cfg.volume_max = vmax
    return changed


class MT5Connection:
    def __init__(self, symbol: str, require_demo=True, path="", login=0, password="", server="",
                 log=print):
        if mt5 is None:
            raise MT5Error("MetaTrader5 package not installed (pip install MetaTrader5, Windows only)")
        self.base_symbol, self.require_demo = symbol, require_demo
        self.path, self.login, self.password, self.server = path, int(login or 0), password, server
        self.log = log
        self.offset_s = 0
        self._connect()

    def _connect(self):
        kw = {"timeout": 60000}
        if self.path:
            kw["path"] = self.path
        if self.login:
            kw.update(login=self.login, password=self.password, server=self.server)
        if not mt5.initialize(**kw):
            raise MT5Error(f"cannot connect to MetaTrader 5: {mt5.last_error()}")
        acc = mt5.account_info()
        if acc is None:
            raise MT5Error("no account logged in to MetaTrader 5")
        if self.login and acc.login != self.login:
            raise MT5Error(f"terminal is logged in to {acc.login}, expected {self.login}")
        self.is_demo = acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
        if self.require_demo and not self.is_demo:
            mt5.shutdown()
            raise MT5Error("real account detected and GB_REQUIRE_DEMO is on: stopping")
        self.symbol = self._resolve(self.base_symbol)
        self.info = mt5.symbol_info(self.symbol)
        self.refresh_offset()

    def healthy(self) -> tuple[bool, str]:
        ti = mt5.terminal_info()
        if ti is None:
            return False, f"terminal not reachable ({mt5.last_error()})"
        if not ti.connected:
            return False, "terminal has no connection to the trade server"
        if mt5.account_info() is None:
            return False, "no account logged in"
        return True, ""

    def trading_allowed(self) -> tuple[bool, str]:
        ti, acc = mt5.terminal_info(), mt5.account_info()
        if ti is not None and not ti.trade_allowed:
            return False, "Algo Trading is switched off in the MT5 terminal (toolbar button)"
        if acc is not None and not acc.trade_expert:
            return False, "automated trading is not allowed for this account"
        return True, ""

    def reconnect(self):
        try:
            mt5.shutdown()
        except Exception:
            pass
        self._connect()

    def _resolve(self, base):
        if mt5.symbol_info(base) is not None:
            mt5.symbol_select(base, True)
            return base
        for s in mt5.symbols_get() or []:
            if s.name.upper().startswith(base.upper()):
                mt5.symbol_select(s.name, True)
                return s.name
        raise MT5Error(f"symbol {base!r} not found at this broker")

    def margin_per_lot(self):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None, None
        m = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, self.symbol, 1.0, tick.ask)
        return (float(m) if m else None), float(tick.ask)

    def refresh_offset(self):
        """Server time offset from UTC, rounded to 30 minutes, from a fresh tick."""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is not None and abs(time.time() - (tick.time - self.offset_s)) < 3600 * 14:
            self.offset_s = int(round((tick.time - time.time()) / 1800.0) * 1800)
        return self.offset_s

    def to_utc(self, server_epoch) -> pd.Timestamp:
        return pd.Timestamp(int(server_epoch) - self.offset_s, unit="s")

    def to_server_epoch(self, ts_utc) -> int:
        return int(pd.Timestamp(ts_utc).value // 10**9) + self.offset_s

    def shutdown(self):
        mt5.shutdown()


class MT5Market:
    def __init__(self, conn: MT5Connection):
        self.c = conn

    def now(self):
        return pd.Timestamp.now("UTC").tz_localize(None).floor("min")

    def hilo(self, start, end):
        rates = mt5.copy_rates_range(self.c.symbol, mt5.TIMEFRAME_M1,
                                     self.c.to_server_epoch(start), self.c.to_server_epoch(end))
        if rates is None or len(rates) == 0:
            return float("nan"), float("nan"), 0
        t = rates["time"] - self.c.offset_s
        lo_e, hi_e = int(pd.Timestamp(start).value // 10**9), int(pd.Timestamp(end).value // 10**9)
        now_e = int(self.now().value // 10**9)
        m = (t >= lo_e) & (t < min(hi_e, now_e))         # closed bars only
        if not m.any():
            return float("nan"), float("nan"), 0
        return float(rates["high"][m].max()), float(rates["low"][m].min()), int(m.sum())

    def daily(self, day):
        rates = mt5.copy_rates_from_pos(self.c.symbol, mt5.TIMEFRAME_D1, 0, 80)
        if rates is None or len(rates) < 25:
            return {"trend": 0, "atr": float("nan"), "prev_close": None}
        d = pd.DataFrame(rates)
        d["day"] = pd.to_datetime(d["time"], unit="s").dt.date     # server-day bars
        d = d[d["day"] < day]                                     # completed days only
        c = d["close"]
        tr = pd.concat([d["high"] - d["low"], (d["high"] - c.shift()).abs(),
                        (d["low"] - c.shift()).abs()], axis=1).max(axis=1)
        ema = c.ewm(span=20, adjust=False).mean()
        return {"trend": int(np.sign(c.iloc[-1] - ema.iloc[-1])),
                "atr": float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]),
                "prev_close": round(float(c.iloc[-1]), 2)}

    def spread(self):
        tick = mt5.symbol_info_tick(self.c.symbol)
        return float(tick.ask - tick.bid) if tick else float("inf")

    def last_price(self):
        tick = mt5.symbol_info_tick(self.c.symbol)
        return float(tick.bid) if tick else float("nan")


def _fallback_setup(group, direction, entry, sl, tp, t, hold_hours):
    return Setup(strategy="adopted", group=group or f"adopted-{t:%Y%m%d%H%M}", direction=direction,
                 entry=entry, sl=sl, tp=tp or None, created=t, expires=t,
                 exit_by=t + pd.Timedelta(hours=hold_hours))


class MT5Broker:
    def __init__(self, conn: MT5Connection, magic: int, count_all_account_pnl=True,
                 deviation_points=30, max_chase=0.25, state: StateStore | None = None,
                 log=print, adopt_hold_hours=4.0, history_days=4):
        self.c = conn
        self.magic = magic
        self.count_all = count_all_account_pnl
        self.deviation = deviation_points
        self.max_chase = max_chase          # fraction of the stop distance
        self.state = state or StateStore(None)
        self.log = log
        self.adopt_hold_hours = adopt_hold_hours
        self._orders: dict[int, Order] = {}
        self._positions: dict[int, Position] = {}
        self._fills: list[Fill] = []
        self.last_error = ""
        self._rebuild_history(history_days)
        self._adopt_leftovers()

    # ---- start-up recovery
    def _rebuild_history(self, days):
        """Closed bot trades of the last days, from the deal history (survives restarts)."""
        frm = self.c.to_server_epoch(pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=days))
        to = self.c.to_server_epoch(pd.Timestamp.now("UTC").tz_localize(None) + pd.Timedelta(days=1))
        deals = [d for d in (mt5.history_deals_get(frm, to) or []) if d.magic == self.magic
                 and d.symbol == self.c.symbol]
        by_pos = {}
        for d in deals:
            by_pos.setdefault(d.position_id, []).append(d)
        live = {p.ticket for p in (mt5.positions_get(symbol=self.c.symbol) or [])}
        for pid, ds in by_pos.items():
            ins = [d for d in ds if d.entry == mt5.DEAL_ENTRY_IN]
            outs = [d for d in ds if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT)]
            if not ins or not outs or pid in live:
                continue
            first, last = min(ins, key=lambda d: d.time), max(outs, key=lambda d: d.time)
            saved = self.state.get(first.order) or self.state.get(pid)
            direction = 1 if first.type == mt5.DEAL_TYPE_BUY else -1
            t_in = self.c.to_utc(first.time)
            setup = saved[0] if saved else _fallback_setup(first.comment, direction, first.price,
                                                            first.price, None, t_in, self.adopt_hold_hours)
            risk = saved[2] if saved else 0.0
            pnl = sum(d.profit + d.commission + d.swap + d.fee for d in ds)
            reason = {mt5.DEAL_REASON_SL: "sl", mt5.DEAL_REASON_TP: "tp"}.get(last.reason, "other")
            self._fills.append(Fill(position_id=str(pid), setup=setup, lots=first.volume,
                                    entry_price=first.price, entry_time=t_in, exit_price=last.price,
                                    exit_time=self.c.to_utc(last.time), reason=reason, pnl=pnl,
                                    planned_risk=risk))
        self._fills.sort(key=lambda f: f.exit_time)

    def _adopt_leftovers(self):
        now = pd.Timestamp.now("UTC").tz_localize(None)
        for o in mt5.orders_get(symbol=self.c.symbol) or []:
            if o.magic != self.magic:
                continue
            saved = self.state.get(o.ticket)
            if saved and saved[0].expires > now:
                self._orders[o.ticket] = Order(id=str(o.ticket), setup=saved[0], lots=saved[1],
                                               planned_risk=saved[2])
                self.log(f"adopted pending order {o.ticket} ({saved[0].group})")
            else:
                mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
                self.state.forget(o.ticket)
                self.log(f"cancelled leftover pending order {o.ticket} (no saved context)")
        for p in mt5.positions_get(symbol=self.c.symbol) or []:
            if p.magic != self.magic:
                continue
            saved = self.state.get(p.identifier) or self.state.get(p.ticket)
            d = 1 if p.type == mt5.POSITION_TYPE_BUY else -1
            t = self.c.to_utc(p.time)
            if saved:
                setup, risk = saved[0], saved[2]
            else:
                setup = _fallback_setup(p.comment, d, p.price_open, p.sl, p.tp, t, self.adopt_hold_hours)
                risk = abs(p.price_open - p.sl) * p.volume * self.c.info.trade_contract_size if p.sl else 0.0
                self.log(f"position {p.ticket} has no saved context: time exit set to {setup.exit_by} UTC")
            self._positions[p.ticket] = Position(id=str(p.ticket), setup=setup, lots=p.volume,
                                                 entry_price=p.price_open, entry_time=t, planned_risk=risk)
            self.log(f"adopted open position {p.ticket} ({setup.group})")
        keep = set(self._orders) | set(self._positions) | {p.identifier for p in (mt5.positions_get(symbol=self.c.symbol) or [])}
        self.state.prune(keep)

    # ---- helpers
    def _send(self, req):
        modes = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
        res = None
        for m in modes:
            req["type_filling"] = m
            res = mt5.order_send(req)
            if res is not None and res.retcode != 10030:      # 10030: unsupported filling mode
                break
        if res is None:
            self.last_error = f"order_send returned nothing: {mt5.last_error()}"
        elif res.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            self.last_error = f"retcode {res.retcode}: {res.comment}"
        return res

    # ---- interface
    def account(self):
        a = mt5.account_info()
        if a is None:
            raise MT5Error("account_info unavailable (terminal disconnected?)")
        return {"equity": a.equity, "balance": a.balance, "is_demo": self.c.is_demo,
                "margin_free": a.margin_free}

    def place(self, setup: Setup, lots: float, planned_risk: float):
        info, tick = self.c.info, mt5.symbol_info_tick(self.c.symbol)
        if tick is None:
            self.last_error = "no tick"
            return None
        digits, point = info.digits, info.point
        d = setup.direction
        a = mt5.account_info()
        margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY if d > 0 else mt5.ORDER_TYPE_SELL,
                                       self.c.symbol, lots, setup.entry)
        if a is not None and margin and margin > a.margin_free * 0.9:
            self.last_error = f"not enough free margin ({margin:.2f} needed, {a.margin_free:.2f} free)"
            return None
        min_dist = (info.trade_stops_level + 2) * point
        crossed = (tick.ask >= setup.entry) if d > 0 else (tick.bid <= setup.entry)
        base = {"symbol": self.c.symbol, "volume": lots, "sl": round(setup.sl, digits),
                "tp": 0.0 if setup.tp is None else round(setup.tp, digits),
                "magic": self.magic, "comment": setup.group[-31:], "deviation": self.deviation}
        if crossed:
            px = tick.ask if d > 0 else tick.bid
            if abs(px - setup.entry) > self.max_chase * setup.risk_per_unit:
                self.last_error = "price already too far past the entry level"
                return None
            req = {**base, "action": mt5.TRADE_ACTION_DEAL, "price": px,
                   "type": mt5.ORDER_TYPE_BUY if d > 0 else mt5.ORDER_TYPE_SELL,
                   "type_time": mt5.ORDER_TIME_GTC}
        else:
            if abs(setup.entry - (tick.ask if d > 0 else tick.bid)) < min_dist:
                self.last_error = "entry too close to price for a stop order"
                return None
            req = {**base, "action": mt5.TRADE_ACTION_PENDING, "price": round(setup.entry, digits),
                   "type": mt5.ORDER_TYPE_BUY_STOP if d > 0 else mt5.ORDER_TYPE_SELL_STOP,
                   "type_time": mt5.ORDER_TIME_SPECIFIED,
                   "expiration": self.c.to_server_epoch(setup.expires)}
        res = self._send(req)
        if res is None or res.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            return None
        o = Order(id=str(res.order), setup=setup, lots=lots, planned_risk=planned_risk)
        self._orders[res.order] = o
        self.state.remember(res.order, setup, lots, planned_risk)
        return o

    def cancel(self, order):
        ticket = int(order.id)
        res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})
        self._orders.pop(ticket, None)
        self.state.forget(ticket)
        return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE

    def pending(self):
        live = {o.ticket for o in (mt5.orders_get(symbol=self.c.symbol) or []) if o.magic == self.magic}
        for t in list(self._orders):
            # gone from the server: either filled (becomes a position) or expired/removed
            if t not in live and not self._became_position(t):
                self._orders.pop(t, None)
                self.state.forget(t)
        return [o for t, o in self._orders.items() if t in live]

    def _became_position(self, ticket):
        for p in mt5.positions_get(symbol=self.c.symbol) or []:
            if p.identifier == ticket and p.magic == self.magic:
                o = self._orders.pop(ticket)
                self._positions[p.ticket] = Position(
                    id=str(p.ticket), setup=o.setup, lots=p.volume, entry_price=p.price_open,
                    entry_time=self.c.to_utc(p.time), planned_risk=o.planned_risk)
                if p.ticket != ticket:
                    self.state.remember(p.ticket, o.setup, o.lots, o.planned_risk)
                return True
        return False

    def positions(self):
        self.pending()                              # converts filled orders into positions
        live = {p.ticket: p for p in (mt5.positions_get(symbol=self.c.symbol) or []) if p.magic == self.magic}
        for t in list(self._positions):
            if t not in live:                        # closed by SL / TP on the server
                pos = self._positions.pop(t)
                self._record_close(pos)
        return list(self._positions.values())

    def _record_close(self, pos: Position):
        deals = mt5.history_deals_get(position=int(pos.id)) or []
        out = [d for d in deals if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT)]
        if not out:
            return
        pnl = sum(d.profit + d.commission + d.swap + d.fee for d in deals)
        last = max(out, key=lambda d: d.time)
        reason = {mt5.DEAL_REASON_SL: "sl", mt5.DEAL_REASON_TP: "tp"}.get(last.reason, "other")
        self._fills.append(Fill(position_id=pos.id, setup=pos.setup, lots=pos.lots,
                                entry_price=pos.entry_price, entry_time=pos.entry_time,
                                exit_price=last.price, exit_time=self.c.to_utc(last.time),
                                reason=reason, pnl=pnl, planned_risk=pos.planned_risk))
        self.state.forget(int(pos.id))

    def close(self, position, reason):
        p = next((x for x in (mt5.positions_get(ticket=int(position.id)) or [])), None)
        if p is None:
            return None
        tick = mt5.symbol_info_tick(self.c.symbol)
        buy = p.type == mt5.POSITION_TYPE_BUY
        res = self._send({"action": mt5.TRADE_ACTION_DEAL, "symbol": self.c.symbol,
                          "volume": p.volume, "position": p.ticket,
                          "type": mt5.ORDER_TYPE_SELL if buy else mt5.ORDER_TYPE_BUY,
                          "price": tick.bid if buy else tick.ask, "deviation": self.deviation,
                          "magic": self.magic, "comment": reason, "type_time": mt5.ORDER_TIME_GTC})
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            return None
        pos = self._positions.pop(p.ticket, position)
        self._record_close(pos)
        return self._fills[-1] if self._fills else None

    def fills_since(self, start):
        return [f for f in self._fills if f.exit_time >= pd.Timestamp(start)]

    def realized_since(self, start):
        frm = self.c.to_server_epoch(start)
        to = self.c.to_server_epoch(pd.Timestamp.now("UTC").tz_localize(None) + pd.Timedelta(days=1))
        deals = mt5.history_deals_get(frm, to)
        if deals is None:
            raise MT5Error(f"deal history unavailable: {mt5.last_error()}")
        out = []
        for d in deals:
            if d.type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL):
                continue                         # skip balance, credit, etc.
            if not self.count_all and d.magic != self.magic:
                continue
            pnl = d.profit + d.commission + d.swap + d.fee
            if pnl:
                out.append((self.c.to_utc(d.time), pnl))
        return out

    def external_risk(self):
        """Risk to stop of positions the bot did not open (manual trades)."""
        if not self.count_all:
            return 0.0, False
        risk, naked = 0.0, False
        for p in mt5.positions_get() or []:
            if p.magic == self.magic:
                continue
            if p.sl == 0:
                naked = True
                continue
            info = mt5.symbol_info(p.symbol)
            buy = p.type == mt5.POSITION_TYPE_BUY
            loss = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL,
                                         p.symbol, p.volume, p.price_current, p.sl)
            if loss is None and info is not None:
                loss = -abs(p.price_current - p.sl) / info.trade_tick_size * info.trade_tick_value * p.volume
            risk += max(0.0, -(loss or 0.0)) - min(0.0, p.profit)   # stop risk plus open loss
        return risk, naked
