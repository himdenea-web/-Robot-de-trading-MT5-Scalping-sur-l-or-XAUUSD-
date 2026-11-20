"""A small in-memory stand-in for the MetaTrader5 package, enough to test the broker's
order flow, restart recovery and reconnect logic without a terminal."""
import time
from types import SimpleNamespace as NS

ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_REAL = 0, 2
ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
TRADE_ACTION_DEAL, TRADE_ACTION_PENDING, TRADE_ACTION_REMOVE = 1, 5, 8
ORDER_TYPE_BUY, ORDER_TYPE_SELL, ORDER_TYPE_BUY_STOP, ORDER_TYPE_SELL_STOP = 0, 1, 4, 5
ORDER_TIME_GTC, ORDER_TIME_SPECIFIED = 0, 2
TRADE_RETCODE_PLACED, TRADE_RETCODE_DONE = 10008, 10009
POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
DEAL_TYPE_BUY, DEAL_TYPE_SELL, DEAL_TYPE_BALANCE = 0, 1, 2
DEAL_ENTRY_IN, DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT = 0, 1, 2
DEAL_REASON_CLIENT, DEAL_REASON_SL, DEAL_REASON_TP = 0, 4, 5
TIMEFRAME_M1, TIMEFRAME_D1 = 1, 16408

S = NS()


def reset(bid=2000.0, spread=0.2, server_offset_h=2, contract=100.0, connected=True):
    S.bid, S.spread, S.offset = bid, spread, server_offset_h * 3600
    S.contract, S.connected, S.initialized, S.trade_allowed = contract, connected, False, True
    S.balance, S.next_ticket = 5000.0, 1000
    S.orders, S.positions, S.deals = {}, {}, []
    S.init_kwargs = None


reset()


def _now():
    return int(time.time()) + S.offset


def _ticket():
    S.next_ticket += 1
    return S.next_ticket


def initialize(**kw):
    S.init_kwargs, S.initialized = kw, True
    return True


def shutdown():
    S.initialized = False


def last_error():
    return (1, "fake")


def terminal_info():
    if not S.initialized:
        return None
    return NS(connected=S.connected, trade_allowed=S.trade_allowed)


def account_info():
    if not S.initialized:
        return None
    eq = S.balance + sum(_floating(p) for p in S.positions.values())
    return NS(login=123, trade_mode=ACCOUNT_TRADE_MODE_DEMO, equity=eq, balance=S.balance,
              margin_free=eq * 0.9, trade_expert=True)


def symbol_info(sym):
    if sym != "XAUUSD":
        return None
    return NS(name=sym, trade_contract_size=S.contract, volume_min=0.01, volume_step=0.01,
              volume_max=100.0, digits=2, point=0.01, trade_stops_level=0, trade_tick_size=0.01,
              trade_tick_value=S.contract * 0.01)


def symbol_select(sym, on):
    return True


def symbols_get():
    return [NS(name="XAUUSD")]


def symbol_info_tick(sym):
    return NS(bid=S.bid, ask=S.bid + S.spread, time=_now())


def order_calc_margin(kind, sym, vol, price):
    return vol * S.contract * price / 25.0


def order_calc_profit(kind, sym, vol, p0, p1):
    d = 1 if kind == ORDER_TYPE_BUY else -1
    return (p1 - p0) * d * vol * S.contract


def _floating(p):
    d = 1 if p.type == POSITION_TYPE_BUY else -1
    px = S.bid if d > 0 else S.bid + S.spread
    return (px - p.price_open) * d * p.volume * S.contract


def order_send(req):
    a = req["action"]
    if a == TRADE_ACTION_REMOVE:
        S.orders.pop(req["order"], None)
        return NS(retcode=TRADE_RETCODE_DONE, order=req["order"], comment="")
    if a == TRADE_ACTION_PENDING:
        t = _ticket()
        S.orders[t] = NS(ticket=t, magic=req["magic"], type=req["type"], price_open=req["price"],
                         sl=req["sl"], tp=req["tp"], volume_initial=req["volume"], comment=req["comment"])
        return NS(retcode=TRADE_RETCODE_PLACED, order=t, comment="")
    if a == TRADE_ACTION_DEAL and "position" in req:
        p = S.positions[req["position"]]
        _close(p, req["price"], DEAL_REASON_CLIENT)
        return NS(retcode=TRADE_RETCODE_DONE, order=_ticket(), comment="")
    if a == TRADE_ACTION_DEAL:
        t = _ticket()
        _open(t, req["type"] == ORDER_TYPE_BUY, req["price"], req)
        return NS(retcode=TRADE_RETCODE_DONE, order=t, comment="")
    raise ValueError(a)


def _open(order_ticket, buy, price, req):
    p = NS(ticket=order_ticket, identifier=order_ticket, magic=req["magic"],
           type=POSITION_TYPE_BUY if buy else POSITION_TYPE_SELL, price_open=price,
           sl=req["sl"], tp=req["tp"], volume=req.get("volume", req.get("volume_initial")),
           time=_now(), comment=req["comment"], symbol="XAUUSD", price_current=price, profit=0.0)
    S.positions[order_ticket] = p
    S.deals.append(NS(ticket=_ticket(), order=order_ticket, position_id=order_ticket, magic=p.magic,
                      symbol="XAUUSD", type=DEAL_TYPE_BUY if buy else DEAL_TYPE_SELL, entry=DEAL_ENTRY_IN,
                      price=price, volume=p.volume, time=p.time, profit=0.0, commission=-1.0, swap=0.0,
                      fee=0.0, comment=p.comment, reason=DEAL_REASON_CLIENT))


def _close(p, price, reason):
    d = 1 if p.type == POSITION_TYPE_BUY else -1
    pnl = (price - p.price_open) * d * p.volume * S.contract
    S.balance += pnl - 2.0
    S.positions.pop(p.ticket)
    S.deals.append(NS(ticket=_ticket(), order=_ticket(), position_id=p.ticket, magic=p.magic,
                      symbol="XAUUSD", type=DEAL_TYPE_SELL if d > 0 else DEAL_TYPE_BUY, entry=DEAL_ENTRY_OUT,
                      price=price, volume=p.volume, time=_now(), profit=pnl, commission=-1.0, swap=0.0,
                      fee=0.0, comment="", reason=reason))


# --- test helpers: market events
def trigger_order(ticket):
    o = S.orders.pop(ticket)
    buy = o.type == ORDER_TYPE_BUY_STOP
    _open(ticket, buy, o.price_open, {"magic": o.magic, "sl": o.sl, "tp": o.tp,
                                      "volume": o.volume_initial, "comment": o.comment})


def hit_stop(ticket):
    p = S.positions[ticket]
    _close(p, p.sl, DEAL_REASON_SL)


def orders_get(symbol=None):
    return list(S.orders.values())


def positions_get(symbol=None, ticket=None):
    ps = list(S.positions.values())
    return [p for p in ps if ticket is None or p.ticket == ticket]


def history_deals_get(*args, position=None):
    if position is not None:
        return [d for d in S.deals if d.position_id == position]
    frm, to = args
    return [d for d in S.deals if frm <= d.time <= to]


def copy_rates_range(sym, tf, frm, to):
    return None


def copy_rates_from_pos(sym, tf, start, n):
    return None
