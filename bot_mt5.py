
import os
import csv
import math
import time
import calendar
import datetime as dt
import traceback

import MetaTrader5 as mt5
import pandas as pd
import requests



def load_env(path=".env"):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()


SYMBOL = os.getenv("SYMBOL", "XAUUSD")   
TIMEFRAME = mt5.TIMEFRAME_M5
MAGIC_NUMBER = 123456
DEVIATION = 30                            


EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
RSI_BUY_LEVEL = 45        
RSI_SELL_LEVEL = 55       
MIN_TREND_ATR = 0.2       


SL_ATR = 1.0   
TP_ATR = 1.5   


RISK_PERCENT = 0.5              
MAX_POSITIONS = 2               
MAX_TRADES_PER_DAY = 2          
MAX_DAILY_LOSS_MONEY = 180.0    
WEEKLY_PROFIT_TARGET = 600.0    
MAX_TOTAL_RISK_PERCENT = 2.0    
CLOSE_ALL_ON_LIMIT = True      
MAX_SPREAD_POINTS = 50          

FORCE_MIN_LOT = True           
MAX_OVERRISK_FACTOR = 3.0       
MAX_LOTS = 2.0                  

# --- Sécurité
REQUIRE_DEMO = True             # refuse de tourner sur un compte réel
TEST_TRADE_ON_START = False     

# --- Fonctionnement
CHECK_SECONDS = 5               
STALE_TICK_SECONDS = 120        
IDLE_PRINT_SECONDS = 300        


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

USE_GEMINI = True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FAIL_OPEN = True         

CSV_FILE = "journal_de_trading.csv"

STATE = {
    "tick_time": None,
    "tick_change": time.monotonic(),
    "last_bar": None,
    "first_pass": True,
    "last_idle": time.monotonic(),
    "halted_today": None,
}



def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception as e:
        log(f"[ERREUR] Telegram : {e}")


def log_trade_csv(symbol, o_type, price, sl, tp, volume, risk, ticket):
    exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["Date", "Symbole", "Type", "Prix", "SL", "TP", "Volume", "Risque", "Ticket"])
        w.writerow([dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, o_type,
                    price, sl, tp, volume, round(risk, 2), ticket])


def resolve_symbol(base):
    """Trouve le vrai nom du symbole chez le broker (XAUUSD, XAUUSDm, XAUUSD.a ...)."""
    if mt5.symbol_info(base) is not None:
        mt5.symbol_select(base, True)
        return base
    for s in mt5.symbols_get() or []:
        if s.name.upper().startswith(base.upper()):
            mt5.symbol_select(s.name, True)
            return s.name
    return None



def compute_indicators(df):
    c = df["close"]
    df["ema_fast"] = c.ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = c.ewm(span=EMA_SLOW, adjust=False).mean()

    d = c.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean() / dn.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + rs)

    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    return df


def get_signal(last, prev):
    """Retourne (signal, explication). Travaille uniquement sur des bougies CLÔTURÉES."""
    strength = abs(last.ema_fast - last.ema_slow) / last.atr if last.atr > 0 else 0
    if strength < MIN_TREND_ATR:
        return None, f"tendance trop faible ({strength:.2f} ATR < {MIN_TREND_ATR})"

    if last.ema_fast > last.ema_slow and last.close > last.ema_slow:
        if prev.rsi < RSI_BUY_LEVEL <= last.rsi:
            return "BUY", "repli terminé dans une tendance haussière"
        return None, f"tendance haussière, j'attends le RSI qui remonte au-dessus de {RSI_BUY_LEVEL} (RSI={last.rsi:.1f})"

    if last.ema_fast < last.ema_slow and last.close < last.ema_slow:
        if prev.rsi > RSI_SELL_LEVEL >= last.rsi:
            return "SELL", "rebond terminé dans une tendance baissière"
        return None, f"tendance baissière, j'attends le RSI qui redescend sous {RSI_SELL_LEVEL} (RSI={last.rsi:.1f})"

    return None, "pas de tendance claire (prix entre les deux EMA)"



def my_positions():
    pos = mt5.positions_get(symbol=SYMBOL)
    return [p for p in (pos or []) if p.magic == MAGIC_NUMBER]


def loss_per_lot(info, price_distance):
    """Perte réelle (devise du compte) pour 1 lot si le prix bouge de price_distance.
    On demande le calcul à MetaTrader lui-même (order_calc_profit), plus fiable que tick_value."""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is not None and price_distance > 0:
        price = tick.ask
        p = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, SYMBOL, 1.0, price, price - price_distance)
        if p:
            return abs(p)
    # plan B si MT5 ne répond pas
    return price_distance / info.trade_tick_size * info.trade_tick_value


def position_risk(p, info):
    if p.sl == 0:
        return 0.0
    return loss_per_lot(info, abs(p.price_open - p.sl)) * p.volume


def calc_lots(info, sl_dist, risk_money):
    lpl = loss_per_lot(info, sl_dist)
    if lpl <= 0:
        return None, "perte par lot invalide"
    step = info.volume_step
    decimals = max(0, -int(math.floor(math.log10(step))))
    lots = round(math.floor(risk_money / lpl / step) * step, decimals)
    if lots < info.volume_min:
        risk_min = info.volume_min * lpl
        if FORCE_MIN_LOT and risk_min <= risk_money * MAX_OVERRISK_FACTOR:
            lots = info.volume_min
        else:
            return None, (f"capital trop petit : le lot minimum {info.volume_min} risquerait "
                          f"{risk_min:.2f} alors que le risque voulu est {risk_money:.2f}")
    lots = min(lots, info.volume_max, MAX_LOTS)
    return lots, None


def server_now():
    """Heure du dernier tick (datetime naïf en UTC)."""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return None
    return dt.datetime.fromtimestamp(tick.time, dt.timezone.utc).replace(tzinfo=None)


def day_start():
    now = server_now()
    if now is None:
        return None
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def week_start():
    d = day_start()
    if d is None:
        return None
    return d - dt.timedelta(days=d.weekday())   # lundi 00:00


def bot_deals_since(start):
    """Deals du robot (magic number) depuis 'start' (datetime naïf UTC)."""
    now = server_now()
    if now is None or start is None:
        return []
    start_epoch = calendar.timegm(start.timetuple())
    deals = mt5.history_deals_get(start - dt.timedelta(days=1), now + dt.timedelta(days=1))
    return [d for d in (deals or [])
            if d.magic == MAGIC_NUMBER and d.time >= start_epoch
            and d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL)]


def trades_opened_today():
    return sum(1 for d in bot_deals_since(day_start()) if d.entry == mt5.DEAL_ENTRY_IN)


def realized_pnl(start):
    return sum(d.profit + d.commission + d.swap + d.fee for d in bot_deals_since(start))


def daily_loss_money():
    """Perte du jour (positive) = réalisé + flottant des positions du robot."""
    floating = sum(p.profit + p.swap for p in my_positions())
    pnl = realized_pnl(day_start()) + floating
    return max(0.0, -pnl)


def loss_budget_left():
    """Perte encore autorisée aujourd'hui, en comptant le pire cas :
    pertes déjà réalisées + risque restant des positions ouvertes (SL touchés)."""
    info = mt5.symbol_info(SYMBOL)
    realized_loss = max(0.0, -realized_pnl(day_start()))
    open_risk = sum(position_risk(p, info) for p in my_positions())
    return MAX_DAILY_LOSS_MONEY - realized_loss - open_risk


# =====================================================================
# 5. ENVOI DES ORDRES
# =====================================================================
def send_order(request, info):
    """Essaie les modes de remplissage supportés (corrige l'erreur 10030)."""
    fm = info.filling_mode        # bits : 1 = FOK, 2 = IOC
    modes = []
    if fm & 2:
        modes.append(mt5.ORDER_FILLING_IOC)
    if fm & 1:
        modes.append(mt5.ORDER_FILLING_FOK)
    modes.append(mt5.ORDER_FILLING_RETURN)
    for m in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
        if m not in modes:
            modes.append(m)

    result = None
    for m in modes:
        request["type_filling"] = m
        result = mt5.order_send(request)
        if result is None:
            log(f"[ERREUR] order_send a renvoyé None : {mt5.last_error()}")
            continue
        if result.retcode == 10030:   # mode de remplissage non supporté -> on essaie le suivant
            continue
        return result
    return result


def place_order(direction, atr, info, lots=None):
    """Ouvre une position. Retourne (ok, message)."""
    tick = mt5.symbol_info_tick(SYMBOL)
    acc = mt5.account_info()
    min_dist = (info.trade_stops_level + 2) * info.point
    sl_dist = max(SL_ATR * atr, min_dist)
    tp_dist = max(TP_ATR * atr, min_dist)

    if direction == "BUY":
        otype, price = mt5.ORDER_TYPE_BUY, tick.ask
        sl, tp = price - sl_dist, price + tp_dist
    else:
        otype, price = mt5.ORDER_TYPE_SELL, tick.bid
        sl, tp = price + sl_dist, price - tp_dist
    sl, tp = round(sl, info.digits), round(tp, info.digits)

    budget = loss_budget_left()
    if budget <= 0:
        return False, f"budget de perte journalier épuisé (max {MAX_DAILY_LOSS_MONEY})"
    trades_left = max(1, MAX_TRADES_PER_DAY - trades_opened_today())
    if lots is None:
        # le risque d'un trade ne dépasse jamais sa part du budget restant
        risk_money = min(acc.equity * RISK_PERCENT / 100, budget / trades_left)
        lots, err = calc_lots(info, sl_dist, risk_money)
        if lots is None:
            return False, err
    risk = lots * loss_per_lot(info, sl_dist)
    if risk > budget:   # sécurité finale (ex. lot minimum forcé)
        return False, f"risque {risk:.2f} > budget de perte restant {budget:.2f}"
    log(f"   lot {lots} | perte estimée si SL touché : {risk:.2f} | budget restant : {budget:.2f}")

    open_risk = sum(position_risk(p, info) for p in my_positions())
    if open_risk + risk > acc.equity * MAX_TOTAL_RISK_PERCENT / 100:
        return False, (f"risque total dépassé ({open_risk + risk:.2f} > "
                       f"{acc.equity * MAX_TOTAL_RISK_PERCENT / 100:.2f})")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lots,
        "type": otype,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": "bot_mt5",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    result = send_order(request, info)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = None if result is None else result.retcode
        comment = "" if result is None else result.comment
        return False, f"ordre refusé (retcode={code} : {comment})"

    log(f"✅ {direction} {lots} lot | prix {result.price} | SL {sl} | TP {tp} | risque ~{risk:.2f}")
    log_trade_csv(SYMBOL, direction, result.price, sl, tp, lots, risk, result.order)
    send_telegram(f"{direction} {SYMBOL}\nPrix : {result.price}\nSL : {sl}\nTP : {tp}\nLots : {lots}\nRisque : ~{risk:.2f}")
    return True, "ok"


def close_all_positions(info):
    for p in my_positions():
        tick = mt5.symbol_info_tick(SYMBOL)
        if p.type == mt5.POSITION_TYPE_BUY:
            otype, price = mt5.ORDER_TYPE_SELL, tick.bid
        else:
            otype, price = mt5.ORDER_TYPE_BUY, tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": p.volume,
            "type": otype, "position": p.ticket, "price": price, "deviation": DEVIATION,
            "magic": MAGIC_NUMBER, "comment": "limite perte jour", "type_time": mt5.ORDER_TIME_GTC,
        }
        res = send_order(req, info)
        ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
        log(f"Fermeture position {p.ticket} : {'OK' if ok else 'ÉCHEC'}")


# =====================================================================
# 6. FILTRE GEMINI (optionnel)
# =====================================================================
def gemini_ok(signal, last):
    if not USE_GEMINI:
        return True, "Gemini désactivé"
    if not GEMINI_API_KEY:
        return GEMINI_FAIL_OPEN, "Gemini activé mais clé absente"
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = (f"Scalping {SYMBOL}. Signal proposé : {signal}. Prix {last.close:.2f}, "
                  f"EMA20 {last.ema_fast:.2f}, EMA50 {last.ema_slow:.2f}, RSI {last.rsi:.1f}, "
                  f"ATR {last.atr:.2f}. Réponds uniquement par OUI ou NON : "
                  f"ce trade est-il raisonnable ?")
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        txt = (resp.text or "").strip().upper()
        return txt.startswith("OUI"), f"Gemini : {txt[:40]}"
    except Exception as e:
        return GEMINI_FAIL_OPEN, f"Gemini erreur ({e})"


# =====================================================================
# 7. LOGIQUE PAR BOUGIE
# =====================================================================
def market_is_open():
    return (time.monotonic() - STATE["tick_change"]) < STALE_TICK_SECONDS


def update_tick_state():
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is not None and tick.time != STATE["tick_time"]:
        STATE["tick_time"] = tick.time
        STATE["tick_change"] = time.monotonic()


def on_new_bar(closed, info):
    last, prev = closed.iloc[-1], closed.iloc[-2]
    tick = mt5.symbol_info_tick(SYMBOL)
    spread = (tick.ask - tick.bid) / info.point
    signal, reason = get_signal(last, prev)
    positions = my_positions()

    log(f"Bougie {last['time']} | close {last.close:.2f} | EMA {last.ema_fast:.2f}/{last.ema_slow:.2f} | "
        f"RSI {last.rsi:.1f} | ATR {last.atr:.2f} | spread {spread:.0f} pts | positions {len(positions)}")
    log(f"   -> {reason}")

    if STATE["first_pass"]:
        STATE["first_pass"] = False
        log("   (analyse initiale : aucun trade sur cette bougie)")
        return
    if signal is None:
        return

    # ---- Portes de sécurité
    today = dt.date.today()
    loss = daily_loss_money()
    if STATE["halted_today"] == today or loss >= MAX_DAILY_LOSS_MONEY:
        if STATE["halted_today"] != today:
            STATE["halted_today"] = today
            msg = (f"⛔ Limite de perte journalière atteinte ({loss:.2f} / {MAX_DAILY_LOSS_MONEY}). "
                   f"Robot en pause jusqu'à demain.")
            log(msg)
            send_telegram(msg)
            if CLOSE_ALL_ON_LIMIT:
                close_all_positions(info)
        else:
            log("   ⛔ Signal ignoré : limite de perte journalière atteinte")
        return

    week_pnl = realized_pnl(week_start())
    if week_pnl >= WEEKLY_PROFIT_TARGET:
        log(f"   🎯 Objectif de la semaine atteint ({week_pnl:.2f} >= {WEEKLY_PROFIT_TARGET}) : "
            f"plus de trade jusqu'à lundi")
        return

    if trades_opened_today() >= MAX_TRADES_PER_DAY:
        log(f"   ⛔ Signal {signal} ignoré : {MAX_TRADES_PER_DAY} trades déjà pris aujourd'hui")
        return

    if not market_is_open():
        log("   ⛔ Signal ignoré : marché fermé (aucun nouveau prix)")
        return
    if spread > MAX_SPREAD_POINTS:
        log(f"   ⛔ Signal {signal} ignoré : spread trop élevé ({spread:.0f} > {MAX_SPREAD_POINTS})")
        return
    if len(positions) >= MAX_POSITIONS:
        log(f"   ⛔ Signal {signal} ignoré : {MAX_POSITIONS} positions déjà ouvertes")
        return
    wanted = mt5.POSITION_TYPE_BUY if signal == "BUY" else mt5.POSITION_TYPE_SELL
    if any(p.type != wanted for p in positions):
        log(f"   ⛔ Signal {signal} ignoré : une position dans l'autre sens est ouverte")
        return

    ok_ai, ai_msg = gemini_ok(signal, last)
    log(f"   {ai_msg}")
    if not ok_ai:
        log(f"   ⛔ Signal {signal} refusé par Gemini")
        return

    ok, msg = place_order(signal, last.atr, info)
    if not ok:
        log(f"   ⛔ Signal {signal} non exécuté : {msg}")


# =====================================================================
# 8. PROGRAMME PRINCIPAL
# =====================================================================
def main():
    global SYMBOL
    if not mt5.initialize():
        log(f"[ERREUR] Impossible de se connecter à MT5 : {mt5.last_error()}")
        log("Ouvre MetaTrader 5, connecte-toi à ton compte, puis relance.")
        return

    acc = mt5.account_info()
    term = mt5.terminal_info()
    if acc is None:
        log("[ERREUR] Aucun compte connecté dans MT5.")
        mt5.shutdown()
        return

    is_demo = acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
    log(f"Compte {acc.login} | {'DÉMO' if is_demo else 'RÉEL'} | solde {acc.balance} {acc.currency} | {acc.server}")
    if REQUIRE_DEMO and not is_demo:
        log("⛔ Compte RÉEL détecté : arrêt. (REQUIRE_DEMO = True)")
        mt5.shutdown()
        return
    if not term.trade_allowed:
        log("⚠️ Le bouton 'Algo Trading' de MT5 est désactivé : clique dessus (il doit être vert).")
    if not acc.trade_allowed:
        log("⚠️ Le trading est désactivé sur ce compte.")

    real_symbol = resolve_symbol(SYMBOL)
    if real_symbol is None:
        log(f"[ERREUR] Symbole '{SYMBOL}' introuvable chez ton broker. Regarde le nom exact dans MT5 (Market Watch).")
        mt5.shutdown()
        return
    SYMBOL = real_symbol
    info = mt5.symbol_info(SYMBOL)
    log(f"Symbole : {SYMBOL} | lot min {info.volume_min} | pas {info.volume_step} | digits {info.digits}")
    log(f"Limites : perte max {MAX_DAILY_LOSS_MONEY}/jour | {MAX_TRADES_PER_DAY} trades/jour | "
        f"objectif {WEEKLY_PROFIT_TARGET}/semaine")
    if info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
        log("⚠️ Le trading n'est pas pleinement autorisé sur ce symbole en ce moment.")
    send_telegram(f"🤖 Robot démarré sur {SYMBOL} ({'démo' if is_demo else 'réel'})")

    if TEST_TRADE_ON_START:
        if not is_demo:
            log("Trade test refusé : compte non démo.")
        else:
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 300)
            if rates is not None and len(rates) > EMA_SLOW + 20:
                df = compute_indicators(pd.DataFrame(rates))
                ok, msg = place_order("BUY", df.iloc[-2].atr, info, lots=info.volume_min)
                log(f"TRADE TEST : {'réussi' if ok else 'échec'} -> {msg}")
            else:
                log("Trade test impossible : pas assez de données.")

    log("Robot en marche. Ctrl+C pour arrêter.")
    while True:
        try:
            update_tick_state()
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 300)
            if rates is not None and len(rates) > EMA_SLOW + 20:
                df = compute_indicators(pd.DataFrame(rates))
                df["time"] = pd.to_datetime(df["time"], unit="s")
                closed = df.iloc[:-1]   # on ignore la bougie en cours de formation
                bar_time = closed.iloc[-1]["time"]
                if bar_time != STATE["last_bar"]:
                    STATE["last_bar"] = bar_time
                    STATE["last_idle"] = time.monotonic()
                    on_new_bar(closed, info)

            if time.monotonic() - STATE["last_idle"] > IDLE_PRINT_SECONDS:
                STATE["last_idle"] = time.monotonic()
                etat = "ouvert" if market_is_open() else "FERMÉ (aucun nouveau prix)"
                log(f"En attente d'une nouvelle bougie... marché {etat}")
            time.sleep(CHECK_SECONDS)
        except KeyboardInterrupt:
            log("Arrêt demandé.")
            break
        except Exception:
            log("[ERREUR] " + traceback.format_exc())
            time.sleep(10)

    mt5.shutdown()


if __name__ == "__main__":
    main()
