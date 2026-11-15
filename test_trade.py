import MetaTrader5 as mt5
import requests
import datetime
import csv
import os

# --- CONFIGURATION ---
SYMBOL = "USDJPY"
TELEGRAM_TOKEN = "8981909540:AAGIQsGF1AZghGPiifcFSFZLbEzAHC2sY"
TELEGRAM_CHAT_ID = "513579813"
CSV_FILE = "journal_de_trading.csv"

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Erreur Telegram: {e}")

# Initialisation MT5
if not mt5.initialize():
    print("Échec de l'initialisation MT5")
    quit()

# Vérification du symbole
selected = mt5.symbol_select(SYMBOL, True)
if not selected:
    print(f"Symbole {SYMBOL} introuvable.")
    mt5.shutdown()
    quit()

symbol_info = mt5.symbol_info(SYMBOL)
tick = mt5.symbol_info_tick(SYMBOL)
ask = tick.ask
point = symbol_info.point

# Paramètres du trade
sl_pips = 100 * point  # 10 pips SL
tp_pips = 200 * point  # 20 pips TP
sl = ask - sl_pips
tp = ask + tp_pips
lot = 0.01

# Liste des modes à tester
filling_modes = [
    mt5.ORDER_FILLING_FOK,
    mt5.ORDER_FILLING_IOC,
    mt5.ORDER_FILLING_RETURN
]

result = None
succes = False

for mode in filling_modes:
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": ask,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": "Test immediat Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mode,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        succes = True
        print(f"✅ Ordre accepté avec le mode de filling : {mode}")
        break
    else:
        print(f"Échec avec mode {mode}: {result.comment} (Code: {result.retcode})")

if succes:
    # 1. Notification Telegram
    msg = f"🚀 **TEST TRADE EXÉCUTÉ**\nSymbole: {SYMBOL}\nType: BUY\nPrix: {ask}\nSL: {sl:.3f}\nTP: {tp:.3f}\nLot: {lot}"
    send_telegram(msg)
    
    # 2. Journalisation Excel / CSV
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Date", "Symbole", "Type", "Prix", "SL", "TP", "Volume", "Ticket"])
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            SYMBOL, "BUY", ask, sl, tp, lot, result.order
        ])
    print("✅ Notification envoyée et trade enregistré dans le CSV.")
else:
    print("❌ Aucun mode de filling n'a été accepté par le broker.")

mt5.shutdown()