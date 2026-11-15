import requests

TELEGRAM_TOKEN = "8981909540:AAH_9MqsQuMihr-KGeKGsdPP5Rc1yM8PiNk"
TELEGRAM_CHAT_ID = "5135798139"

url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
payload = {
    "chat_id": TELEGRAM_CHAT_ID,
    "text": "🎉 Salut ! Mon bot de trading communique enfin avec Telegram !"
}

print("Envoi en cours...")
response = requests.post(url, data=payload)
print(f"Code de retour : {response.status_code}")
print(f"Réponse de Telegram : {response.text}")