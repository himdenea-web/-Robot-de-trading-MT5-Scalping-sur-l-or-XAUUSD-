# -Robot-de-trading-MT5-Scalping-sur-l-or-XAUUSD-
Robot de trading automatique en Python pour MetaTrader 5, spécialisé sur l’or (XAUUSD). Stratégie de tendance EMA 20/50 + RSI, stop loss et take profit basés sur l’ATR, gestion du risque (perte max par jour, 2 trades/jour, objectif hebdo), alertes Telegram. Compte démo uniquement. Projet éducatif.
✨ Fonctionnalités
Stratégie de tendance : EMA 20 / EMA 50 + repli du RSI (14), sur bougies M5 clôturées.
Sorties basées sur l'ATR : stop loss = 1,0 × ATR, take profit = 1,5 × ATR.
Calcul automatique du lot selon le risque voulu (par trade et par jour).
Garde-fous :
perte maximale par jour (MAX_DAILY_LOSS_MONEY, le risque est calculé dans le pire cas où tous les SL sont touchés) ;
2 trades maximum par jour (MAX_TRADES_PER_DAY) ;
objectif hebdomadaire (WEEKLY_PROFIT_TARGET) : le robot arrête d'ouvrir des trades une fois atteint ;
risque total plafonné, filtre de spread, lot maximum ;
refus de tourner sur un compte réel (REQUIRE_DEMO = True).
Transparence : à chaque nouvelle bougie, le robot affiche pourquoi il trade ou non.
Notifications Telegram (optionnel).
Journal CSV de tous les trades (journal_de_trading.csv).
📋 Prérequis
Windows (la bibliothèque MetaTrader5 de Python fonctionne sous Windows)
MetaTrader 5 installé, connecté à un compte démo, avec le bouton Algo Trading activé (vert)
Python 3.12
🚀 Installation
bash
git clone https://github.com/<ton-pseudo>/<nom-du-depot>.git
cd <nom-du-depot>
pip install -r requirements.txt
🔑 Configuration
Copie le fichier d'exemple :
bash
copy .env.example .env
Remplis le fichier .env (il n'est jamais envoyé sur GitHub) :
env
SYMBOL=XAUUSD
TELEGRAM_TOKEN=ton_token_telegram
TELEGRAM_CHAT_ID=ton_chat_id

Telegram est facultatif.

Ajuste les paramètres en haut de bot_mt5.py :
Paramètre	Rôle	Valeur par défaut
MAX_DAILY_LOSS_MONEY	Perte max par jour (devise du compte)	180
MAX_TRADES_PER_DAY	Trades max par jour	2
WEEKLY_PROFIT_TARGET	Gain hebdo à partir duquel on arrête	600
RISK_PERCENT	Risque par trade (% de l'equity)	0.5
MAX_POSITIONS	Positions ouvertes en même temps	2
MAX_SPREAD_POINTS	Spread maximum accepté	50
REQUIRE_DEMO	Refuse un compte réel	True
▶️ Lancer le robot
Ouvre MetaTrader 5 et connecte-toi à ton compte démo.
Lance le robot :
bash
py -3.12 bot_mt5.py

ou double-clique sur lancer.bat (pense à adapter le chemin du dossier dans le fichier).

Ctrl+C pour arrêter.
🧪 Scripts de test
Fichier	Rôle
test_mt5.py	Vérifie la connexion à MetaTrader 5
test_telegram.py	Envoie un message de test sur Telegram
test_trade.py	Ouvre un trade test de 0,01 lot (démo uniquement)

Ces scripts doivent lire leurs clés depuis le fichier .env, jamais directement dans le code.

🗂️ Structure du projet
├── bot_mt5.py          # Le robot
├── lancer.bat          # Lancement rapide sous Windows
├── requirements.txt    # Dépendances Python
├── .env.example        # Modèle de configuration (sans secrets)
├── test_mt5.py
├── test_telegram.py
└── test_trade.py
⚠️ Avertissement

Ce projet est fourni à des fins d'apprentissage. Les performances passées ne préjugent pas des performances futures. Teste toujours sur un compte démo et n'utilise jamais d'argent que tu ne peux pas te permettre de perdre.

## goldbot : nouveau framework (stratégie interchangeable)

`goldbot/` remplace progressivement `bot_mt5.py`. Le même moteur tourne en réel (MT5) et en backtest (barres M1 simulées), donc le backtest teste exactement le code qui trade.

- **Stratégie interchangeable** : `goldbot/strategies/` (premier plug-in : Opening Range Breakout). Une nouvelle stratégie = une classe `Strategy` + une ligne dans le registre.
- **Règles du compte financé** (par défaut : compte 5 000 $ en deux phases, sans limite de temps) : objectif +400 $ en phase 1 puis +200 $ en phase 2 (`GB_PROFIT_TARGET`), perte max 200 $ par jour, plancher statique à 4 600 $ (`GB_MAX_TOTAL_LOSS=400`), exposition cumulée max 150 $ (`GB_MAX_OPEN_RISK`), Stop Loss sur chaque ordre, 2 trades par jour. Chaque limite est vérifiée au pire cas (réalisé + risque ouvert + nouveau trade + marge) sur tout le compte ; la taille est réduite à la marge restante, et le trade est sauté s'il reste moins de 20 $.
- **Annonces majeures** : pas de nouveau setup de 6 minutes avant à 5 minutes après une annonce à fort impact (calendrier ForexFactory, fichier CSV ou heures fixes), car les gains réalisés dans les 5 minutes autour d'une annonce ne comptent qu'à 40 % en phase financée.
- **Attention** : les trades identiques entre utilisateurs et le copy trading sont interdits. Ne pas faire tourner ce robot sur les comptes financés de plusieurs personnes.
- **Mode alerte** (par défaut) : le robot envoie le setup sur Telegram sans passer d'ordre. **Mode trade** : il place les ordres.
- **Comité IA multi-agents (optionnel)** : 3 analystes (tendance, volatilité, risque) + un président, tous sur **un seul fournisseur** choisi dans la config (`GB_LLM_PROVIDER` + `GB_LLM_API_KEY` + `GB_LLM_MODEL`) : `anthropic`, `openai`, ou `openai_compatible` (OpenRouter, Groq, Ollama...). Le comité peut seulement refuser un setup, jamais en créer ou modifier un.

```bash
copy .env.example .env          # puis remplir
python -m goldbot --check       # connexion, réglages, état du compte
python -m goldbot               # mode alerte
python -m goldbot --mode trade --llm on
python -m pytest tests          # tests
```

Backtest du framework : `backtest/framework_backtest.ipynb`. Chances de réussir le challenge en deux phases : `backtest/funded_challenge.ipynb`. Recherche sur 7 488 stratégies : `backtest/strategy_search.ipynb`. Revue de l'indicateur d'origine : `backtest/indicator_review_backtest.ipynb`.
