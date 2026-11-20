# Running goldbot 24/7 (Windows VPS or always-on laptop)

The MetaTrader 5 Python package only works on Windows, next to a running MT5 terminal. So the
bot runs on a Windows machine that stays on: a Windows VPS (recommended) or your own laptop.

```
Windows machine (VPS or laptop)
  MT5 terminal (FundedNext, logged in, Algo Trading on)
     ^  MetaTrader5 Python API
  goldbot (python -m goldbot)  <-  watchdog run_goldbot.ps1  <-  scheduled task at logon
     |  Telegram: fills, refusals, link lost/restored, daily status
     |  logs/goldbot.log, goldbot_journal.csv, goldbot_state.json
```

## VPS or laptop?

| | Windows VPS | Laptop |
|---|---|---|
| Uptime | always on, data-centre power and network | sleep, Wi-Fi drops, updates, lid, battery |
| Latency to the broker | low if the VPS is near the broker's server | home internet |
| Cost | about 15 to 40 USD a month (2 GB RAM is enough for MT5 + Python) | free |
| Setup | Remote Desktop | as below, plus keep_awake.ps1 |

A laptop works for alert mode and demo testing. For a funded account, a VPS is safer: a laptop
that sleeps while a trade is open misses the 12:00 time exit (the stop and target still sit on
the broker's server, so the loss stays capped).

## Setup (same steps on both)

1. **MT5 terminal.** Install FundedNext's MT5 terminal, log in, tick *Save password*.
   *Tools > Options > Expert Advisors*: tick *Allow algorithmic trading*; untick the options that
   disable it on account or profile change. Make sure the **Algo Trading** button in the toolbar
   is green. The bot warns on Telegram if it is off.
2. **Python 3.11+** from python.org (tick *Add to PATH*), then the code:
   ```powershell
   git clone <repository url> goldbot ; cd goldbot
   git checkout review/backtest-indicator
   powershell -ExecutionPolicy Bypass -File deploy\windows\install.ps1
   ```
3. **Settings.** Edit `.env` (created from `.env.example`). The most important lines:
   ```
   GB_MT5_PATH=C:\Program Files\FundedNext MT5 Terminal\terminal64.exe
   GB_MT5_LOGIN=12345678
   GB_MT5_PASSWORD=...            # stays on this machine; .env is never committed
   GB_MT5_SERVER=FundedNext-Server
   GB_REQUIRE_DEMO=true           # --check prints DEMO or REAL; prop accounts are usually DEMO-type
   GB_RISK_PER_TRADE=50
   GB_START_BALANCE=5000
   GB_PROFIT_TARGET=400           # phase 1; 200 in phase 2; 0 once funded
   GB_ACCOUNT_START=2026-11-23    # first day of the current phase
   TELEGRAM_TOKEN=... / TELEGRAM_CHAT_ID=...
   ```
4. **Check.** `.\.venv\Scripts\python.exe -m goldbot --check` connects, prints the contract size,
   lot limits, the real leverage, all limits and the account state, then exits. Fix anything it
   reports before going further.
5. **Start automatically.**
   ```powershell
   powershell -ExecutionPolicy Bypass -File deploy\windows\register_task.ps1 -Mode alert
   Start-ScheduledTask -TaskName goldbot
   ```
   Use `-Mode trade` once alert mode has behaved for a while on the demo.

### VPS only
* Pick a Windows Server VPS. Ask FundedNext which city their MT5 server is in and choose a VPS near it.
* Leave Remote Desktop by **closing the window (disconnect)**, never *Sign out*: signing out
  closes MT5 and the bot. To survive a VPS reboot, enable automatic sign-in
  (Sysinternals *Autologon*), so the logon task starts again.

### Laptop only
* Run `deploy\windows\keep_awake.ps1` once as administrator (no sleep, lid ignored on power).
* Keep it plugged in, on a wired or stable connection, and set Windows Update active hours to
  cover 03:00 to 17:00 New York time.

## Day to day

| Action | How |
|---|---|
| See what it does | Telegram messages; `logs\goldbot.log`; `goldbot_journal.csv` |
| Stop cleanly | create an empty file `STOP` next to `goldbot_state.json` (the watchdog ends too) |
| Restart | `Start-ScheduledTask -TaskName goldbot` |
| Next phase | set `GB_PROFIT_TARGET`, `GB_START_BALANCE`, `GB_ACCOUNT_START`, `GB_MT5_LOGIN` for the new account, then restart |
| Remove | `Unregister-ScheduledTask -TaskName goldbot -Confirm:$false` |

## What happens when things go wrong

| Event | Behaviour |
|---|---|
| Bot crashes | the watchdog restarts it after 30 s; on start it adopts its own orders and positions (time exits and planned risk from `goldbot_state.json`) and rebuilds today's trades from the MT5 history, so the 2-per-day limit and daily loss still hold |
| MT5 link drops | Telegram warning, reconnect every 30 s, Telegram when restored; no new orders meanwhile |
| Algo Trading switched off | Telegram warning every 30 minutes; orders are refused by MT5 |
| Order refused (margin, price moved) | Telegram message with the reason |
| Second copy started | refuses to start (exit code 3) |
| Real account while GB_REQUIRE_DEMO=true | refuses to start (exit code 2) |
| Machine off or asleep | stop and target are on the broker's server; the 12:00 time exit is done at the next start |

## FundedNext rules to confirm before trading live

* EAs are allowed on accounts below 50,000 "as long as their settings are aligned with the
  account's parameters"; a separate *EA add-on* is also mentioned. Confirm with support that an
  automated Python bot through the MT5 API is fine on your account.
* The real XAUUSD spread on their server during New York hours: the strategy holds up to about
  0.30 (see `backtest/` results), and becomes marginal at 0.40.
* Copy trading or identical trades between different people's accounts are forbidden: run this
  bot on one person's account.
