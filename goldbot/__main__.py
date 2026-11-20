"""Run the bot live on MetaTrader 5.

    python -m goldbot                    # settings from .env (alert mode by default)
    python -m goldbot --mode trade       # place orders (demo account unless GB_REQUIRE_DEMO=false)
    python -m goldbot --llm on           # enable the committee (provider from GB_LLM_PROVIDER)
    python -m goldbot --check            # connect, print settings and account state, then exit
"""
import argparse
import time
import traceback

import pandas as pd

from goldbot.config import Config
from goldbot.engine import Engine
from goldbot.notify import Journal, Telegram
from goldbot.strategies import make_strategy


def log(msg):
    print(f"[{pd.Timestamp.utcnow():%Y-%m-%d %H:%M:%S} UTC] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(prog="goldbot")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--mode", choices=["alert", "trade"])
    ap.add_argument("--llm", choices=["on", "off"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    cfg = Config.from_env(a.env)
    if a.mode:
        cfg.mode = a.mode
    if a.llm:
        cfg.judge.enabled = a.llm == "on"
    cfg.validate()

    from goldbot.brokers.mt5 import MT5Broker, MT5Connection, MT5Market
    conn = MT5Connection(cfg.symbol, require_demo=cfg.require_demo)
    market, broker = MT5Market(conn), MT5Broker(conn, cfg.magic, cfg.risk.count_all_account_pnl)
    strategy = make_strategy(cfg.strategy, **cfg.strategy_params)
    committee = None
    if cfg.judge.enabled:
        from goldbot.judge import Committee
        committee = Committee(cfg.judge)
    notifier = Telegram(cfg.telegram_token, cfg.telegram_chat_id, log=log)
    engine = Engine(cfg, strategy, market, broker, committee, notifier, Journal(cfg.journal_path))

    acc = broker.account()
    log(f"{conn.symbol} | {'DEMO' if conn.is_demo else 'REAL'} | equity {acc['equity']:.2f} | "
        f"server offset {conn.offset_s / 3600:+.1f}h | mode {cfg.mode.upper()}")
    log(f"strategy {strategy.describe()}")
    r = cfg.risk
    log(f"risk {r.risk_per_trade_usd}/trade | daily limit {r.daily_loss_limit} | floor "
        f"{r.start_balance - r.max_total_loss if r.max_total_loss else 'off'} | target "
        f"{r.start_balance + r.profit_target if r.profit_target else 'off'} | max open risk "
        f"{r.max_open_risk or 'off'} | news {cfg.news.source}")
    if acc["balance"] < r.start_balance - r.max_total_loss + r.min_risk_usd and r.max_total_loss:
        log("WARNING: balance is at or near the floor; check GB_START_BALANCE")
    log(f"committee {'ON via ' + committee.provider.describe() if committee else 'OFF'}"
        + (f" | fail mode {cfg.judge.fail_mode}" if committee else ""))
    st = engine.account_state(market.now())
    log(f"account state: {st.as_dict(cfg.risk)}")
    if a.check:
        conn.shutdown()
        return

    notifier.send(f"goldbot started on {conn.symbol} ({cfg.mode} mode, "
                  f"{'demo' if conn.is_demo else 'REAL'} account)")
    last_minute = None
    while True:
        try:
            now = market.now()
            if now != last_minute:          # one engine step per new minute (closed M1 bar)
                last_minute = now
                conn.refresh_offset()
                engine.step()
            time.sleep(cfg.poll_seconds)
        except KeyboardInterrupt:
            log("stopped by user")
            break
        except Exception:
            log("[error] " + traceback.format_exc())
            time.sleep(10)
    conn.shutdown()


if __name__ == "__main__":
    main()
