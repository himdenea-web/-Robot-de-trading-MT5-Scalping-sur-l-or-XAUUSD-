"""Unattended live runner (VPS or always-on laptop).

Responsibilities around the engine:
  * one running copy only (a local port lock), logs to a rotating file and the console
  * connects to MT5 (optionally starting and logging in the terminal), retrying with backoff
  * uses the broker's real contract size, lot limits and leverage for sizing
  * one engine step per new minute; checks the terminal link every loop and reconnects
  * warns on Telegram when the link drops or comes back, when Algo Trading is switched off,
    when errors repeat, and sends a daily status message
  * exits cleanly on Ctrl+C or a stop request (create the file STOP next to the state file)
The watchdog script in deploy/windows restarts the process if it ever exits with an error.
"""
import logging
import logging.handlers
import os
import socket
import time
import traceback

import pandas as pd

from goldbot.config import Config
from goldbot.engine import Engine
from goldbot.notify import Journal, Telegram
from goldbot.state import StateStore
from goldbot.strategies import make_strategy

NY = "America/New_York"


class FatalError(Exception):
    """A problem a restart cannot fix (exit code 2: configuration, 3: already running)."""

    def __init__(self, msg, code=2):
        super().__init__(msg)
        self.code = code


def setup_logging(path: str) -> logging.Logger:
    lg = logging.getLogger("goldbot")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s UTC %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fmt.converter = time.gmtime
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    lg.addHandler(ch)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=10, encoding="utf-8")
        fh.setFormatter(fmt)
        lg.addHandler(fh)
    return lg


class SingleInstance:
    """Binding a local port fails if another copy already holds it (also after a crash the
    port is released by the OS, unlike a lock file)."""

    def __init__(self, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.bind(("127.0.0.1", port))
        except OSError:
            raise FatalError(f"another goldbot is already running (port {port} in use); not starting", 3)


class RateLimited:
    def __init__(self, notifier, every_s=1800):
        self.n, self.every, self.last = notifier, every_s, {}

    def send(self, key, text):
        t = time.time()
        if t - self.last.get(key, 0) >= self.every:
            self.last[key] = t
            self.n.send(text)


def status_text(engine, broker, cfg) -> str:
    st = engine.account_state(engine.market.now())
    d = st.as_dict(cfg.risk)
    acc = broker.account()
    parts = [f"balance {acc['balance']:.2f}", f"equity {acc['equity']:.2f}",
             f"today {d['day_realized_usd']:+.2f}", f"trades today {d['trades_today']}",
             f"open risk {d['open_risk_usd']:.2f}"]
    if "room_to_floor_usd" in d:
        parts.append(f"room to floor {d['room_to_floor_usd']:.2f}")
    if "to_profit_target_usd" in d:
        parts.append(f"to target {d['to_profit_target_usd']:.2f}")
    if st.trading_days is not None:
        parts.append(f"trading days {st.trading_days}/{cfg.risk.min_trading_days}")
    return "goldbot status: " + ", ".join(parts)


def connect(cfg, log, alerts):
    from goldbot.brokers.mt5 import MT5Connection, MT5Error
    delay = 10
    while True:
        try:
            return MT5Connection(cfg.symbol, require_demo=cfg.require_demo, path=cfg.mt5_path,
                                 login=cfg.mt5_login, password=cfg.mt5_password,
                                 server=cfg.mt5_server, log=log.info)
        except MT5Error as e:
            if "real account" in str(e) or "expected" in str(e):
                raise FatalError(str(e), 2)
            log.warning(f"MT5 connection failed: {e}; retrying in {delay}s")
            alerts.send("connect", f"goldbot cannot connect to MT5: {e}")
            time.sleep(delay)
            delay = min(delay * 2, 300)


def run(cfg: Config, check=False):
    log = setup_logging(cfg.log_file)
    lock = SingleInstance(cfg.lock_port)  # noqa: F841  (held for the life of the process)
    notifier = Telegram(cfg.telegram_token, cfg.telegram_chat_id, log=log.info)
    alerts = RateLimited(notifier)

    from goldbot.brokers.mt5 import MT5Broker, MT5Market, apply_symbol_specs
    conn = connect(cfg, log, alerts)
    margin, price = conn.margin_per_lot()
    changed = apply_symbol_specs(cfg.risk, conn.info, margin, price)
    i = conn.info
    log.info(f"{conn.symbol} | {'DEMO' if conn.is_demo else 'REAL'} | {i.trade_contract_size:g} oz per lot, "
             f"lots {i.volume_min:g} to {i.volume_max:g} step {i.volume_step:g}, leverage 1:{cfg.risk.leverage:g}"
             + (f" | sizing updated from broker: {changed}" if changed else ""))

    state = StateStore(cfg.state_file)
    market = MT5Market(conn)
    broker = MT5Broker(conn, cfg.magic, cfg.risk.count_all_account_pnl, state=state, log=log.info)
    strategy = make_strategy(cfg.strategy, **cfg.strategy_params)
    committee = None
    if cfg.judge.enabled:
        from goldbot.judge import Committee
        committee = Committee(cfg.judge)
    engine = Engine(cfg, strategy, market, broker, committee, notifier, Journal(cfg.journal_path))
    engine.news.log = log.info

    r = cfg.risk
    log.info(f"mode {cfg.mode.upper()} | strategy {strategy.describe()}")
    log.info(f"risk {r.risk_per_trade_usd}/trade | daily limit {r.daily_loss_limit} | floor "
             f"{r.start_balance - r.max_total_loss if r.max_total_loss else 'off'} | target "
             f"{r.start_balance + r.profit_target if r.profit_target else 'off'} | max open risk "
             f"{r.max_open_risk or 'off'} | commission {r.commission_rate:.4%}/side | news {cfg.news.source}")
    log.info(f"committee {'ON via ' + committee.provider.describe() if committee else 'OFF'}")
    ok, why = conn.trading_allowed()
    if cfg.mode == "trade" and not ok:
        log.warning(why)
    log.info(status_text(engine, broker, cfg))
    if check:
        conn.shutdown()
        return 0

    notifier.send(f"goldbot started on {conn.symbol} ({cfg.mode} mode, "
                  f"{'demo' if conn.is_demo else 'REAL'} account)\n" + status_text(engine, broker, cfg))
    stop_file = os.path.join(os.path.dirname(os.path.abspath(cfg.state_file)), "STOP")
    last_minute, errors, link_down, last_beat_day = None, 0, False, None
    while True:
        try:
            if os.path.isfile(stop_file):
                os.remove(stop_file)
                log.info("STOP file found: stopping")
                notifier.send("goldbot stopped (STOP file)")
                break
            healthy, why = conn.healthy()
            if not healthy:
                if not link_down:
                    link_down = True
                    log.warning(f"MT5 link lost: {why}")
                    notifier.send(f"goldbot: MT5 link lost ({why}); reconnecting")
                try:
                    conn.reconnect()
                except Exception as e:
                    log.warning(f"reconnect failed: {e}")
                    time.sleep(30)
                continue
            if link_down:
                link_down = False
                log.info("MT5 link restored")
                notifier.send("goldbot: MT5 link restored")
            if cfg.mode == "trade":
                ok, why = conn.trading_allowed()
                if not ok:
                    alerts.send("algo", f"goldbot cannot trade: {why}")
            now = market.now()
            if now != last_minute:          # one engine step per new minute (closed M1 bar)
                last_minute = now
                conn.refresh_offset()
                engine.step()
                errors = 0
                if cfg.heartbeat_ny:
                    ny = pd.Timestamp(now).tz_localize("UTC").tz_convert(NY)
                    if ny.strftime("%H:%M") >= cfg.heartbeat_ny and ny.date() != last_beat_day:
                        last_beat_day = ny.date()
                        notifier.send(status_text(engine, broker, cfg))
            time.sleep(cfg.poll_seconds)
        except KeyboardInterrupt:
            log.info("stopped by user")
            break
        except Exception:
            errors += 1
            log.error(traceback.format_exc())
            if errors >= 3:
                alerts.send("errors", f"goldbot: {errors} errors in a row, last: "
                                      f"{traceback.format_exc().strip().splitlines()[-1]}")
            time.sleep(min(10 * errors, 120))
    conn.shutdown()
    return 0
