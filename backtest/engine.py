"""Fast backtest engine for the bot_mt5.py strategy (EMA trend + RSI pullback, ATR exits).

- Signals are computed on M5 bars with the same formulas as bot_mt5.compute_indicators /
  get_signal (checked for equality in the notebook).
- Exits are resolved on M1 bars, so the order of SL/TP hits inside a 5 minute candle is
  known instead of guessed. If SL and TP fall inside one single M1 bar, SL is assumed first.
- Prices are HistData bid quotes. Longs enter at ask (bid + spread) and exit on bid;
  shorts enter at bid and exit on ask (bid + spread), like MetaTrader.
"""
import glob
import io
import math
import os
import zipfile
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(HERE, "data", "histdata")
CACHE = os.path.join(HERE, "data", "xauusd_m1.parquet")
CONTRACT = 100.0          # oz per lot
SERVER_UTC_OFFSET = 2     # typical MT5 broker server time (UTC+2), used for day/week resets


# ------------------------------------------------------------------ data
def load_m1():
    if os.path.isfile(CACHE):
        return pd.read_parquet(CACHE)
    frames = []
    for z in sorted(glob.glob(os.path.join(HIST, "XAUUSD_*.zip"))):
        with zipfile.ZipFile(z) as zf:
            name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            frames.append(pd.read_csv(io.BytesIO(zf.read(name)), sep=";", header=None,
                                      names=["t", "open", "high", "low", "close", "v"]))
    df = pd.concat(frames, ignore_index=True)
    # HistData timestamps are EST without daylight saving (fixed UTC-5)
    df["time"] = pd.to_datetime(df["t"], format="%Y%m%d %H%M%S") + pd.Timedelta(hours=5)
    df = (df.drop(columns=["t", "v"]).drop_duplicates("time").sort_values("time")
            .reset_index(drop=True))
    df.to_parquet(CACHE)
    return df


def resample(m1, rule="5min"):
    out = (m1.set_index("time")
             .resample(rule, label="left", closed="left")
             .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
             .dropna().reset_index())
    return out


# ------------------------------------------------------------------ indicators / signal
@dataclass(frozen=True)
class Params:
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    rsi_buy: float = 45
    rsi_sell: float = 55
    min_trend_atr: float = 0.2
    sl_atr: float = 1.0
    tp_atr: float = 1.5
    # optional filters (all off = original bot)
    hours_utc: tuple = ()          # allowed entry hours (UTC); empty = all
    htf_filter: bool = False       # require H1 close above/below H1 EMA50 in trade direction
    min_atr_spreads: float = 0.0   # skip if ATR < k * spread
    max_hold_min: int = 0          # time exit after N minutes (0 = none)


def indicators(df, p: Params):
    c = df["close"]
    out = df.copy()
    out["ema_fast"] = c.ewm(span=p.ema_fast, adjust=False).mean()
    out["ema_slow"] = c.ewm(span=p.ema_slow, adjust=False).mean()
    d = c.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = (up.ewm(alpha=1 / p.rsi_period, adjust=False).mean()
          / dn.ewm(alpha=1 / p.rsi_period, adjust=False).mean())
    out["rsi"] = 100 - 100 / (1 + rs)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / p.atr_period, adjust=False).mean()
    return out


def add_htf(m5, m1):
    """H1 close vs H1 EMA50, known only once the H1 bar has closed (no lookahead)."""
    h1 = resample(m1, "1h")
    h1["h1_ema"] = h1["close"].ewm(span=50, adjust=False).mean()
    h1["h1_bias"] = np.sign(h1["close"] - h1["h1_ema"])
    h1["avail"] = h1["time"] + pd.Timedelta(hours=1)
    m5 = m5.copy()
    m5["close_time"] = m5["time"] + pd.Timedelta(minutes=5)
    m5 = pd.merge_asof(m5.sort_values("close_time"), h1[["avail", "h1_bias"]],
                       left_on="close_time", right_on="avail", direction="backward")
    return m5.drop(columns=["avail", "close_time"])


def signals(ind, p: Params, spread=0.30):
    """+1 buy / -1 sell / 0, evaluated on the CLOSED bar i (same rules as get_signal)."""
    ef, es, c, rsi, atr = (ind[k].to_numpy() for k in ("ema_fast", "ema_slow", "close", "rsi", "atr"))
    prev_rsi = np.r_[np.nan, rsi[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        strength = np.where(atr > 0, np.abs(ef - es) / atr, 0.0)
    strong = strength >= p.min_trend_atr
    up = (ef > es) & (c > es)
    dn = (ef < es) & (c < es)
    buy = strong & up & (prev_rsi < p.rsi_buy) & (p.rsi_buy <= rsi)
    sell = strong & dn & (prev_rsi > p.rsi_sell) & (p.rsi_sell >= rsi)
    sig = np.where(buy, 1, np.where(sell, -1, 0))
    sig[: p.ema_slow + 20] = 0
    if p.hours_utc:
        # the order is sent at the open of the next bar
        hrs = (ind["time"] + pd.Timedelta(minutes=5)).dt.hour.to_numpy()
        sig[~np.isin(hrs, p.hours_utc)] = 0
    if p.htf_filter:
        sig[sig != ind["h1_bias"].fillna(0).to_numpy()] = 0
    if p.min_atr_spreads > 0:
        sig[atr < p.min_atr_spreads * spread] = 0
    return sig


# ------------------------------------------------------------------ exits on M1
class ExitSolver:
    def __init__(self, m1):
        self.t = m1["time"].to_numpy()
        self.o, self.h, self.l, self.c = (m1[k].to_numpy() for k in ("open", "high", "low", "close"))

    def idx(self, ts):
        return int(np.searchsorted(self.t, np.datetime64(ts)))

    def solve(self, j, d, atr, p: Params, spread, slip=0.0):
        """Trade entered at the open of M1 bar j. Returns (exit_j, entry, exit_px, why, sl_dist)."""
        n = len(self.o)
        if j >= n:
            return None
        sl_d, tp_d = max(p.sl_atr * atr, 0.02), max(p.tp_atr * atr, 0.02)
        if d > 0:
            entry = self.o[j] + spread + slip
            sl, tp = entry - sl_d, entry + tp_d
            lo_adj = hi_adj = 0.0          # long exits on bid
        else:
            entry = self.o[j] - slip
            sl, tp = entry + sl_d, entry - tp_d
            lo_adj = hi_adj = spread       # short exits on ask = bid + spread
        limit = j + p.max_hold_min if p.max_hold_min else n
        start, win = j, 720
        while start < min(n, limit):
            end = min(n, limit, start + win)
            hh, ll, oo = self.h[start:end] + hi_adj, self.l[start:end] + lo_adj, self.o[start:end] + lo_adj
            if d > 0:
                hit_sl, hit_tp = ll <= sl, hh >= tp
            else:
                hit_sl, hit_tp = hh >= sl, ll <= tp
            any_hit = hit_sl | hit_tp
            if any_hit.any():
                k = int(np.argmax(any_hit))
                jj = start + k
                if k > 0 or jj > j:   # gap through a level at a bar open fills at the open
                    if (d > 0 and oo[k] <= sl) or (d < 0 and oo[k] >= sl):
                        return jj, entry, oo[k] - d * slip, "sl", sl_d
                    if (d > 0 and oo[k] >= tp) or (d < 0 and oo[k] <= tp):
                        return jj, entry, oo[k], "tp", sl_d
                if hit_sl[k]:
                    return jj, entry, sl - d * slip, ("sl" if not hit_tp[k] else "sl-ambig"), sl_d
                return jj, entry, tp, "tp", sl_d
            start, win = end, win * 2
        jj = min(n, limit) - 1
        return jj, entry, self.c[jj] + lo_adj, ("time" if p.max_hold_min else "end"), sl_d


# ------------------------------------------------------------------ simulations
def _entries(m5, sig, ex: ExitSolver):
    """Signal on closed bar i -> entry at open of bar i+1 (index into M1)."""
    ii = np.flatnonzero(sig[:-1])
    times = m5["time"].to_numpy()[ii + 1]
    js = np.searchsorted(ex.t, times)
    return ii, js


def run_raw(m5, sig, ex, p: Params, spread=0.30, slip=0.0, delay_bars=0):
    """Every signal, one position at a time, fixed 1R. Returns trades with R multiples."""
    ii, js = _entries(m5, sig, ex)
    atr = m5["atr"].to_numpy()
    rows, busy_until = [], -1
    for i, j in zip(ii, js):
        j = j + delay_bars * 5
        if j <= busy_until:
            continue
        r = ex.solve(j, sig[i], atr[i], p, spread, slip)
        if r is None:
            continue
        xj, entry, xp, why, sl_d = r
        busy_until = xj
        R = (xp - entry) * sig[i] / sl_d
        rows.append((ex.t[j], ex.t[xj], int(sig[i]), entry, xp, why, sl_d, R, atr[i]))
    tr = pd.DataFrame(rows, columns=["entry_time", "exit_time", "dir", "entry", "exit", "why",
                                     "sl_dist", "R", "atr"])
    return tr


@dataclass
class Rules:
    start_equity: float = 100_000.0
    risk_percent: float = 0.5
    max_positions: int = 2
    max_trades_per_day: int = 2
    max_daily_loss: float = 180.0
    weekly_target: float = 600.0
    max_total_risk_pct: float = 2.0
    close_all_on_limit: bool = True
    force_min_lot: bool = True
    max_overrisk: float = 3.0
    max_lots: float = 2.0
    vol_min: float = 0.01
    vol_step: float = 0.01


def _server_day(ts):
    s = pd.Timestamp(ts) + pd.Timedelta(hours=SERVER_UTC_OFFSET)
    return s.normalize()


def run_bot(m5, sig, ex, p: Params, rules: Rules = Rules(), spread=0.30, slip=0.0):
    """Faithful replay of bot_mt5.py gates (daily loss budget, 2 trades/day, weekly target,
    2 positions max, no opposite positions, close-all on daily limit, lot sizing)."""
    ii, js = _entries(m5, sig, ex)
    atr, close = m5["atr"].to_numpy(), m5["close"].to_numpy()
    equity = rules.start_equity
    open_pos, done = [], []
    halted_day = None

    def realize(upto_j):
        nonlocal equity
        keep = []
        for q in open_pos:
            if q["xj"] <= upto_j:
                equity += q["pnl"]
                done.append(q)
            else:
                keep.append(q)
        return keep

    for i, j in zip(ii, js):
        if j >= len(ex.o):
            break
        open_pos = realize(j - 1)
        d = int(sig[i])
        day = _server_day(ex.t[j])
        week = day - pd.Timedelta(days=day.weekday())
        day_real = sum(q["pnl"] for q in done if q["xday"] == day)
        floating = sum((close[i] - q["entry"]) * q["dir"] * q["lots"] * CONTRACT
                       - (spread * q["lots"] * CONTRACT if q["dir"] < 0 else 0) for q in open_pos)
        loss = max(0.0, -(day_real + floating))
        if halted_day == day or loss >= rules.max_daily_loss:
            if halted_day != day:
                halted_day = day
                if rules.close_all_on_limit:
                    for q in open_pos:
                        px = ex.o[j] + (spread if q["dir"] < 0 else 0)
                        q.update(xj=j, exit=px, why="daily-limit", xday=day,
                                 pnl=(px - q["entry"]) * q["dir"] * q["lots"] * CONTRACT)
                    open_pos = realize(j)
            continue
        if sum(q["pnl"] for q in done if q["xweek"] == week) >= rules.weekly_target:
            continue
        opened = sum(1 for q in done + open_pos if q["day"] == day)
        if opened >= rules.max_trades_per_day:
            continue
        if len(open_pos) >= rules.max_positions or any(q["dir"] != d for q in open_pos):
            continue
        r = ex.solve(j, d, atr[i], p, spread, slip)
        if r is None:
            continue
        xj, entry, xp, why, sl_d = r
        lpl = sl_d * CONTRACT
        realized_loss = max(0.0, -day_real)
        open_risk = sum(q["risk"] for q in open_pos)
        budget = rules.max_daily_loss - realized_loss - open_risk
        if budget <= 0:
            continue
        trades_left = max(1, rules.max_trades_per_day - opened)
        risk_money = min(equity * rules.risk_percent / 100, budget / trades_left)
        lots = round(math.floor(risk_money / lpl / rules.vol_step) * rules.vol_step, 2)
        if lots < rules.vol_min:
            if rules.force_min_lot and rules.vol_min * lpl <= risk_money * rules.max_overrisk:
                lots = rules.vol_min
            else:
                continue
        lots = min(lots, rules.max_lots)
        risk = lots * lpl
        if risk > budget or open_risk + risk > equity * rules.max_total_risk_pct / 100:
            continue
        xday = _server_day(ex.t[xj])
        open_pos.append(dict(entry_time=ex.t[j], exit_time=ex.t[xj], j=j, xj=xj, dir=d,
                             entry=entry, exit=xp, why=why, lots=lots, risk=risk, day=day,
                             xday=xday, xweek=xday - pd.Timedelta(days=xday.weekday()),
                             pnl=(xp - entry) * d * lots * CONTRACT))
    open_pos = realize(len(ex.o))
    tr = pd.DataFrame(done)
    if not tr.empty:
        tr["exit_time"] = [ex.t[x] for x in tr["xj"]]
        tr = tr.sort_values("exit_time").reset_index(drop=True)
        tr["R"] = tr["pnl"] / tr["risk"]
    return tr, equity


# ------------------------------------------------------------------ metrics
def summary(tr, col="R"):
    if tr is None or len(tr) == 0:
        return dict(trades=0, win_pct=np.nan, avg=np.nan, total=0.0, pf=np.nan,
                    max_dd=0.0, t_stat=np.nan)
    x = tr[col].to_numpy()
    eq = np.cumsum(x)
    dd = (eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
    wins = x > 0
    pf = x[wins].sum() / -x[~wins].sum() if (~wins).any() and x[~wins].sum() < 0 else np.inf
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return dict(trades=len(x), win_pct=100 * wins.mean(), avg=x.mean(), total=x.sum(),
                pf=pf, max_dd=dd, t_stat=x.mean() / (sd / np.sqrt(len(x))) if sd else np.nan)


def psr(x, sr_benchmark=0.0):
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado) of per-trade returns."""
    from scipy.stats import norm, skew, kurtosis
    x = np.asarray(x, float)
    n = len(x)
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = skew(x), kurtosis(x, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom)), sr


def expected_max_sr(n_trials, sr_var):
    """Expected maximum Sharpe of n_trials skill-less strategies (for the Deflated Sharpe)."""
    from scipy.stats import norm
    g = 0.5772156649
    return np.sqrt(sr_var) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e)))
