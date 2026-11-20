"""Large strategy search on XAUUSD with data-mining controls.

Every strategy is (entry rule, session window, trend filter, exit rule). Entry rules are
computed on 5-minute bars and filled at the next 1-minute open; exits (ATR stop, R-multiple
target, maximum holding time, forced flat at 16:45 New York) are resolved on 1-minute bars
with spread and slippage, one position at a time and at most 2 trades per trading day.

    from search import build_universe, run_all
    U = build_universe()            # data, indicators and every entry rule (cached)
    res = run_all(U)                # trades for every strategy

The selection period is 2022-2024 (in-sample, IS). 2025-2026 is the untouched holdout (OOS).
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

NY = "America/New_York"
SPREAD, SLIP = 0.30, 0.10
IS_END = pd.Timestamp("2025-01-01")
MAX_PER_DAY = 2

SESSIONS = {             # New York local hours of the signal bar close [start, end)
    "asia": (19.0, 26.0),       # 19:00 to 02:00
    "london": (3.0, 8.0),
    "ny": (8.0, 12.0),
    "day": (3.0, 16.0),
}
EXITS = list(itertools.product((1.5, 3.0, 5.0), (1.0, 2.0, 3.0), (60, 240)))  # sl_atr, rr, hold


# ------------------------------------------------------------------ data
class Data:
    def __init__(self, m1):
        self.m1 = m1
        t = m1["time"]
        self.t1 = t.to_numpy().astype("datetime64[ns]").astype(np.int64)
        self.o, self.h, self.l, self.c = (m1[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
        ny1 = t.dt.tz_localize("UTC").dt.tz_convert(NY).dt.tz_localize(None)
        day1 = (ny1 + pd.Timedelta(hours=7)).dt.normalize()          # trading day rolls at 17:00
        # forced flat at 16:45 New York of each trading day
        days = day1.drop_duplicates()
        cut_local = days + pd.Timedelta(hours=16, minutes=45)
        cut_utc = (cut_local.dt.tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
                   .dt.tz_convert("UTC").dt.tz_localize(None))
        cut_map = pd.Series(cut_utc.to_numpy(), index=days.to_numpy())
        cut_for_bar = cut_map.reindex(day1.to_numpy()).to_numpy().astype("datetime64[ns]").astype(np.int64)
        self.eod = (np.searchsorted(self.t1, cut_for_bar) - 1).astype(np.int64)

        m5 = E.resample(m1, "5min")
        self.m5 = m5
        ny5 = m5["time"].dt.tz_localize("UTC").dt.tz_convert(NY).dt.tz_localize(None)
        close_local = ny5 + pd.Timedelta(minutes=5)
        self.hour = (close_local.dt.hour + close_local.dt.minute / 60.0).to_numpy()
        self.day = (close_local + pd.Timedelta(hours=7)).dt.normalize()
        self.day_np = self.day.to_numpy()
        self.day_id = pd.factorize(self.day)[0].astype(np.int64)
        # entry: first M1 bar at or after the M5 bar's close, if it is within 5 minutes
        tclose = (m5["time"] + pd.Timedelta(minutes=5)).to_numpy().astype("datetime64[ns]").astype(np.int64)
        j = np.searchsorted(self.t1, tclose)
        ok = j < len(self.t1)
        jj = np.where(ok, j, len(self.t1) - 1)
        ok &= (self.t1[jj] - tclose) <= 5 * 60 * 10**9
        self.entry_j = np.where(ok, jj, -1).astype(np.int64)

        o, h, l, c = (m5[k] for k in ("open", "high", "low", "close"))
        self.O, self.H, self.L, self.C = o, h, l, c
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        self.atr = tr.ewm(alpha=1 / 14, adjust=False).mean().to_numpy()

        # daily bars on the trading day, trend = sign(close - EMA20) of the previous day
        d = pd.DataFrame({"day": self.day, "o": o, "h": h, "l": l, "c": c})
        g = d.groupby("day")
        D = pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last()})
        D["trend"] = np.sign(D.c - D.c.ewm(span=20, adjust=False).mean()).shift(1)
        D["pdh"], D["pdl"] = D.h.shift(1), D.l.shift(1)
        self.D = D
        self.trend = D.trend.reindex(self.day).fillna(0).to_numpy()
        self.pdh = D.pdh.reindex(self.day).to_numpy()
        self.pdl = D.pdl.reindex(self.day).to_numpy()
        self.oos = (self.day >= IS_END).to_numpy()


# ------------------------------------------------------------------ indicator helpers
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def cross_up(a, b):
    a, b = np.asarray(a), np.asarray(b)
    out = np.zeros(len(a), bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def cross_dn(a, b):
    return cross_up(-np.asarray(a), -np.asarray(b))


def events(long_, short_):
    s = np.zeros(len(long_), np.int8)
    s[np.asarray(long_, bool)] = 1
    s[np.asarray(short_, bool)] = -1
    return s


@njit(cache=True)
def _supertrend(h, l, c, atr, mult):
    n = len(c)
    up = np.empty(n)
    dn = np.empty(n)
    dirn = np.ones(n, np.int8)
    for i in range(n):
        mid = (h[i] + l[i]) / 2
        bu, bd = mid + mult * atr[i], mid - mult * atr[i]
        if i == 0:
            up[i], dn[i] = bu, bd
            continue
        up[i] = bu if (bu < up[i - 1] or c[i - 1] > up[i - 1]) else up[i - 1]
        dn[i] = bd if (bd > dn[i - 1] or c[i - 1] < dn[i - 1]) else dn[i - 1]
        if dirn[i - 1] == 1:
            dirn[i] = -1 if c[i] < dn[i] else 1
        else:
            dirn[i] = 1 if c[i] > up[i] else -1
    return dirn


def session_mean(D: Data):
    """Running mean of typical price since the 18:00 reopen, with running std (VWAP without volume)."""
    tp = (D.H + D.L + D.C) / 3
    g = tp.groupby(D.day_id)
    n = g.cumcount() + 1
    m = g.cumsum() / n
    m2 = (tp ** 2).groupby(D.day_id).cumsum() / n
    sd = np.sqrt((m2 - m ** 2).clip(lower=0))
    return m, sd


def range_levels(D: Data, start_h, end_h):
    """High/low of each trading day's window [start_h, end_h) (local hours of bar close, 24+ for after midnight)."""
    hr = D.hour.copy()
    hr = np.where(hr >= 17.0 + 1e-9, hr - 24.0, hr)   # trading-day clock: 18:00 = -6
    s, e = (start_h - 24 if start_h >= 17 else start_h), (end_h - 24 if end_h >= 17 else end_h)
    inwin = (hr > s + 1e-9) & (hr <= e + 1e-9)
    df = pd.DataFrame({"d": D.day_id, "h": D.H.where(inwin), "l": D.L.where(inwin)})
    hi = df.groupby("d").h.transform("max").to_numpy()
    lo = df.groupby("d").l.transform("min").to_numpy()
    after = hr > e + 1e-9
    return hi, lo, after, hr


# ------------------------------------------------------------------ entry rules
def entry_rules(D: Data):
    """Returns {name: (signal array int8, generic: bool)}; generic rules get the session windows."""
    C, H, L = D.C, D.H, D.L
    c = C.to_numpy()
    R = {}

    for f, s in [(9, 21), (20, 50), (20, 100), (50, 200)]:
        a, b = ema(C, f), ema(C, s)
        R[f"ema_cross_{f}_{s}"] = (events(cross_up(a, b), cross_dn(a, b)), True)
    for n in [12, 24, 48, 96, 288]:
        hh, ll = H.rolling(n).max().shift(1), L.rolling(n).min().shift(1)
        R[f"donchian_{n}"] = (events(cross_up(C, hh), cross_dn(C, ll)), True)
    for k in [2.0, 2.5, 3.0]:
        m, sd = C.rolling(20).mean(), C.rolling(20).std()
        up, dn = m + k * sd, m - k * sd
        R[f"bb_fade_{k}"] = (events(cross_dn(C, dn), cross_up(C, up)), True)
        R[f"bb_break_{k}"] = (events(cross_up(C, up), cross_dn(C, dn)), True)
    for n, lo, hi in [(2, 5, 95), (2, 10, 90), (7, 20, 80), (14, 25, 75), (14, 30, 70)]:
        r = rsi(C, n).to_numpy()
        R[f"rsi_fade_{n}_{lo}"] = (events(cross_dn(r, np.full_like(r, lo)), cross_up(r, np.full_like(r, hi))), True)
    for lo in [5, 10, 20]:
        for e in [200, 600]:
            r = rsi(C, 2).to_numpy()
            tr = (C > ema(C, e)).to_numpy()
            R[f"connors_rsi2_{lo}_{e}"] = (events(cross_dn(r, np.full_like(r, lo)) & tr,
                                                  cross_up(r, np.full_like(r, 100 - lo)) & ~tr), True)
    m, sd = session_mean(D)
    for k in [1.5, 2.0, 2.5]:
        R[f"vwap_fade_{k}"] = (events(cross_dn(C, m - k * sd), cross_up(C, m + k * sd)), True)
    R["vwap_cross"] = (events(cross_up(C, m), cross_dn(C, m)), True)
    bm, bs = C.rolling(20).mean(), C.rolling(20).std()
    atr = pd.Series(D.atr)
    squeeze = ((bm + 2 * bs) < (ema(C, 20) + 1.5 * atr)) & ((bm - 2 * bs) > (ema(C, 20) - 1.5 * atr))
    for mbars in [6, 12]:
        sq_prev = squeeze.rolling(mbars).min().shift(1).fillna(0).astype(bool)
        R[f"squeeze_{mbars}"] = (events(cross_up(C, bm + 2 * bs) & sq_prev, cross_dn(C, bm - 2 * bs) & sq_prev), True)
    for k in [2.0, 3.0]:
        mid = ema(C, 20)
        R[f"keltner_break_{k}"] = (events(cross_up(C, mid + k * atr), cross_dn(C, mid - k * atr)), True)
    for f, s, sg in [(12, 26, 9), (24, 52, 18)]:
        mac = ema(C, f) - ema(C, s)
        sig = ema(mac, sg)
        R[f"macd_{f}_{s}"] = (events(cross_up(mac, sig), cross_dn(mac, sig)), True)
    for n, mult in [(10, 3.0), (20, 4.0)]:
        a = (pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()], axis=1).max(axis=1)
             .ewm(alpha=1 / n, adjust=False).mean().to_numpy())
        d = _supertrend(H.to_numpy(), L.to_numpy(), c, a, mult).astype(np.int64)
        ch = np.zeros(len(d), bool)
        ch[1:] = d[1:] != d[:-1]
        R[f"supertrend_{n}_{mult}"] = (events(ch & (d > 0), ch & (d < 0)), True)
    rng = (H - L).to_numpy()
    body = (C - D.O).to_numpy()
    for k in [2.0, 3.0]:
        big = rng > k * np.roll(D.atr, 1)
        R[f"burst_follow_{k}"] = (events(big & (body > 0), big & (body < 0)), True)
        R[f"burst_fade_{k}"] = (events(big & (body < 0), big & (body > 0)), True)
    # previous-day levels
    R["pdh_break"] = (events(cross_up(c, D.pdh), cross_dn(c, D.pdl)), True)
    swept_h = (H.to_numpy() > D.pdh) & (c < D.pdh)
    swept_l = (L.to_numpy() < D.pdl) & (c > D.pdl)
    R["pdh_sweep_fade"] = (events(swept_l, swept_h), True)

    # ---- session-specific rules (no session window multiplication)
    def first_per_day(sig):
        s = pd.Series(sig)
        nz = s != 0
        first = nz & (~nz).groupby(D.day_id).cumprod().shift(1, fill_value=1).astype(bool)
        first = nz & (nz.astype(int).groupby(D.day_id).cumsum() == 1)
        return np.where(first, sig, 0).astype(np.int8)

    def window_only(sig, a, b, hr):
        return np.where((hr > a + 1e-9) & (hr <= b + 1e-9), sig, 0).astype(np.int8)

    # Asian range 19:00-03:00, trade breaks/sweeps 03:00-10:00
    hi, lo, after, hr = range_levels(D, 19.0, 3.0)
    brk = events(cross_up(c, hi) & after, cross_dn(c, lo) & after)
    R["asia_range_break"] = (first_per_day(window_only(brk, 3.0, 10.0, hr)), False)
    sw = events((L.to_numpy() < lo) & (c > lo) & after, (H.to_numpy() > hi) & (c < hi) & after)
    R["asia_range_sweep_fade"] = (first_per_day(window_only(sw, 3.0, 10.0, hr)), False)
    # opening range breakouts and failed-break fades
    for name, start in [("lon", 3.0), ("comex", 8.0 + 20 / 60), ("eq", 9.5)]:
        for mins in [15, 30, 60]:
            end = start + mins / 60
            hi, lo, after, hr = range_levels(D, start, end)
            brk = events(cross_up(c, hi) & after, cross_dn(c, lo) & after)
            R[f"orb_{name}_{mins}"] = (first_per_day(window_only(brk, end, end + 2.0, hr)), False)
            if mins == 30:
                fail = events((L.to_numpy() < lo) & (c > lo) & after, (H.to_numpy() > hi) & (c < hi) & after)
                R[f"orb_fail_fade_{name}_{mins}"] = (first_per_day(window_only(fail, end, end + 2.0, hr)), False)
    # fixed-time entries (time-of-day drift), both directions
    for t in [20.0, 3.0, 8.5, 9.5, 10.0, 14.0]:
        hit = np.abs(D.hour - (t if t < 17 else t)) < 1e-6
        hit = hit if t < 17 else np.abs(D.hour - t) < 1e-6
        for dn, d in [("long", 1), ("short", -1)]:
            R[f"time_{int(t*100):04d}_{dn}"] = (np.where(hit, d, 0).astype(np.int8), False)
    # intraday momentum: sign of the move since the 18:00 reopen, entered at a fixed time
    day_open = D.O.groupby(D.day_id).transform("first").to_numpy()
    for t in [10.0, 12.0, 13.0]:
        hit = np.abs(D.hour - t) < 1e-6
        R[f"day_momentum_{int(t)}"] = (np.where(hit, np.sign(c - day_open), 0).astype(np.int8), False)
        R[f"day_reversal_{int(t)}"] = (np.where(hit, -np.sign(c - day_open), 0).astype(np.int8), False)
    return R


# ------------------------------------------------------------------ simulation
@njit(cache=True)
def _sim(sig_bar, sig_dir, entry_j, atr, day_id, eod, o, h, l, c,
         sl_atr, rr, hold, spread, slip, max_per_day, out_bar, out_r):
    """One position at a time, at most max_per_day entries per trading day. Returns #trades."""
    n1 = len(o)
    k = 0
    busy_until = -1
    cur_day = -1
    cnt = 0
    for s in range(len(sig_bar)):
        b = sig_bar[s]
        j = entry_j[b]
        if j < 0 or j <= busy_until:
            continue
        if day_id[b] != cur_day:
            cur_day = day_id[b]
            cnt = 0
        if cnt >= max_per_day:
            continue
        lim = min(j + hold, eod[j], n1 - 1)
        if lim <= j:
            continue
        d = sig_dir[s]
        sl_d = max(sl_atr * atr[b], 0.5)
        adj = 0.0 if d > 0 else spread       # long exits on bid, short exits on ask
        if d > 0:
            entry = o[j] + spread + slip
            sl, tp = entry - sl_d, entry + rr * sl_d
        else:
            entry = o[j] - slip
            sl, tp = entry + sl_d, entry - rr * sl_d
        ex = c[lim] + adj
        ej = lim
        for i in range(j, lim + 1):
            hi, lo, op = h[i] + adj, l[i] + adj, o[i] + adj
            if d > 0:
                if i > j and op <= sl:
                    ex = op - slip
                    ej = i
                    break
                if i > j and op >= tp:
                    ex = op
                    ej = i
                    break
                if lo <= sl:
                    ex = sl - slip
                    ej = i
                    break
                if hi >= tp:
                    ex = tp
                    ej = i
                    break
            else:
                if i > j and op >= sl:
                    ex = op + slip
                    ej = i
                    break
                if i > j and op <= tp:
                    ex = op
                    ej = i
                    break
                if hi >= sl:
                    ex = sl + slip
                    ej = i
                    break
                if lo <= tp:
                    ex = tp
                    ej = i
                    break
        out_bar[k] = b
        out_r[k] = d * (ex - entry) / sl_d
        k += 1
        busy_until = ej
        cnt += 1
    return k


@njit(parallel=True, cache=True)
def _sim_many(starts, ends, sig_bar_all, sig_dir_all, entry_j, atr, day_id, eod, o, h, l, c,
              p_sl, p_rr, p_hold, spread, slip, max_per_day, out_bar, out_r, out_n, cap):
    for q in prange(len(p_sl)):
        a, b = starts[q], ends[q]
        out_n[q] = _sim(sig_bar_all[a:b], sig_dir_all[a:b], entry_j, atr, day_id, eod, o, h, l, c,
                        p_sl[q], p_rr[q], p_hold[q], spread, slip, max_per_day,
                        out_bar[q * cap:(q + 1) * cap], out_r[q * cap:(q + 1) * cap])


def build_universe(cache=os.path.join(HERE, "data", "search_universe.pkl")):
    if os.path.isfile(cache):
        return pd.read_pickle(cache)
    D = Data(E.load_m1())
    rules = entry_rules(D)
    entries = {}
    for name, (sig, generic) in rules.items():
        sig = np.asarray(sig, np.int8)
        for flt in ("none", "trend"):
            s2 = sig if flt == "none" else np.where(sig == D.trend, sig, 0).astype(np.int8)
            if generic:
                for sess, (a, b) in SESSIONS.items():
                    hr = np.where(D.hour < 17.0, D.hour, D.hour)
                    if sess == "asia":
                        win = (D.hour > 19.0 + 1e-9) | (D.hour <= 2.0 + 1e-9)
                    else:
                        win = (hr > a + 1e-9) & (hr <= b + 1e-9)
                    entries[(name, sess, flt)] = np.where(win, s2, 0).astype(np.int8)
            else:
                entries[(name, "fixed", flt)] = s2
    U = {"D": D, "entries": entries}
    pd.to_pickle(U, cache)
    return U


def run_all(U, spread=SPREAD, slip=SLIP, exits=EXITS, random_dirs=False, seed=0):
    """Simulates every (entry rule x exit). Returns (catalog DataFrame, list of (bar, R) arrays)."""
    D = U["D"]
    keys = list(U["entries"])
    rng = np.random.default_rng(seed)
    bars, dirs, starts, ends = [], [], [], []
    pos = 0
    for kk in keys:
        s = U["entries"][kk]
        b = np.flatnonzero(s).astype(np.int64)
        d = s[b].astype(np.int64)
        if random_dirs:
            d = rng.choice(np.array([-1, 1]), len(b))
        bars.append(b)
        dirs.append(d)
        starts.append(pos)
        pos += len(b)
        ends.append(pos)
    sig_bar_all, sig_dir_all = np.concatenate(bars), np.concatenate(dirs)
    cat, p_sl, p_rr, p_hold, st, en = [], [], [], [], [], []
    for i, kk in enumerate(keys):
        for sl, rr, hold in exits:
            cat.append((*kk, sl, rr, hold))
            p_sl.append(sl)
            p_rr.append(rr)
            p_hold.append(hold)
            st.append(starts[i])
            en.append(ends[i])
    n = len(cat)
    cap = MAX_PER_DAY * (D.day_id.max() + 2)
    out_bar = np.full(n * cap, -1, np.int64)
    out_r = np.zeros(n * cap, np.float64)
    out_n = np.zeros(n, np.int64)
    _sim_many(np.array(st), np.array(en), sig_bar_all, sig_dir_all, D.entry_j, D.atr, D.day_id, D.eod,
              D.o, D.h, D.l, D.c, np.array(p_sl, float), np.array(p_rr, float), np.array(p_hold, np.int64),
              spread, slip, MAX_PER_DAY, out_bar, out_r, out_n, cap)
    catalog = pd.DataFrame(cat, columns=["rule", "session", "filter", "sl_atr", "rr", "hold"])
    catalog["family"] = catalog.rule.str.replace(r"_[\d.]+(_[\d.]+)*$", "", regex=True)
    trades = [(out_bar[q * cap:q * cap + out_n[q]], out_r[q * cap:q * cap + out_n[q]]) for q in range(n)]
    return catalog, trades


def stats(catalog, trades, D: Data):
    rows = []
    for (b, r) in trades:
        oos = D.oos[b]
        row = {}
        for tag, m in (("is", ~oos), ("oos", oos), ("all", np.ones(len(b), bool))):
            x = r[m]
            nn = len(x)
            mu = x.mean() if nn else np.nan
            sd = x.std(ddof=1) if nn > 1 else np.nan
            row[f"n_{tag}"] = nn
            row[f"r_{tag}"] = mu
            row[f"t_{tag}"] = mu / sd * np.sqrt(nn) if nn > 1 and sd > 0 else np.nan
            row[f"win_{tag}"] = (x > 0).mean() if nn else np.nan
        rows.append(row)
    return pd.concat([catalog, pd.DataFrame(rows)], axis=1)


def daily_matrix(trades, D: Data, mask_days):
    """Days x strategies matrix of summed R per trading day (0 when flat)."""
    nd = D.day_id.max() + 1
    M = np.zeros((nd, len(trades)), np.float32)
    for q, (b, r) in enumerate(trades):
        np.add.at(M[:, q], D.day_id[b], r)
    return M[mask_days]


def stationary_bootstrap_idx(n, mean_block, rng):
    idx = np.empty(n, np.int64)
    idx[0] = rng.integers(n)
    p = 1.0 / mean_block
    jumps = rng.random(n) < p
    starts = rng.integers(0, n, n)
    for i in range(1, n):
        idx[i] = starts[i] if jumps[i] else (idx[i - 1] + 1) % n
    return idx


def reality_check(M, n_boot=500, mean_block=5, seed=0):
    """White's Reality Check / Hansen SPA (studentized, consistent centering) on a days x strategies matrix.
    Returns (best column, observed max statistic, RC p-value, SPA p-value)."""
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    mu = M.mean(0)
    sd = M.std(0, ddof=1)
    sd[sd == 0] = np.inf
    stat = np.sqrt(n) * mu / sd
    T = np.nanmax(stat)
    best = int(np.nanargmax(stat))
    # SPA consistent centering: only strategies that are not clearly bad keep their mean
    thr = -np.sqrt(2 * np.log(np.log(n))) * sd / np.sqrt(n)
    mu_spa = np.where(mu >= thr, mu, 0.0)
    rc, spa = [], []
    for _ in range(n_boot):
        ix = stationary_bootstrap_idx(n, mean_block, rng)
        mb = M[ix].mean(0)
        rc.append(np.nanmax(np.sqrt(n) * (mb - mu) / sd))
        spa.append(np.nanmax(np.sqrt(n) * (mb - mu_spa) / sd))
    return best, T, float(np.mean(np.array(rc) >= T)), float(np.mean(np.array(spa) >= T))


# ------------------------------------------------------------------ funded challenge
@njit(cache=True)
def _challenge(week_of_trade, day_of_trade, r, n_weeks, risk, daily_limit, week_limit, target, buffer):
    """Each week is a fresh first week: +target before the limits. Returns (passed, breached) per week."""
    passed = np.zeros(n_weeks, np.bool_)
    breached = np.zeros(n_weeks, np.bool_)
    wk, dy = -1, -1
    wpnl = dpnl = 0.0
    done = False
    for i in range(len(r)):
        if week_of_trade[i] != wk:
            wk = week_of_trade[i]
            wpnl = 0.0
            dy = -1
            done = False
        if day_of_trade[i] != dy:
            dy = day_of_trade[i]
            dpnl = 0.0
        if done:
            continue
        if dpnl - risk - buffer < -daily_limit or wpnl - risk - buffer < -week_limit:
            continue
        p = r[i] * risk
        wpnl += p
        dpnl += p
        if dpnl <= -daily_limit or wpnl <= -week_limit:
            breached[wk] = True
            done = True
        elif wpnl >= target:
            passed[wk] = True
            done = True
    return passed, breached


def week_index(D: Data):
    wk = (D.day - pd.to_timedelta(D.day.dt.weekday, unit="D"))
    return pd.factorize(wk)[0].astype(np.int64), wk


def challenge(b, r, D: Data, wk_id, risk, mask=None, daily_limit=200.0, week_limit=400.0, target=600.0, buffer=15.0):
    if mask is not None:
        b, r = b[mask], r[mask]
    w = wk_id[b]
    n_weeks = int(wk_id.max()) + 1
    p, br = _challenge(w, D.day_id[b], r, n_weeks, float(risk), daily_limit, week_limit, target, buffer)
    return p, br
