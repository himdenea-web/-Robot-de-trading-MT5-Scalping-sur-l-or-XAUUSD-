"""Two-phase funded challenge with no time limit (Stellar Lite style, 5,000 USD account).

Phase 1: reach +400 before the static floor (-400 from the start balance).
Phase 2: new account, reach +200 before the same floor.
Both phases: daily loss at most 200; total open risk at most 150; every trade has a stop.
Sizing is checked worst case before each trade (like goldbot): the trade risk is reduced to the
room left (day and floor) minus a buffer, and a trade is skipped if that room is below
min_risk. A challenge is "stuck" (counted as failed) when the floor room is below min_risk.

    res = simulate(day_id, R, starts, risk=100)   # one row per start: outcome, days used
"""
import numpy as np
import pandas as pd
from numba import njit

OUTCOME = {0: "unresolved", 1: "passed", 2: "failed"}


@njit(cache=True)
def _phase(day_id, r, i0, risk, target, floor, daily, buffer, min_risk):
    """Runs one phase from trade i0. Returns (outcome, next trade index, last day)."""
    bal = 0.0
    dpnl = 0.0
    cur = -1
    n = len(r)
    i = i0
    while i < n:
        if day_id[i] != cur:
            cur = day_id[i]
            dpnl = 0.0
        room_total = bal + floor - buffer
        if room_total < min_risk:
            return 2, i, cur
        room = min(room_total, dpnl + daily - buffer)
        x = min(risk, room)
        if x >= min_risk:
            p = r[i] * x
            bal += p
            dpnl += p
            if bal <= -floor or dpnl <= -daily:
                return 2, i + 1, cur
            if bal >= target:
                return 1, i + 1, cur
        i += 1
    return 0, n, cur


@njit(cache=True)
def _run(day_id, r, start_idx, risk, t1, t2, floor, daily, buffer, min_risk, out):
    for s in range(len(start_idx)):
        i0 = start_idx[s]
        o1, i1, d1 = _phase(day_id, r, i0, risk, t1, floor, daily, buffer, min_risk)
        out[s, 0] = o1
        out[s, 1] = d1 - day_id[i0] if i0 < len(r) else -1
        if o1 != 1:
            out[s, 2] = o1
            out[s, 3] = -1
            continue
        o2, i2, d2 = _phase(day_id, r, i1, risk, t2, floor, daily, buffer, min_risk)
        out[s, 2] = o2
        out[s, 3] = d2 - day_id[i0]


def simulate(day_id, r, start_days, risk, t1=400.0, t2=200.0, floor=400.0, daily=200.0,
             buffer=15.0, min_risk=25.0):
    """day_id: trading-day number of each trade (sorted); r: R multiple of each trade;
    start_days: day numbers at which a challenge starts. Returns a DataFrame per start."""
    day_id = np.asarray(day_id, np.int64)
    r = np.asarray(r, np.float64)
    start_idx = np.searchsorted(day_id, np.asarray(start_days, np.int64)).astype(np.int64)
    out = np.zeros((len(start_idx), 4), np.int64)
    _run(day_id, r, start_idx, float(min(risk, 150.0)), t1, t2, floor, daily, buffer, min_risk, out)
    df = pd.DataFrame(out, columns=["p1", "p1_days", "p2", "total_days"])
    df["start_day"] = np.asarray(start_days)
    df["result"] = np.where(df.p2 == 1, "passed",
                            np.where((df.p1 == 2) | (df.p2 == 2), "failed", "unresolved"))
    return df


def summarize(df):
    res = df[df.result != "unresolved"]
    n = len(res)
    p = (res.result == "passed").mean() if n else np.nan
    p1 = (df.p1 == 1).sum() / max(1, (df.p1 != 0).sum())
    days = df.loc[df.result == "passed", "total_days"]
    return {"starts": len(df), "resolved": n, "pass_both": p, "pass_phase1": p1,
            "median_days_to_pass": float(days.median()) if len(days) else np.nan}
