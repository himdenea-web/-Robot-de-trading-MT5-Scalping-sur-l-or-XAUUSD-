"""Grid search with walk-forward re-optimization for the New York opening range breakout.

    python orb_wf.py            # builds data/orb_wf_grid.pkl (about 5 minutes) then reports

Each quarter from 2023 on, the settings with the best result over the previous train window
are chosen and traded, unseen, in that quarter. The stitched quarters are the walk-forward
result. It is compared with the current fixed settings, the average setting, and random picks.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import orb as O  # noqa: E402

GRID_FILE = os.path.join(HERE, "data", "orb_wf_grid.pkl")
GRID = {
    "range_min": [15, 30, 45, 60, 90],
    "window_min": [60, 120, 180],
    "stop_frac": [0.25, 0.33, 0.5, 0.75, 1.0],
    "target_r": [1.0, 1.5, 2.0, 3.0, 4.0],
    "trend_span": [0, 5, 10, 20, 50, 100],      # 0 = no trend filter
}
CURRENT = dict(range_min=60, window_min=120, stop_frac=0.5, target_r=3.0, trend_span=20)


def build_grid():
    if os.path.isfile(GRID_FILE):
        return pd.read_pickle(GRID_FILE)
    m = O.M1(E.load_m1())
    keys = list(GRID)
    out = {}
    for vals in itertools.product(*GRID.values()):
        k = dict(zip(keys, vals))
        p = O.OrbParams(session="ny_comex", range_min=k["range_min"], window_min=k["window_min"],
                        stop="range", stop_frac=k["stop_frac"], target_r=k["target_r"],
                        trend_filter=k["trend_span"] > 0, trend_span=k["trend_span"] or 20)
        t = O.run_orb(m, p)
        out[vals] = pd.DataFrame({"day": pd.to_datetime(t["day"]), "R": t["R"].to_numpy(),
                                  "risk": t["risk"].to_numpy()})
    pd.to_pickle(out, GRID_FILE)
    return out


def walk_forward(grid, train_months=12, score="mean", min_trades=30, start="2023-01-01", end="2026-12-01"):
    quarters = pd.date_range(start, end, freq="QS")
    picks, oos = [], []
    for q0, q1 in zip(quarters[:-1], quarters[1:]):
        tr0 = q0 - pd.DateOffset(months=train_months)
        best, best_s = None, -np.inf
        for k, t in grid.items():
            x = t.R[(t.day >= tr0) & (t.day < q0)]
            if len(x) < min_trades:
                continue
            s = x.mean() if score == "mean" else x.mean() / x.std(ddof=1) * np.sqrt(len(x))
            if s > best_s:
                best, best_s = k, s
        t = grid[best]
        seg = t[(t.day >= q0) & (t.day < q1)]
        picks.append((q0.date(), best, best_s, len(seg), seg.R.mean()))
        oos.append(seg.R.to_numpy())
    return pd.DataFrame(picks, columns=["quarter", "settings", "train_score", "n", "R_test"]), np.concatenate(oos)


def segment(grid, key, start="2023-01-01"):
    t = grid[key]
    return t.R[t.day >= start].to_numpy()


def describe(x):
    return {"trades": len(x), "mean R": x.mean(), "t": x.mean() / x.std(ddof=1) * np.sqrt(len(x)),
            "total R": x.sum(), "win rate": (x > 0).mean()}


if __name__ == "__main__":
    grid = build_grid()
    cur = tuple(CURRENT[k] for k in GRID)
    rows = {"current fixed settings": describe(segment(grid, cur))}
    for tm in (12, 24):
        for sc in ("mean", "t"):
            picks, x = walk_forward(grid, tm, sc)
            rows[f"walk-forward, {tm}m train, best {sc}"] = describe(x)
    allm = np.array([segment(grid, k).mean() for k in grid])
    rows["average of all 2,250 settings"] = {"mean R": allm.mean()}
    print(pd.DataFrame(rows).T.round(3))
    print("share of settings with positive mean R since 2023:", (allm > 0).mean().round(3))
