"""Evaluate the LLM committee on historical setups (costs money: one setup = 4 LLM calls).

Runs the engine on the holdout period with the committee OFF to collect every setup the
risk manager accepted, then asks the configured provider's committee about a random sample
of them, and compares the outcome (R) of setups it approved with setups it rejected.
Answers are cached in data/llm_replay_cache.json, so a rerun does not bill again.

    set GB_LLM_PROVIDER=anthropic & set GB_LLM_API_KEY=...   (or put them in .env)
    python backtest/llm_replay.py --n 60 --start 2025-01-01

Note: the model sees only the setup's numbers (no dates). It may still have general
knowledge of gold's history, so treat a good result as a hypothesis for forward testing.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import engine as E  # noqa: E402
from goldbot.backtest import run_backtest  # noqa: E402
from goldbot.config import Config  # noqa: E402
from goldbot.judge import Committee  # noqa: E402
from goldbot.judge.committee import Decision  # noqa: E402

CACHE = os.path.join(HERE, "data", "llm_replay_cache.json")


class RecordingCommittee:
    """Approves everything and records each setup with the context the real committee gets."""

    def __init__(self):
        self.seen = []

    def evaluate(self, setup, account, snap):
        self.seen.append((setup, account, snap))
        return Decision(True, "recorded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="setups to send (4 LLM calls each)")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--env", default=os.path.join(os.path.dirname(HERE), ".env"))
    a = ap.parse_args()

    cfg = Config.from_env(a.env)
    if cfg.judge.provider == "fake":
        print("note: using the offline fake provider")
    m1 = E.load_m1()
    cfg.judge.enabled = True
    rec = RecordingCommittee()
    res = run_backtest(cfg, m1, start=a.start, committee=rec)
    trades = res["trades"].set_index("group")
    pool = [(s, acc, snap) for s, acc, snap in rec.seen if s.group in trades.index]
    rng = np.random.default_rng(a.seed)
    sample = [pool[i] for i in rng.choice(len(pool), min(a.n, len(pool)), replace=False)]
    print(f"{len(pool)} filled setups since {a.start}; sending {len(sample)} to "
          f"{cfg.judge.provider} ({len(sample) * 4} calls max)")

    cache = json.load(open(CACHE)) if os.path.isfile(CACHE) else {}
    committee = Committee(cfg.judge)
    rows = []
    for k, (s, acc, snap) in enumerate(sample, 1):
        key = hashlib.sha1(f"{committee.provider.describe()}|{s.group}|{s.side}".encode()).hexdigest()
        if key not in cache:
            d = committee.evaluate(s, acc, snap)
            cache[key] = {"approved": d.approved, "reason": d.summary(), "error": d.error}
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            json.dump(cache, open(CACHE, "w"), indent=1)
        c = cache[key]
        rows.append({"group": s.group, "side": s.side, "R": float(trades.loc[s.group, "R"]),
                     "approved": c["approved"], "error": c["error"]})
        print(f"[{k}/{len(sample)}] {s.group} {s.side}: {'APPROVED' if c['approved'] else 'rejected'}"
              f"{' (' + c['error'] + ')' if c['error'] else ''}")
    df = pd.DataFrame(rows)
    ok = df[df.error == ""]
    print("\nResult (setups without provider errors):")
    print(ok.groupby("approved")["R"].agg(["count", "mean", "sum"]).round(3))
    if ok.approved.nunique() == 2:
        a_, r_ = ok[ok.approved].R, ok[~ok.approved].R
        diff = a_.mean() - r_.mean()
        boot = [rng.choice(a_, len(a_)).mean() - rng.choice(r_, len(r_)).mean() for _ in range(5000)]
        print(f"approved minus rejected: {diff:+.3f} R, 95% CI [{np.percentile(boot, 2.5):+.3f}, "
              f"{np.percentile(boot, 97.5):+.3f}]")
        print("Keep the committee only if this interval is clearly above zero on enough setups.")


if __name__ == "__main__":
    main()
