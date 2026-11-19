"""Multi-agent entry committee (optional, veto-only).

Three analysts look at the same setup from different angles, in parallel, all through the
single configured provider. A chair then reads their opinions and makes the final call.

Hard guarantees, enforced in code (not by prompts):
  * the committee can only APPROVE or REJECT the exact setup; it never creates a trade or
    changes entry, stop, target or size
  * the funded-account limits are checked by the risk manager before the committee runs,
    and the committee cannot override them
  * if the provider fails or times out, `fail_mode` decides: "open" trades the tested
    strategy as backtested, "closed" skips the trade
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field

from goldbot.judge.providers import ProviderError, make_provider

VOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "approve": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["approve", "confidence", "reason"],
    "additionalProperties": False,
}

COMMON = (
    "You are part of a review committee for an automated XAUUSD (spot gold) trading system. "
    "The system has already found a rule-based setup and checked the account's risk limits. "
    "Your only power is to approve or reject this exact setup; you cannot change its levels "
    "or size. The base strategy is a backtested opening range breakout, so reject only when "
    "the data in front of you gives a concrete reason. Use only the numbers provided, do not "
    "assume news or prices you were not given. Reply with JSON: approve (boolean), "
    "confidence (0 to 1), reason (one short sentence)."
)

AGENTS = {
    "trend_analyst": COMMON + (
        " Your angle: trend alignment. Judge whether the breakout direction agrees with the "
        "daily trend (trend_d1: +1 up, -1 down), where price sits versus the previous close, "
        "and whether the opening range broke with or against the prevailing move."
    ),
    "volatility_analyst": COMMON + (
        " Your angle: volatility and structure. Judge whether the opening range is a sensible "
        "size versus the daily ATR (range_atr_ratio: very small ranges often fake out, very "
        "large ones leave little room), whether the stop distance is reasonable versus ATR, "
        "and whether the spread is a small share of the risk."
    ),
    "risk_officer": COMMON + (
        " Your angle: account risk. You see the account's position against its funded-account "
        "rules (daily loss limit, first-week loss limit, weekly profit target). The hard limits "
        "are already enforced; judge whether taking this trade now is prudent given how close "
        "the account is to a limit or to the weekly target, and recent losses."
    ),
}

CHAIR = COMMON + (
    " You are the chair. You receive the setup and the three analysts' opinions. Weigh them "
    "and make the final decision. If analysts disagree, explain which argument decided it."
)


@dataclass
class Decision:
    approved: bool
    reason: str
    votes: dict = field(default_factory=dict)
    error: str = ""
    latency_s: float = 0.0
    provider: str = ""

    def summary(self) -> str:
        v = ", ".join(f"{k}={'yes' if x.get('approve') else 'no'}" for k, x in self.votes.items())
        head = "APPROVED" if self.approved else "REJECTED"
        tail = f" [{self.error}]" if self.error else ""
        return f"{head} by committee ({v}): {self.reason}{tail}"


def build_context(setup, account: dict, market_snapshot: dict) -> dict:
    """Numbers the agents see. No dates or prices outside what the system measured."""
    risk_unit = setup.risk_per_unit
    return {
        "setup": {
            "side": setup.side,
            "strategy": setup.strategy,
            "entry": round(setup.entry, 2),
            "stop": round(setup.sl, 2),
            "target": None if setup.tp is None else round(setup.tp, 2),
            "stop_distance_usd": round(risk_unit, 2),
            "reward_risk": None if setup.tp is None else round(abs(setup.tp - setup.entry) / risk_unit, 2),
            **{k: v for k, v in setup.context.items() if isinstance(v, (int, float, str, bool))},
        },
        "market": market_snapshot,
        "account": account,
    }


class Committee:
    def __init__(self, cfg_judge, provider=None):
        self.cfg = cfg_judge
        self.provider = provider or make_provider(
            cfg_judge.provider, api_key=cfg_judge.api_key, model=cfg_judge.model,
            base_url=cfg_judge.base_url, timeout_s=cfg_judge.timeout_s)

    def _ask(self, system, payload):
        out = self.provider.complete_json(system, json.dumps(payload, sort_keys=True), VOTE_SCHEMA)
        return {"approve": bool(out.get("approve")),
                "confidence": float(out.get("confidence", 0) or 0),
                "reason": str(out.get("reason", ""))[:300]}

    def evaluate(self, setup, account: dict, market_snapshot: dict) -> Decision:
        t0 = time.monotonic()
        ctx = build_context(setup, account, market_snapshot)
        fail_open = self.cfg.fail_mode == "open"
        pool = ThreadPoolExecutor(max_workers=len(AGENTS))
        try:
            futs = {name: pool.submit(self._ask, prompt, ctx) for name, prompt in AGENTS.items()}
            deadline = t0 + self.cfg.timeout_s
            votes = {name: f.result(timeout=max(0.1, deadline - time.monotonic()))
                     for name, f in futs.items()}
            n_yes = sum(v["approve"] for v in votes.values())
            if n_yes < self.cfg.min_approvals:
                return Decision(False, f"only {n_yes} of {len(votes)} analysts approved",
                                votes, latency_s=time.monotonic() - t0,
                                provider=self.provider.describe())
            if not self.cfg.use_chair:
                return Decision(True, f"{n_yes} of {len(votes)} analysts approved", votes,
                                latency_s=time.monotonic() - t0, provider=self.provider.describe())
            chair = self._ask(CHAIR, {**ctx, "analysts": votes})
            votes["chair"] = chair
            return Decision(chair["approve"], chair["reason"], votes,
                            latency_s=time.monotonic() - t0, provider=self.provider.describe())
        except (ProviderError, FutTimeout, ValueError, TypeError) as e:
            err = f"committee unavailable ({type(e).__name__}: {e}); fail mode {self.cfg.fail_mode}"
            return Decision(fail_open, "fail-open: trading the tested strategy" if fail_open
                            else "fail-closed: skipping", {}, error=err,
                            latency_s=time.monotonic() - t0, provider=self.provider.describe())
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
