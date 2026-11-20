"""Small JSON state file so a restart does not lose the context of live orders.

MT5 keeps the orders and positions, but not why they exist (time exit, expiry, invalidation
level, planned risk). The broker saves each setup it sends here, keyed by MT5 ticket, and
reads them back on start to adopt its own leftover orders and positions.
"""
import json
import os
import tempfile

import pandas as pd

from goldbot.models import Setup

_TS = ("created", "expires", "exit_by")


def setup_to_dict(s: Setup) -> dict:
    d = {k: getattr(s, k) for k in ("strategy", "group", "direction", "entry", "sl", "tp",
                                     "invalidate")}
    d.update({k: pd.Timestamp(getattr(s, k)).isoformat() for k in _TS})
    d["context"] = {k: v for k, v in (s.context or {}).items() if isinstance(v, (int, float, str, bool))}
    return d


def setup_from_dict(d: dict) -> Setup:
    return Setup(strategy=d["strategy"], group=d["group"], direction=int(d["direction"]),
                 entry=float(d["entry"]), sl=float(d["sl"]),
                 tp=None if d.get("tp") is None else float(d["tp"]),
                 created=pd.Timestamp(d["created"]), expires=pd.Timestamp(d["expires"]),
                 exit_by=pd.Timestamp(d["exit_by"]), context=d.get("context", {}),
                 invalidate=None if d.get("invalidate") is None else float(d["invalidate"]))


class StateStore:
    def __init__(self, path: str | None):
        self.path = path
        self.data = {"orders": {}}
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except (OSError, ValueError):
                self.data = {"orders": {}}
        self.data.setdefault("orders", {})

    def _save(self):
        if not self.path:
            return
        d = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".goldbot_state_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)              # atomic on Windows and POSIX

    def remember(self, ticket: int, setup: Setup, lots: float, planned_risk: float):
        self.data["orders"][str(ticket)] = {"setup": setup_to_dict(setup), "lots": lots,
                                            "planned_risk": planned_risk}
        self._save()

    def get(self, ticket: int):
        r = self.data["orders"].get(str(ticket))
        if r is None:
            return None
        return setup_from_dict(r["setup"]), float(r["lots"]), float(r["planned_risk"])

    def forget(self, ticket: int):
        if self.data["orders"].pop(str(ticket), None) is not None:
            self._save()

    def prune(self, keep: set):
        before = len(self.data["orders"])
        self.data["orders"] = {k: v for k, v in self.data["orders"].items() if k in {str(x) for x in keep}}
        if len(self.data["orders"]) != before:
            self._save()
