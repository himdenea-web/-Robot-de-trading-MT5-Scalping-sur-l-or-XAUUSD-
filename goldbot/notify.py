"""Telegram notifications and the CSV journal."""
import csv
import datetime as dt
import os

import requests

JOURNAL_FIELDS = ["time_utc", "event", "group", "side", "price", "sl", "tp", "lots", "risk",
                  "pnl", "r", "reason", "latency"]


class Telegram:
    def __init__(self, token: str, chat_id: str, log=print):
        self.token, self.chat_id, self.log = token, chat_id, log

    def send(self, text: str):
        self.log(text)
        if not self.token or not self.chat_id:
            return
        try:
            requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                          data={"chat_id": self.chat_id, "text": text}, timeout=10)
        except requests.RequestException as e:
            self.log(f"[telegram] send failed: {e}")


class Journal:
    """Append-only CSV with a fixed header (one row per event)."""

    def __init__(self, path: str):
        self.path = path
        if not os.path.isfile(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(JOURNAL_FIELDS)

    def write(self, event: str, **kw):
        row = {"time_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
               "event": event, **kw}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(k, "") for k in JOURNAL_FIELDS])
