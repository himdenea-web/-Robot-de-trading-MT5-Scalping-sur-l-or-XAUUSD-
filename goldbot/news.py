"""Major announcement calendar and blackout windows.

Some funded accounts count only 40% of gains realized within 5 minutes before or after a
major announcement (losses count 100%). The engine uses this calendar to
  * refuse new setups from before_min before a release until after_min after it, and
  * close positions that are in profit just before the blackout starts (close_winners).

Sources (GB_NEWS_SOURCE):
    off             no calendar
    forexfactory    the public weekly calendar feed, high-impact events of GB_NEWS_CURRENCIES
    file            a CSV you maintain (GB_NEWS_FILE): columns time, title; times without an
                    offset are New York local time
    fixed:08:30,10:00,14:00   those New York times on every weekday (conservative, offline;
                    useful in backtests)
All times are handled as naive UTC, like the rest of goldbot.
"""
import json
import time as _time
from datetime import time as dtime

import pandas as pd

from goldbot.config import NewsConfig

NY = "America/New_York"
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def _to_utc_naive(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY, ambiguous=False, nonexistent="shift_forward")
    return ts.tz_convert("UTC").tz_localize(None)


class NewsCalendar:
    def __init__(self, cfg: NewsConfig, log=print, fetch=None):
        self.cfg = cfg
        self.log = log
        self._fetch = fetch                    # injectable for tests: () -> list of dicts
        self._events: list[tuple[pd.Timestamp, str]] = []
        self._loaded_at = 0.0
        self._fixed: list[dtime] = []
        if cfg.source.startswith("fixed:"):
            self._fixed = [dtime.fromisoformat(x.strip()) for x in cfg.source[6:].split(",") if x.strip()]

    @property
    def enabled(self) -> bool:
        return self.cfg.source != "off"

    # ------------------------------------------------------------------ loading
    def _load(self):
        src = self.cfg.source
        if src == "file":
            df = pd.read_csv(self.cfg.file)
            self._events = sorted((_to_utc_naive(t), str(n)) for t, n in zip(df["time"], df["title"]))
        elif src == "forexfactory":
            rows = self._fetch() if self._fetch else self._download()
            ev = []
            for r in rows:
                if str(r.get("impact", "")).lower() != "high":
                    continue
                if str(r.get("country", "")).upper() not in self.cfg.currencies:
                    continue
                try:
                    ev.append((_to_utc_naive(r["date"]), str(r.get("title", ""))))
                except (KeyError, ValueError):
                    continue
            self._events = sorted(ev)
        self._loaded_at = _time.time()

    def _download(self):
        import requests
        resp = requests.get(FF_URL, timeout=15, headers={"User-Agent": "goldbot"})
        resp.raise_for_status()
        return json.loads(resp.text)

    def refresh(self, force=False):
        if self.cfg.source in ("off",) or self._fixed:
            return
        if not force and _time.time() - self._loaded_at < self.cfg.refresh_hours * 3600:
            return
        try:
            self._load()
            self.log(f"news calendar: {len(self._events)} events loaded from {self.cfg.source}")
        except Exception as e:  # keep the previous calendar; retry in 15 minutes
            self._loaded_at = _time.time() - self.cfg.refresh_hours * 3600 + 900
            self.log(f"news calendar refresh failed ({e}); keeping {len(self._events)} known events")

    # ------------------------------------------------------------------ queries
    def events_between(self, start, end) -> list[tuple[pd.Timestamp, str]]:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if self._fixed:
            out = []
            for day in pd.date_range((start - pd.Timedelta(days=1)).normalize(),
                                     (end + pd.Timedelta(days=1)).normalize(), freq="D"):
                if day.weekday() >= 5:
                    continue
                for t in self._fixed:
                    ts = _to_utc_naive(pd.Timestamp.combine(day.date(), t))
                    if start <= ts <= end:
                        out.append((ts, f"fixed {t:%H:%M} New York"))
            return out
        return [(t, n) for t, n in self._events if start <= t <= end]

    def blackout(self, now):
        """The event whose blackout window contains now, or None."""
        now = pd.Timestamp(now)
        ev = self.events_between(now - pd.Timedelta(minutes=self.cfg.after_min),
                                 now + pd.Timedelta(minutes=self.cfg.before_min))
        return ev[0] if ev else None

    def next_within(self, now, minutes):
        """The next event starting its blackout within `minutes` from now, or None."""
        now = pd.Timestamp(now)
        ev = self.events_between(now, now + pd.Timedelta(minutes=self.cfg.before_min + minutes))
        return ev[0] if ev else None
