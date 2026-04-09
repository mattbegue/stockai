"""
Earnings surprise data for PEAD (Post-Earnings Announcement Drift) strategy.

Fetches EPS estimate vs actual + surprise % via yfinance and caches locally.
This is distinct from EarningsCalendar (which only stores dates for risk filtering).

Entry timing logic:
  - AMC report (time >= 16:00 ET): enter at NEXT trading day open
  - BMO report (time < 09:30 ET): enter at SAME day open
  - During-market report: rare, treat as AMC → next day
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "earnings_surprise_cache"


@dataclass
class EarningsEvent:
    """A single historical earnings event with surprise data."""

    ticker: str
    report_date: date          # Date the earnings were reported
    entry_date: date           # First trading day to enter (next day if AMC, same if BMO)
    eps_estimate: float
    eps_actual: float
    surprise_pct: float        # (actual - estimate) / |estimate| * 100
    is_bmo: bool               # True = before market open, False = after market close

    @property
    def is_beat(self) -> bool:
        return self.surprise_pct > 0

    @property
    def is_miss(self) -> bool:
        return self.surprise_pct < 0


class EarningsSurpriseData:
    """
    Per-ticker earnings surprise registry with local JSON cache.

    Provides O(1) lookup: given a date, which tickers just reported and
    what was their EPS surprise?

    Usage::

        surprise_data = EarningsSurpriseData()
        surprise_data.load(tickers)

        # Get all events where entry_date == today
        events = surprise_data.events_for_entry_date(some_date)
        for event in events:
            if event.surprise_pct > 5.0:
                # Take the trade
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # ticker → list of EarningsEvent, sorted by report_date
        self._events: dict[str, list[EarningsEvent]] = {}
        # entry_date → list of EarningsEvent (built on demand)
        self._by_entry_date: dict[date, list[EarningsEvent]] | None = None

    # ------------------------------------------------------------------
    # Loading / caching
    # ------------------------------------------------------------------

    def load(
        self,
        tickers: list[str],
        n_quarters: int = 25,
        force_refresh: bool = False,
        rate_limit_delay: float = 0.3,
    ) -> None:
        """
        Fetch and cache earnings surprise data for all tickers.

        Args:
            tickers: List of stock tickers to fetch
            n_quarters: Quarters of history to fetch (default 25 ~ 6 years)
            force_refresh: Ignore cache and re-fetch from yfinance
            rate_limit_delay: Seconds between yfinance requests
        """
        newly_fetched = 0
        for ticker in tickers:
            if ticker in self._events and not force_refresh:
                continue
            cached = self._load_cache(ticker)
            if cached is not None and not force_refresh:
                self._events[ticker] = cached
            else:
                fetched = self._fetch(ticker, n_quarters)
                self._events[ticker] = fetched
                self._save_cache(ticker, fetched)
                newly_fetched += 1
                if newly_fetched % 10 == 0:
                    time.sleep(rate_limit_delay)

        self._by_entry_date = None  # Invalidate index
        loaded = sum(1 for t in tickers if self._events.get(t))
        logger.info(f"EarningsSurpriseData: loaded {loaded}/{len(tickers)} tickers")

    def _fetch(self, ticker: str, n_quarters: int) -> list[EarningsEvent]:
        """Fetch earnings surprise events from yfinance."""
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            df = t.get_earnings_dates(limit=n_quarters)
            if df is None or df.empty:
                return []

            events = []
            for ts, row in df.iterrows():
                eps_estimate = row.get("EPS Estimate", np.nan)
                eps_actual = row.get("Reported EPS", np.nan)
                surprise_pct = row.get("Surprise(%)", np.nan)

                # Skip future (unreported) earnings
                if pd.isna(eps_actual):
                    continue
                # Skip if estimate is missing or zero (can't compute surprise)
                if pd.isna(eps_estimate) or pd.isna(surprise_pct):
                    continue

                report_date = ts.date() if hasattr(ts, "date") else ts

                # Determine BMO vs AMC from the timestamp hour (UTC-based)
                # yfinance reports as: 08:00 ET = BMO, 16:00 ET = AMC
                # In UTC: 08:00 ET = 12:00 or 13:00 UTC; 16:00 ET = 20:00 or 21:00 UTC
                hour_et = ts.hour if ts.tzinfo is None else ts.astimezone(
                    __import__("datetime").timezone(__import__("datetime").timedelta(hours=-5))
                ).hour
                is_bmo = hour_et < 12  # Before noon ET = BMO

                # Entry date: same day if BMO, next business day if AMC
                if is_bmo:
                    entry_date = report_date
                else:
                    entry_date = _next_bday(report_date)

                events.append(EarningsEvent(
                    ticker=ticker,
                    report_date=report_date,
                    entry_date=entry_date,
                    eps_estimate=float(eps_estimate),
                    eps_actual=float(eps_actual),
                    surprise_pct=float(surprise_pct),  # yfinance Surprise(%) is already in pct (6.25 = 6.25%)
                    is_bmo=is_bmo,
                ))

            return sorted(events, key=lambda e: e.report_date)

        except Exception as exc:
            logger.debug(f"EarningsSurpriseData: failed to fetch {ticker}: {exc}")
            return []

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str) -> Path:
        safe = ticker.replace(".", "_")
        return self._cache_dir / f"{safe}.json"

    def _load_cache(self, ticker: str) -> Optional[list[EarningsEvent]]:
        path = self._cache_path(ticker)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            events = []
            for r in raw["events"]:
                events.append(EarningsEvent(
                    ticker=r["ticker"],
                    report_date=date.fromisoformat(r["report_date"]),
                    entry_date=date.fromisoformat(r["entry_date"]),
                    eps_estimate=r["eps_estimate"],
                    eps_actual=r["eps_actual"],
                    surprise_pct=r["surprise_pct"],
                    is_bmo=r["is_bmo"],
                ))
            return events
        except Exception:
            return None

    def _save_cache(self, ticker: str, events: list[EarningsEvent]) -> None:
        path = self._cache_path(ticker)
        try:
            records = [
                {
                    "ticker": e.ticker,
                    "report_date": e.report_date.isoformat(),
                    "entry_date": e.entry_date.isoformat(),
                    "eps_estimate": e.eps_estimate,
                    "eps_actual": e.eps_actual,
                    "surprise_pct": e.surprise_pct,
                    "is_bmo": e.is_bmo,
                }
                for e in events
            ]
            path.write_text(json.dumps({"events": records}))
        except Exception as exc:
            logger.debug(f"EarningsSurpriseData: cache write failed for {ticker}: {exc}")

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def _build_entry_date_index(self) -> None:
        """Build entry_date → [EarningsEvent] lookup."""
        self._by_entry_date = {}
        for events in self._events.values():
            for event in events:
                self._by_entry_date.setdefault(event.entry_date, []).append(event)

    def events_for_entry_date(self, entry_date: date) -> list[EarningsEvent]:
        """All tickers whose PEAD entry falls on this date."""
        if self._by_entry_date is None:
            self._build_entry_date_index()
        return self._by_entry_date.get(entry_date, [])

    def get_event(self, ticker: str, report_date: date) -> Optional[EarningsEvent]:
        """Look up a specific earnings event by ticker and report date."""
        for event in self._events.get(ticker, []):
            if event.report_date == report_date:
                return event
        return None

    def all_events(self, ticker: str) -> list[EarningsEvent]:
        """All historical events for a ticker, sorted by report date."""
        return self._events.get(ticker, [])

    def summary(self) -> dict:
        """Quick stats for validation."""
        all_ev = [e for evs in self._events.values() for e in evs]
        if not all_ev:
            return {"total_events": 0}
        surprises = [e.surprise_pct for e in all_ev]
        return {
            "total_events": len(all_ev),
            "tickers_with_data": sum(1 for evs in self._events.values() if evs),
            "date_range": (
                min(e.report_date for e in all_ev).isoformat(),
                max(e.report_date for e in all_ev).isoformat(),
            ),
            "beats": sum(1 for e in all_ev if e.is_beat),
            "misses": sum(1 for e in all_ev if e.is_miss),
            "median_surprise_pct": float(np.median(surprises)),
            "mean_surprise_pct": float(np.mean(surprises)),
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _next_bday(d: date) -> date:
    """Next business day after d."""
    return np.busday_offset(d, 1, roll="forward").astype(date)
