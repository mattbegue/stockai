"""
Earnings calendar for risk filtering (P2-4).

Fetches historical earnings dates via yfinance and caches locally.
Used to:
  1. Generate `days_to_earnings` and `earnings_within_hold` features
  2. Hard-filter signals whose holding window straddles an earnings date

Earnings dates create discontinuous jumps (gaps) that make short-term
directional predictions unreliable — the outcome is dominated by the
surprise relative to consensus, not the technical setup.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# How many trading-day business days to look ahead for earnings proximity
_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "earnings_cache"


class EarningsCalendar:
    """
    Per-ticker earnings date registry with local file cache.

    Usage::

        calendar = EarningsCalendar()
        calendar.load(tickers)        # Fetches + caches all at once

        days = calendar.days_to_next(ticker, signal_date)
        within = calendar.within_hold(ticker, signal_date, holding_days=5)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Dict: ticker → sorted list of datetime.date objects
        self._dates: dict[str, list[date]] = {}

    # ------------------------------------------------------------------
    # Loading / caching
    # ------------------------------------------------------------------

    def load(
        self,
        tickers: list[str],
        n_quarters: int = 20,
        force_refresh: bool = False,
    ) -> None:
        """
        Fetch and cache earnings dates for all tickers.

        Args:
            tickers: List of stock tickers to fetch
            n_quarters: How many quarters back to fetch (default 20 = 5 years)
            force_refresh: Ignore cache and re-fetch from yfinance
        """
        for ticker in tickers:
            if ticker in self._dates and not force_refresh:
                continue
            cached = self._load_cache(ticker)
            if cached is not None and not force_refresh:
                self._dates[ticker] = cached
            else:
                fetched = self._fetch(ticker, n_quarters)
                self._dates[ticker] = fetched
                self._save_cache(ticker, fetched)

        loaded = sum(1 for t in tickers if self._dates.get(t))
        logger.info(f"EarningsCalendar: loaded {loaded}/{len(tickers)} tickers")

    def _fetch(self, ticker: str, n_quarters: int) -> list[date]:
        """Fetch earnings dates for a ticker via yfinance."""
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            df = t.get_earnings_dates(limit=n_quarters)
            if df is None or df.empty:
                return []
            # Index is the earnings date (timezone-aware); normalize to date
            dates = [
                ts.date() if hasattr(ts, "date") else ts
                for ts in df.index
            ]
            # Drop future (unpublished) earnings — NaN in Reported EPS
            known_dates = [
                d for d, row in zip(dates, df.itertuples())
                if not (hasattr(row, "Reported_EPS") and pd.isna(row.Reported_EPS))
                or d <= date.today()
            ]
            return sorted(set(known_dates))
        except Exception as exc:
            logger.debug(f"EarningsCalendar: failed to fetch {ticker}: {exc}")
            return []

    def _cache_path(self, ticker: str) -> Path:
        safe = ticker.replace(".", "_")
        return self._cache_dir / f"{safe}.json"

    def _load_cache(self, ticker: str) -> Optional[list[date]]:
        path = self._cache_path(ticker)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            return [date.fromisoformat(d) for d in raw["dates"]]
        except Exception:
            return None

    def _save_cache(self, ticker: str, dates: list[date]) -> None:
        path = self._cache_path(ticker)
        try:
            path.write_text(json.dumps({"dates": [d.isoformat() for d in dates]}))
        except Exception as exc:
            logger.debug(f"EarningsCalendar: cache write failed for {ticker}: {exc}")

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def days_to_next(
        self,
        ticker: str,
        signal_date: pd.Timestamp | date,
        max_days: int = 90,
    ) -> float:
        """
        Calendar days until the next known earnings date after signal_date.

        Returns max_days if no earnings found within max_days, or if the
        ticker is not in the calendar.
        """
        ref = signal_date.date() if hasattr(signal_date, "date") else signal_date
        dates = self._dates.get(ticker, [])
        future = [d for d in dates if d > ref]
        if not future:
            return float(max_days)
        delta = (min(future) - ref).days
        return float(min(delta, max_days))

    def within_hold(
        self,
        ticker: str,
        signal_date: pd.Timestamp | date,
        holding_days: int = 5,
    ) -> bool:
        """
        Returns True if any earnings date falls within
        [signal_date, signal_date + holding_days calendar days].

        A hold window that straddles earnings carries jump risk that the
        model cannot price — the signal should be skipped.
        """
        ref = signal_date.date() if hasattr(signal_date, "date") else signal_date
        hold_end = ref + timedelta(days=holding_days * 2)  # 2× for calendar vs trading days
        dates = self._dates.get(ticker, [])
        return any(ref < d <= hold_end for d in dates)

    def has_data(self, ticker: str) -> bool:
        """Returns True if we have at least one earnings date for this ticker."""
        return bool(self._dates.get(ticker))
