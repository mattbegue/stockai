"""Alpaca API data fetcher."""

import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from tqdm import tqdm

from futures.config import get_settings


class AlpacaFetcher:
    """Fetch historical price data from Alpaca Markets API."""

    def __init__(self, rate_limit_delay: float = 0.5, max_retries: int = 3):
        """
        Initialize the Alpaca client.

        Args:
            rate_limit_delay: Seconds to wait between API requests (default 0.5)
            max_retries: Maximum number of retries for failed requests (default 3)
        """
        settings = get_settings()

        # Import here to allow module to load without alpaca installed
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        self._client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        self._StockBarsRequest = StockBarsRequest
        self._TimeFrame = TimeFrame
        self._DataFeed = DataFeed
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries

    def fetch(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        timeframe: str = "day",
    ) -> pd.DataFrame:
        """
        Fetch historical bars for a single ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (defaults to 2 years ago)
            end_date: End date (defaults to today)
            timeframe: 'day' or 'hour'

        Returns:
            DataFrame with OHLCV data
        """
        settings = get_settings()

        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=settings.default_lookback_days)

        tf = self._TimeFrame.Day if timeframe == "day" else self._TimeFrame.Hour

        request = self._StockBarsRequest(
            symbol_or_symbols=ticker,
            start=datetime.combine(start_date, datetime.min.time()),
            end=datetime.combine(end_date, datetime.max.time()),
            timeframe=tf,
            feed=self._DataFeed.IEX,  # Use free IEX feed instead of paid SIP
        )

        bars = self._client.get_stock_bars(request)

        if not bars or ticker not in bars.data:
            return pd.DataFrame()

        records = []
        for bar in bars.data[ticker]:
            records.append({
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vwap": bar.vwap,
                "trade_count": bar.trade_count,
            })

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Normalize to date for daily data
        if timeframe == "day":
            df.index = df.index.date
            df.index = pd.to_datetime(df.index)

        return df

    def fetch_multi(
        self,
        tickers: list[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        timeframe: str = "day",
        show_progress: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical bars for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            timeframe: 'day' or 'hour'
            show_progress: Show progress bar

        Returns:
            Dict mapping ticker to DataFrame
        """
        results = {}
        iterator = tqdm(tickers, desc="Fetching") if show_progress else tickers

        for i, ticker in enumerate(iterator):
            # Rate limiting: wait between requests (skip first request)
            if i > 0 and self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

            # Retry logic with exponential backoff
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    df = self.fetch(ticker, start_date, end_date, timeframe)
                    if not df.empty:
                        results[ticker] = df
                    break
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s...
                        backoff = 2**attempt
                        time.sleep(backoff)
                    continue
            else:
                # All retries exhausted
                print(f"Error fetching {ticker} after {self.max_retries} retries: {last_error}")

        return results


class DataManager:
    """High-level interface combining fetcher and storage."""

    def __init__(self):
        """Initialize with default fetcher and storage."""
        from .storage import Storage

        self.fetcher = AlpacaFetcher()
        self.storage = Storage()

    def get_prices(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get price data, fetching from API if needed.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            refresh: Force refresh from API

        Returns:
            DataFrame with OHLCV data
        """
        if refresh or self.storage.needs_update(ticker):
            df = self.fetcher.fetch(ticker, start_date, end_date)
            if not df.empty:
                self.storage.save_prices(ticker, df)
            return df

        return self.storage.load_prices(ticker, start_date, end_date)

    def get_multi(
        self,
        tickers: list[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        refresh: bool = False,
        show_progress: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Get price data for multiple tickers."""
        results = {}
        iterator = tqdm(tickers, desc="Loading") if show_progress else tickers

        for ticker in iterator:
            df = self.get_prices(ticker, start_date, end_date, refresh)
            if not df.empty:
                results[ticker] = df

        return results
