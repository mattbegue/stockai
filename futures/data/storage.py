"""SQLite storage for historical price data."""

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from futures.config import get_settings


class Storage:
    """SQLite-based storage for OHLCV price data."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize storage with database path."""
        self.db_path = db_path or get_settings().db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    vwap REAL,
                    trade_count INTEGER,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    ticker TEXT PRIMARY KEY,
                    last_updated TEXT,
                    first_date TEXT,
                    last_date TEXT
                )
            """)

    def save_prices(self, ticker: str, df: pd.DataFrame):
        """
        Save price data for a ticker.

        Args:
            ticker: Stock ticker symbol
            df: DataFrame with columns [open, high, low, close, volume]
                and DatetimeIndex
        """
        if df.empty:
            return

        records = []
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
            records.append((
                ticker,
                date_str,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
                float(row.get("vwap", 0)) if pd.notna(row.get("vwap")) else None,
                int(row.get("trade_count", 0)) if pd.notna(row.get("trade_count")) else None,
            ))

        with self._connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO prices
                (ticker, date, open, high, low, close, volume, vwap, trade_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)

            # Update metadata
            dates = df.index.sort_values()
            conn.execute("""
                INSERT OR REPLACE INTO metadata (ticker, last_updated, first_date, last_date)
                VALUES (?, ?, ?, ?)
            """, (
                ticker,
                datetime.now().isoformat(),
                dates[0].strftime("%Y-%m-%d"),
                dates[-1].strftime("%Y-%m-%d"),
            ))

    def load_prices(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Load price data for a ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date filter (inclusive)
            end_date: End date filter (inclusive)

        Returns:
            DataFrame with OHLCV data and DatetimeIndex
        """
        query = "SELECT * FROM prices WHERE ticker = ?"
        params = [ticker]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY date"

        with self._connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.drop(columns=["ticker"], inplace=True)
        return df

    def load_multi(
        self,
        tickers: list[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> dict[str, pd.DataFrame]:
        """Load price data for multiple tickers."""
        return {
            ticker: self.load_prices(ticker, start_date, end_date)
            for ticker in tickers
        }

    def get_available_tickers(self) -> list[str]:
        """Get list of all tickers with stored data."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker")
            return [row[0] for row in cursor.fetchall()]

    def get_date_range(self, ticker: str) -> tuple[Optional[date], Optional[date]]:
        """Get the date range available for a ticker."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT first_date, last_date FROM metadata WHERE ticker = ?",
                (ticker,)
            )
            row = cursor.fetchone()
            if row:
                return (
                    date.fromisoformat(row[0]) if row[0] else None,
                    date.fromisoformat(row[1]) if row[1] else None,
                )
            return None, None

    def needs_update(self, ticker: str, lookback_days: int = 1) -> bool:
        """Check if ticker data needs updating."""
        _, last_date = self.get_date_range(ticker)
        if not last_date:
            return True
        days_old = (date.today() - last_date).days
        return days_old > lookback_days
