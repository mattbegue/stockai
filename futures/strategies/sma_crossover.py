"""Simple Moving Average Crossover Strategy."""

import pandas as pd

from futures.indicators import SMA
from .base import Strategy, Signal


class SMACrossover(Strategy):
    """
    Classic SMA crossover strategy.

    BUY when fast SMA crosses above slow SMA.
    SELL when fast SMA crosses below slow SMA.
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        """
        Initialize SMA crossover strategy.

        Args:
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
        """
        super().__init__(fast_period=fast_period, slow_period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period

        self.fast_sma = SMA(period=fast_period)
        self.slow_sma = SMA(period=slow_period)

    @property
    def name(self) -> str:
        return f"sma_cross_{self.fast_period}_{self.slow_period}"

    @property
    def required_history(self) -> int:
        return self.slow_period + 1

    def precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA columns to dataframe."""
        result = df.copy()
        result[f"sma_{self.fast_period}"] = self.fast_sma(df)
        result[f"sma_{self.slow_period}"] = self.slow_sma(df)
        return result

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """Generate crossover signals."""
        signals = {}

        for ticker, df in data.items():
            if len(df) < self.required_history:
                signals[ticker] = Signal.HOLD
                continue

            # Compute or get cached SMAs
            fast_col = f"sma_{self.fast_period}"
            slow_col = f"sma_{self.slow_period}"

            if fast_col in df.columns:
                fast = df[fast_col]
                slow = df[slow_col]
            else:
                fast = self.fast_sma(df)
                slow = self.slow_sma(df)

            # Check for crossover in the last bar
            if len(fast) < 2 or pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
                signals[ticker] = Signal.HOLD
                continue

            curr_fast, curr_slow = fast.iloc[-1], slow.iloc[-1]
            prev_fast, prev_slow = fast.iloc[-2], slow.iloc[-2]

            # Bullish crossover: fast crosses above slow
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                signals[ticker] = Signal.BUY
            # Bearish crossover: fast crosses below slow
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                signals[ticker] = Signal.SELL
            else:
                signals[ticker] = Signal.HOLD

        return signals


class TripleSMACrossover(Strategy):
    """
    Triple SMA crossover with trend filter.

    Only take long signals when price is above the trend SMA.
    Only take short signals when price is below the trend SMA.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 20,
        trend_period: int = 50,
    ):
        """Initialize triple SMA crossover strategy."""
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            trend_period=trend_period,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_period = trend_period

    @property
    def name(self) -> str:
        return f"triple_sma_{self.fast_period}_{self.slow_period}_{self.trend_period}"

    @property
    def required_history(self) -> int:
        return self.trend_period + 1

    def precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA columns."""
        result = df.copy()
        result[f"sma_{self.fast_period}"] = SMA(self.fast_period)(df)
        result[f"sma_{self.slow_period}"] = SMA(self.slow_period)(df)
        result[f"sma_{self.trend_period}"] = SMA(self.trend_period)(df)
        return result

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """Generate signals with trend filter."""
        signals = {}

        for ticker, df in data.items():
            if len(df) < self.required_history:
                signals[ticker] = Signal.HOLD
                continue

            fast = df.get(f"sma_{self.fast_period}", SMA(self.fast_period)(df))
            slow = df.get(f"sma_{self.slow_period}", SMA(self.slow_period)(df))
            trend = df.get(f"sma_{self.trend_period}", SMA(self.trend_period)(df))

            if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]) or pd.isna(trend.iloc[-1]):
                signals[ticker] = Signal.HOLD
                continue

            price = df["close"].iloc[-1]
            curr_fast, curr_slow = fast.iloc[-1], slow.iloc[-1]
            prev_fast, prev_slow = fast.iloc[-2], slow.iloc[-2]
            trend_val = trend.iloc[-1]

            # Bullish: fast crosses above slow AND price above trend
            if prev_fast <= prev_slow and curr_fast > curr_slow and price > trend_val:
                signals[ticker] = Signal.BUY
            # Bearish: fast crosses below slow AND price below trend
            elif prev_fast >= prev_slow and curr_fast < curr_slow and price < trend_val:
                signals[ticker] = Signal.SELL
            else:
                signals[ticker] = Signal.HOLD

        return signals
