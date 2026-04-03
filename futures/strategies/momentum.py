"""Momentum-based strategies."""

import pandas as pd
import numpy as np

from futures.indicators import MACD, RSI, SMA, ROC
from .base import Strategy, Signal


class MomentumStrategy(Strategy):
    """
    Momentum strategy using MACD and RSI confirmation.

    BUY when MACD histogram is positive and RSI is rising.
    SELL when MACD histogram is negative and RSI is falling.
    """

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
    ):
        """Initialize momentum strategy."""
        super().__init__(
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            rsi_period=rsi_period,
        )
        self.macd = MACD(macd_fast, macd_slow, macd_signal)
        self.rsi = RSI(period=rsi_period)
        self.rsi_period = rsi_period

    @property
    def name(self) -> str:
        return f"momentum_macd_rsi{self.rsi_period}"

    @property
    def required_history(self) -> int:
        return 30  # Need enough for MACD slow + signal

    def precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD and RSI columns."""
        result = df.copy()

        macd = self.macd(df)
        result["macd_histogram"] = macd["histogram"]

        result[f"rsi_{self.rsi_period}"] = self.rsi(df)

        return result

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """Generate momentum signals."""
        signals = {}

        for ticker, df in data.items():
            if len(df) < self.required_history:
                signals[ticker] = Signal.HOLD
                continue

            # Get indicators
            if "macd_histogram" in df.columns:
                histogram = df["macd_histogram"]
                rsi = df[f"rsi_{self.rsi_period}"]
            else:
                macd_result = self.macd(df)
                histogram = macd_result["histogram"]
                rsi = self.rsi(df)

            if pd.isna(histogram.iloc[-1]) or pd.isna(rsi.iloc[-1]):
                signals[ticker] = Signal.HOLD
                continue

            curr_hist = histogram.iloc[-1]
            prev_hist = histogram.iloc[-2]
            rsi_rising = rsi.iloc[-1] > rsi.iloc[-2]

            # Bullish: positive histogram increasing + RSI rising
            if curr_hist > 0 and curr_hist > prev_hist and rsi_rising:
                signals[ticker] = Signal.BUY
            # Bearish: negative histogram decreasing + RSI falling
            elif curr_hist < 0 and curr_hist < prev_hist and not rsi_rising:
                signals[ticker] = Signal.SELL
            else:
                signals[ticker] = Signal.HOLD

        return signals


class TrendFollowing(Strategy):
    """
    Trend following strategy based on price breakouts.

    BUY when price breaks above recent high.
    SELL when price breaks below recent low.
    """

    def __init__(self, breakout_period: int = 20, atr_period: int = 14):
        """Initialize trend following strategy."""
        super().__init__(breakout_period=breakout_period, atr_period=atr_period)
        self.breakout_period = breakout_period
        self.atr_period = atr_period

    @property
    def name(self) -> str:
        return f"trend_follow_{self.breakout_period}"

    @property
    def required_history(self) -> int:
        return self.breakout_period + 1

    def precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add breakout channel columns."""
        result = df.copy()
        result["channel_high"] = df["high"].rolling(self.breakout_period).max()
        result["channel_low"] = df["low"].rolling(self.breakout_period).min()
        return result

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """Generate breakout signals."""
        signals = {}

        for ticker, df in data.items():
            if len(df) < self.required_history:
                signals[ticker] = Signal.HOLD
                continue

            if "channel_high" in df.columns:
                prev_high = df["channel_high"].iloc[-2]
                prev_low = df["channel_low"].iloc[-2]
            else:
                prev_high = df["high"].iloc[:-1].rolling(self.breakout_period).max().iloc[-1]
                prev_low = df["low"].iloc[:-1].rolling(self.breakout_period).min().iloc[-1]

            if pd.isna(prev_high) or pd.isna(prev_low):
                signals[ticker] = Signal.HOLD
                continue

            curr_close = df["close"].iloc[-1]

            # Breakout above recent high
            if curr_close > prev_high:
                signals[ticker] = Signal.BUY
            # Breakdown below recent low
            elif curr_close < prev_low:
                signals[ticker] = Signal.SELL
            else:
                signals[ticker] = Signal.HOLD

        return signals


class RelativeStrength(Strategy):
    """
    Relative strength strategy - compare stock to benchmark.

    BUY when stock outperforms benchmark.
    SELL when stock underperforms benchmark.
    """

    def __init__(self, benchmark: str = "SPY", lookback: int = 20):
        """
        Initialize relative strength strategy.

        Args:
            benchmark: Ticker to compare against
            lookback: Period for relative strength calculation
        """
        super().__init__(benchmark=benchmark, lookback=lookback)
        self.benchmark = benchmark
        self.lookback = lookback

    @property
    def name(self) -> str:
        return f"rel_strength_{self.benchmark}_{self.lookback}"

    @property
    def required_history(self) -> int:
        return self.lookback + 1

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """Generate relative strength signals."""
        signals = {}

        if self.benchmark not in data:
            return {ticker: Signal.HOLD for ticker in data.keys()}

        benchmark_df = data[self.benchmark]
        if len(benchmark_df) < self.required_history:
            return {ticker: Signal.HOLD for ticker in data.keys()}

        benchmark_return = (
            benchmark_df["close"].iloc[-1] / benchmark_df["close"].iloc[-self.lookback] - 1
        )

        for ticker, df in data.items():
            if ticker == self.benchmark:
                signals[ticker] = Signal.HOLD
                continue

            if len(df) < self.required_history:
                signals[ticker] = Signal.HOLD
                continue

            stock_return = df["close"].iloc[-1] / df["close"].iloc[-self.lookback] - 1
            relative_strength = stock_return - benchmark_return

            # Outperforming benchmark significantly
            if relative_strength > 0.02:  # 2% outperformance
                signals[ticker] = Signal.BUY
            # Underperforming benchmark significantly
            elif relative_strength < -0.02:  # 2% underperformance
                signals[ticker] = Signal.SELL
            else:
                signals[ticker] = Signal.HOLD

        return signals
