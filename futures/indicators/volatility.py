"""Volatility indicators."""

import pandas as pd
import numpy as np

from .base import Indicator


class ATR(Indicator):
    """Average True Range - volatility indicator."""

    def __init__(self, period: int = 14):
        """Initialize ATR indicator."""
        super().__init__(period=period)
        self.period = period

    @property
    def name(self) -> str:
        return f"atr_{self.period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's smoothing
        atr = true_range.ewm(alpha=1 / self.period, min_periods=self.period).mean()

        return atr


class StandardDeviation(Indicator):
    """Rolling standard deviation."""

    def __init__(self, period: int = 20, column: str = "close"):
        """Initialize StandardDeviation indicator."""
        super().__init__(period=period, column=column)
        self.period = period
        self.column = column

    @property
    def name(self) -> str:
        return f"std_{self.period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute rolling standard deviation."""
        return df[self.column].rolling(window=self.period).std()


class HistoricalVolatility(Indicator):
    """Annualized historical volatility."""

    def __init__(self, period: int = 20, trading_days: int = 252, column: str = "close"):
        """Initialize HistoricalVolatility indicator."""
        super().__init__(period=period, trading_days=trading_days, column=column)
        self.period = period
        self.trading_days = trading_days
        self.column = column

    @property
    def name(self) -> str:
        return f"hvol_{self.period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute annualized historical volatility."""
        log_returns = np.log(df[self.column] / df[self.column].shift(1))
        return log_returns.rolling(window=self.period).std() * np.sqrt(self.trading_days)


class KeltnerChannels(Indicator):
    """Keltner Channels - volatility-based envelope."""

    def __init__(self, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        """Initialize Keltner Channels indicator."""
        super().__init__(ema_period=ema_period, atr_period=atr_period, multiplier=multiplier)
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier

    @property
    def name(self) -> str:
        return f"kc_{self.ema_period}_{self.atr_period}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Keltner Channels.

        Returns DataFrame with columns: middle, upper, lower
        """
        middle = df["close"].ewm(span=self.ema_period, adjust=False).mean()

        atr = ATR(period=self.atr_period).compute(df)

        upper = middle + (atr * self.multiplier)
        lower = middle - (atr * self.multiplier)

        return pd.DataFrame({
            "middle": middle,
            "upper": upper,
            "lower": lower,
        }, index=df.index)


class DonchianChannels(Indicator):
    """Donchian Channels - breakout indicator."""

    def __init__(self, period: int = 20):
        """Initialize Donchian Channels indicator."""
        super().__init__(period=period)
        self.period = period

    @property
    def name(self) -> str:
        return f"dc_{self.period}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Donchian Channels.

        Returns DataFrame with columns: upper, lower, middle
        """
        upper = df["high"].rolling(window=self.period).max()
        lower = df["low"].rolling(window=self.period).min()
        middle = (upper + lower) / 2

        return pd.DataFrame({
            "upper": upper,
            "lower": lower,
            "middle": middle,
        }, index=df.index)
