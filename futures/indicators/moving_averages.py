"""Moving average indicators."""

import pandas as pd

from .base import Indicator


class SMA(Indicator):
    """Simple Moving Average."""

    def __init__(self, period: int = 20, column: str = "close"):
        """
        Initialize SMA indicator.

        Args:
            period: Lookback period for the average
            column: Column to compute average on (default: close)
        """
        super().__init__(period=period, column=column)
        self.period = period
        self.column = column

    @property
    def name(self) -> str:
        return f"sma_{self.period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute simple moving average."""
        return df[self.column].rolling(window=self.period).mean()


class EMA(Indicator):
    """Exponential Moving Average."""

    def __init__(self, period: int = 20, column: str = "close"):
        """
        Initialize EMA indicator.

        Args:
            period: Lookback period (span) for the EMA
            column: Column to compute average on
        """
        super().__init__(period=period, column=column)
        self.period = period
        self.column = column

    @property
    def name(self) -> str:
        return f"ema_{self.period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute exponential moving average."""
        return df[self.column].ewm(span=self.period, adjust=False).mean()


class WMA(Indicator):
    """Weighted Moving Average."""

    def __init__(self, period: int = 20, column: str = "close"):
        """Initialize WMA indicator."""
        super().__init__(period=period, column=column)
        self.period = period
        self.column = column

    @property
    def name(self) -> str:
        return f"wma_{self.period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute weighted moving average with linear weights."""
        weights = pd.Series(range(1, self.period + 1))
        return (
            df[self.column]
            .rolling(window=self.period)
            .apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
        )


class VWAP(Indicator):
    """Volume Weighted Average Price."""

    def __init__(self, period: int = 20):
        """Initialize VWAP indicator."""
        super().__init__(period=period)
        self.period = period

    @property
    def name(self) -> str:
        return f"vwap_{self.period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute rolling VWAP."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_volume = typical_price * df["volume"]

        return (
            tp_volume.rolling(window=self.period).sum()
            / df["volume"].rolling(window=self.period).sum()
        )


class DEMA(Indicator):
    """Double Exponential Moving Average."""

    def __init__(self, period: int = 20, column: str = "close"):
        """Initialize DEMA indicator."""
        super().__init__(period=period, column=column)
        self.period = period
        self.column = column

    @property
    def name(self) -> str:
        return f"dema_{self.period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute DEMA: 2 * EMA - EMA(EMA)."""
        ema1 = df[self.column].ewm(span=self.period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        return 2 * ema1 - ema2
