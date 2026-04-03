"""Momentum indicators."""

import pandas as pd

from .base import Indicator


class RSI(Indicator):
    """Relative Strength Index."""

    def __init__(self, period: int = 14, column: str = "close"):
        """Initialize RSI indicator."""
        super().__init__(period=period, column=column)
        self.period = period
        self.column = column

    @property
    def name(self) -> str:
        return f"rsi_{self.period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute RSI using Wilder's smoothing method."""
        delta = df[self.column].diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1 / self.period, min_periods=self.period).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, min_periods=self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


class MACD(Indicator):
    """Moving Average Convergence Divergence."""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close",
    ):
        """Initialize MACD indicator."""
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            column=column,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.column = column

    @property
    def name(self) -> str:
        return f"macd_{self.fast_period}_{self.slow_period}_{self.signal_period}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute MACD line, signal line, and histogram.

        Returns DataFrame with columns: macd, signal, histogram
        """
        fast_ema = df[self.column].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df[self.column].ewm(span=self.slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }, index=df.index)


class BollingerBands(Indicator):
    """Bollinger Bands."""

    def __init__(self, period: int = 20, std_dev: float = 2.0, column: str = "close"):
        """Initialize Bollinger Bands indicator."""
        super().__init__(period=period, std_dev=std_dev, column=column)
        self.period = period
        self.std_dev = std_dev
        self.column = column

    @property
    def name(self) -> str:
        return f"bb_{self.period}_{self.std_dev}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Bollinger Bands.

        Returns DataFrame with columns: middle, upper, lower, bandwidth, pct_b
        """
        middle = df[self.column].rolling(window=self.period).mean()
        std = df[self.column].rolling(window=self.period).std()

        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)

        bandwidth = (upper - lower) / middle
        pct_b = (df[self.column] - lower) / (upper - lower)

        return pd.DataFrame({
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "bandwidth": bandwidth,
            "pct_b": pct_b,
        }, index=df.index)


class Stochastic(Indicator):
    """Stochastic Oscillator."""

    def __init__(self, k_period: int = 14, d_period: int = 3):
        """Initialize Stochastic indicator."""
        super().__init__(k_period=k_period, d_period=d_period)
        self.k_period = k_period
        self.d_period = d_period

    @property
    def name(self) -> str:
        return f"stoch_{self.k_period}_{self.d_period}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Stochastic %K and %D.

        Returns DataFrame with columns: k, d
        """
        low_min = df["low"].rolling(window=self.k_period).min()
        high_max = df["high"].rolling(window=self.k_period).max()

        k = 100 * (df["close"] - low_min) / (high_max - low_min)
        d = k.rolling(window=self.d_period).mean()

        return pd.DataFrame({"k": k, "d": d}, index=df.index)


class ROC(Indicator):
    """Rate of Change."""

    def __init__(self, period: int = 10, column: str = "close"):
        """Initialize ROC indicator."""
        super().__init__(period=period, column=column)
        self.period = period
        self.column = column

    @property
    def name(self) -> str:
        return f"roc_{self.period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute Rate of Change as percentage."""
        return df[self.column].pct_change(periods=self.period) * 100


class MFI(Indicator):
    """Money Flow Index - volume-weighted RSI."""

    def __init__(self, period: int = 14):
        """Initialize MFI indicator."""
        super().__init__(period=period)
        self.period = period

    @property
    def name(self) -> str:
        return f"mfi_{self.period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute Money Flow Index."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]

        tp_diff = typical_price.diff()
        positive_flow = money_flow.where(tp_diff > 0, 0)
        negative_flow = money_flow.where(tp_diff < 0, 0)

        positive_sum = positive_flow.rolling(window=self.period).sum()
        negative_sum = negative_flow.rolling(window=self.period).sum()

        mfi = 100 - (100 / (1 + positive_sum / negative_sum))
        return mfi
