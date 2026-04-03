"""Base indicator class."""

from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


class Indicator(ABC):
    """
    Abstract base class for all technical indicators.

    Indicators are stateless transformations that take price data
    and return computed indicator values.
    """

    def __init__(self, **params):
        """Initialize indicator with parameters."""
        self.params = params

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this indicator configuration."""
        pass

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute the indicator values.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            Series or DataFrame with indicator values, same index as input
        """
        pass

    def __call__(self, df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """Allow calling indicator as a function."""
        return self.compute(df)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


class IndicatorPipeline:
    """
    Pipeline to compute multiple indicators at once.

    Example:
        pipeline = IndicatorPipeline([
            SMA(period=20),
            SMA(period=50),
            RSI(period=14),
        ])
        features = pipeline.compute(df)
    """

    def __init__(self, indicators: list[Indicator]):
        """Initialize with list of indicators."""
        self.indicators = indicators

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators and return as DataFrame."""
        results = {}
        for indicator in self.indicators:
            result = indicator.compute(df)
            if isinstance(result, pd.Series):
                results[indicator.name] = result
            else:
                # Multi-column indicator
                for col in result.columns:
                    results[f"{indicator.name}_{col}"] = result[col]

        return pd.DataFrame(results, index=df.index)

    def __repr__(self) -> str:
        return f"IndicatorPipeline({self.indicators})"
