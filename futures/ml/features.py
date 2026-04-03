"""Feature engineering for ML models."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from futures.indicators import (
    SMA, EMA, RSI, MACD, BollingerBands,
    ATR, StandardDeviation
)
from futures.indicators.base import IndicatorPipeline


@dataclass
class FeatureSet:
    """Container for feature matrix and labels."""

    X: pd.DataFrame
    y: Optional[pd.Series] = None
    feature_names: list[str] = field(default_factory=list)
    dates: Optional[pd.DatetimeIndex] = None

    def __post_init__(self):
        if not self.feature_names:
            self.feature_names = list(self.X.columns)
        if self.dates is None:
            self.dates = self.X.index

    def train_test_split(
        self,
        test_size: float = 0.2,
        shuffle: bool = False,
    ) -> tuple["FeatureSet", "FeatureSet"]:
        """Split into train and test sets (time-aware by default)."""
        n = len(self.X)
        split_idx = int(n * (1 - test_size))

        if shuffle:
            indices = np.random.permutation(n)
            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]
        else:
            train_idx = range(split_idx)
            test_idx = range(split_idx, n)

        train = FeatureSet(
            X=self.X.iloc[train_idx],
            y=self.y.iloc[train_idx] if self.y is not None else None,
            feature_names=self.feature_names,
        )
        test = FeatureSet(
            X=self.X.iloc[test_idx],
            y=self.y.iloc[test_idx] if self.y is not None else None,
            feature_names=self.feature_names,
        )

        return train, test


class FeatureEngineering:
    """
    Feature engineering pipeline for ML trading models.

    Creates features from price data for use in classification/regression.
    """

    def __init__(
        self,
        include_price_features: bool = True,
        include_momentum_features: bool = True,
        include_volatility_features: bool = True,
        include_volume_features: bool = True,
        lookback_periods: list[int] = None,
        forward_return_periods: list[int] = None,
    ):
        """
        Initialize feature engineering pipeline.

        Args:
            include_price_features: Include price-based features
            include_momentum_features: Include momentum indicators
            include_volatility_features: Include volatility measures
            include_volume_features: Include volume analysis
            lookback_periods: Periods for rolling features
            forward_return_periods: Periods for target calculation
        """
        self.include_price_features = include_price_features
        self.include_momentum_features = include_momentum_features
        self.include_volatility_features = include_volatility_features
        self.include_volume_features = include_volume_features
        self.lookback_periods = lookback_periods or [5, 10, 20, 50]
        self.forward_return_periods = forward_return_periods or [1, 5, 10]

    def create_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        target_horizon: int = 1,
    ) -> FeatureSet:
        """
        Create feature matrix from price data.

        Args:
            df: DataFrame with OHLCV data
            target_column: Column to use for target (default: 'close')
            target_horizon: Days ahead for target return

        Returns:
            FeatureSet with features and optional labels
        """
        features = pd.DataFrame(index=df.index)

        # Price features
        if self.include_price_features:
            features = self._add_price_features(features, df)

        # Momentum features
        if self.include_momentum_features:
            features = self._add_momentum_features(features, df)

        # Volatility features
        if self.include_volatility_features:
            features = self._add_volatility_features(features, df)

        # Volume features
        if self.include_volume_features:
            features = self._add_volume_features(features, df)

        # Create target (forward return)
        target = None
        if target_column or target_horizon:
            col = target_column or "close"
            target = df[col].pct_change(target_horizon).shift(-target_horizon)

        # Drop NaN rows
        valid_mask = features.notna().all(axis=1)
        if target is not None:
            valid_mask = valid_mask & target.notna()

        features = features[valid_mask]
        if target is not None:
            target = target[valid_mask]

        return FeatureSet(
            X=features,
            y=target,
            feature_names=list(features.columns),
            dates=features.index,
        )

    def _add_price_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        close = df["close"]

        # Returns at different horizons
        for period in self.lookback_periods:
            features[f"return_{period}d"] = close.pct_change(period)

        # Price relative to SMAs
        for period in self.lookback_periods:
            sma = SMA(period=period)(df)
            features[f"price_vs_sma{period}"] = (close - sma) / sma

        # Price position in range
        for period in self.lookback_periods:
            high_roll = df["high"].rolling(period).max()
            low_roll = df["low"].rolling(period).min()
            features[f"price_position_{period}d"] = (close - low_roll) / (high_roll - low_roll)

        # Gap (today open vs yesterday close)
        features["gap"] = (df["open"] - close.shift(1)) / close.shift(1)

        # Intraday range
        features["intraday_range"] = (df["high"] - df["low"]) / df["open"]

        # Close position within day
        features["close_position"] = (close - df["low"]) / (df["high"] - df["low"])

        return features

    def _add_momentum_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicator features."""
        # RSI at different periods
        for period in [7, 14, 21]:
            features[f"rsi_{period}"] = RSI(period=period)(df)

        # MACD
        macd = MACD()(df)
        features["macd_line"] = macd["macd"]
        features["macd_signal"] = macd["signal"]
        features["macd_histogram"] = macd["histogram"]
        features["macd_hist_change"] = macd["histogram"].diff()

        # Rate of change
        for period in [5, 10, 20]:
            features[f"roc_{period}"] = df["close"].pct_change(period) * 100

        # Momentum (price difference)
        for period in [5, 10]:
            features[f"momentum_{period}"] = df["close"].diff(period)

        return features

    def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        # ATR
        for period in [7, 14, 21]:
            features[f"atr_{period}"] = ATR(period=period)(df)

        # ATR as percentage of price
        atr_14 = ATR(period=14)(df)
        features["atr_pct"] = atr_14 / df["close"]

        # Bollinger Band features
        bb = BollingerBands(period=20)(df)
        features["bb_pct_b"] = bb["pct_b"]
        features["bb_bandwidth"] = bb["bandwidth"]

        # Historical volatility
        for period in [10, 20]:
            log_returns = np.log(df["close"] / df["close"].shift(1))
            features[f"volatility_{period}d"] = log_returns.rolling(period).std() * np.sqrt(252)

        # Volatility regime (current vs longer term)
        vol_10 = features["volatility_10d"]
        vol_20 = features["volatility_20d"]
        features["vol_regime"] = vol_10 / vol_20

        return features

    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        volume = df["volume"]

        # Volume relative to average
        for period in [5, 10, 20]:
            vol_sma = volume.rolling(period).mean()
            features[f"volume_vs_avg_{period}d"] = volume / vol_sma

        # Volume trend
        features["volume_change"] = volume.pct_change()
        features["volume_ma_slope"] = volume.rolling(10).mean().diff(5)

        # On-balance volume (simplified)
        price_direction = np.sign(df["close"].diff())
        features["obv_direction"] = price_direction * volume
        features["obv_ma"] = features["obv_direction"].rolling(10).mean()

        # Volume-price correlation
        for period in [10, 20]:
            features[f"vol_price_corr_{period}d"] = (
                volume.rolling(period)
                .corr(df["close"])
            )

        return features

    def create_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        threshold: float = 0.0,
    ) -> pd.Series:
        """
        Create classification labels.

        Args:
            df: DataFrame with price data
            horizon: Days ahead to look
            threshold: Return threshold for BUY signal

        Returns:
            Series with labels: 1 (BUY), 0 (HOLD), -1 (SELL)
        """
        forward_return = df["close"].pct_change(horizon).shift(-horizon)

        labels = pd.Series(0, index=df.index)
        labels[forward_return > threshold] = 1
        labels[forward_return < -threshold] = -1

        return labels


def create_cross_sectional_features(
    data: dict[str, pd.DataFrame],
    benchmark: str = "SPY",
) -> dict[str, pd.DataFrame]:
    """
    Create cross-sectional features comparing stocks to each other.

    Args:
        data: Dict mapping ticker to DataFrame
        benchmark: Benchmark ticker for relative strength

    Returns:
        Dict with features added to each DataFrame
    """
    result = {}

    # Get benchmark data
    benchmark_df = data.get(benchmark)
    if benchmark_df is None:
        return data

    benchmark_returns = benchmark_df["close"].pct_change()

    for ticker, df in data.items():
        features = df.copy()

        if ticker == benchmark:
            result[ticker] = features
            continue

        # Relative strength vs benchmark
        stock_returns = df["close"].pct_change()
        features["relative_return_1d"] = stock_returns - benchmark_returns

        for period in [5, 10, 20]:
            stock_ret = df["close"].pct_change(period)
            bench_ret = benchmark_df["close"].pct_change(period)
            features[f"relative_return_{period}d"] = stock_ret - bench_ret

        # Beta (rolling)
        for period in [20, 60]:
            cov = stock_returns.rolling(period).cov(benchmark_returns)
            var = benchmark_returns.rolling(period).var()
            features[f"beta_{period}d"] = cov / var

        # Correlation with benchmark
        features["corr_benchmark_20d"] = stock_returns.rolling(20).corr(benchmark_returns)

        result[ticker] = features

    return result
