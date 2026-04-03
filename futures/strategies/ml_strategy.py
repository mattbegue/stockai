"""ML-based trading strategy."""

from typing import Optional

import pandas as pd

from futures.ml.features import FeatureEngineering, FeatureSet
from futures.ml.models import ModelWrapper, RandomForestModel
from futures.indicators import SMA
from .base import Strategy, Signal, FilterStrategy


class MLStrategy(Strategy):
    """
    ML-based trading strategy.

    Uses a trained classifier to generate trading signals.
    Can be combined with filter strategies for two-stage screening.
    """

    def __init__(
        self,
        model: ModelWrapper,
        feature_engineering: Optional[FeatureEngineering] = None,
        confidence_threshold: float = 0.6,
        filter_strategy: Optional[FilterStrategy] = None,
    ):
        """
        Initialize ML strategy.

        Args:
            model: Trained ML model
            feature_engineering: Feature engineering pipeline
            confidence_threshold: Min probability for signal
            filter_strategy: Optional filter to screen stocks first
        """
        super().__init__(
            confidence_threshold=confidence_threshold,
        )
        self.model = model
        self.feature_engineering = feature_engineering or FeatureEngineering()
        self.confidence_threshold = confidence_threshold
        self.filter_strategy = filter_strategy

        # Cache for computed features
        self._feature_cache: dict[str, pd.DataFrame] = {}

    @property
    def name(self) -> str:
        base = "ml_strategy"
        if self.filter_strategy:
            base += f"_filtered_{self.filter_strategy.name}"
        return base

    @property
    def required_history(self) -> int:
        base = 60  # Need enough for feature calculation
        if self.filter_strategy:
            return max(base, self.filter_strategy.required_history)
        return base

    def precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Precompute features for efficiency."""
        result = df.copy()

        # Let filter strategy add its indicators too
        if self.filter_strategy:
            result = self.filter_strategy.precompute_indicators(result)

        return result

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """Generate ML-based signals."""
        signals = {}

        # First pass: apply filter if present
        if self.filter_strategy:
            candidates = self.filter_strategy.screen(data)
            filtered_data = {k: v for k, v in data.items() if k in candidates}
        else:
            filtered_data = data

        # Generate signals for filtered stocks
        for ticker, df in filtered_data.items():
            if len(df) < self.required_history:
                signals[ticker] = Signal.HOLD
                continue

            try:
                signal = self._generate_single_signal(ticker, df)
                signals[ticker] = signal
            except Exception:
                signals[ticker] = Signal.HOLD

        # HOLD for stocks that didn't pass filter
        for ticker in data.keys():
            if ticker not in signals:
                signals[ticker] = Signal.HOLD

        return signals

    def _generate_single_signal(self, ticker: str, df: pd.DataFrame) -> Signal:
        """Generate signal for a single stock."""
        # Create features for the last row
        feature_set = self.feature_engineering.create_features(df)

        if feature_set.X.empty:
            return Signal.HOLD

        # Get prediction for the last available row
        last_features = FeatureSet(
            X=feature_set.X.iloc[[-1]],
            feature_names=feature_set.feature_names,
        )

        # Get prediction probabilities
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(last_features)

            # Assuming 3 classes: -1, 0, 1 (SELL, HOLD, BUY)
            if proba.shape[1] == 3:
                sell_prob = proba[0, 0]
                buy_prob = proba[0, 2]

                if buy_prob > self.confidence_threshold:
                    return Signal.BUY
                elif sell_prob > self.confidence_threshold:
                    return Signal.SELL
            elif proba.shape[1] == 2:
                # Binary: down (0) vs up (1)
                up_prob = proba[0, 1]
                if up_prob > self.confidence_threshold:
                    return Signal.BUY
                elif up_prob < (1 - self.confidence_threshold):
                    return Signal.SELL
        else:
            # No probabilities, use direct prediction
            pred = self.model.predict(last_features)[0]
            if pred > 0:
                return Signal.BUY
            elif pred < 0:
                return Signal.SELL

        return Signal.HOLD


class SMAFilterStrategy(FilterStrategy):
    """
    Simple SMA filter strategy.

    Passes stocks that are above their moving average (uptrend).
    """

    def __init__(self, period: int = 50):
        """Initialize SMA filter."""
        super().__init__(period=period)
        self.period = period
        self.sma = SMA(period=period)

    @property
    def name(self) -> str:
        return f"sma_filter_{self.period}"

    @property
    def required_history(self) -> int:
        return self.period + 1

    def precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result[f"sma_{self.period}"] = self.sma(df)
        return result

    def screen(self, data: dict[str, pd.DataFrame]) -> list[str]:
        """Return tickers trading above their SMA."""
        passed = []

        for ticker, df in data.items():
            if len(df) < self.required_history:
                continue

            sma_col = f"sma_{self.period}"
            if sma_col in df.columns:
                sma = df[sma_col].iloc[-1]
            else:
                sma = self.sma(df).iloc[-1]

            if pd.notna(sma) and df["close"].iloc[-1] > sma:
                passed.append(ticker)

        return passed


class VolumeFilterStrategy(FilterStrategy):
    """
    Volume filter - only trade stocks with sufficient volume.
    """

    def __init__(self, min_avg_volume: int = 1_000_000, lookback: int = 20):
        """Initialize volume filter."""
        super().__init__(min_avg_volume=min_avg_volume, lookback=lookback)
        self.min_avg_volume = min_avg_volume
        self.lookback = lookback

    @property
    def name(self) -> str:
        return f"volume_filter_{self.min_avg_volume}"

    @property
    def required_history(self) -> int:
        return self.lookback

    def screen(self, data: dict[str, pd.DataFrame]) -> list[str]:
        """Return tickers with sufficient average volume."""
        passed = []

        for ticker, df in data.items():
            if len(df) < self.required_history:
                continue

            avg_volume = df["volume"].iloc[-self.lookback:].mean()
            if avg_volume >= self.min_avg_volume:
                passed.append(ticker)

        return passed


def create_ml_strategy(
    train_data: dict[str, pd.DataFrame],
    model_class: type = RandomForestModel,
    use_sma_filter: bool = True,
    sma_period: int = 50,
    target_horizon: int = 5,
    **model_kwargs,
) -> MLStrategy:
    """
    Factory function to create and train an ML strategy.

    Args:
        train_data: Historical data for training
        model_class: Model class to use
        use_sma_filter: Whether to use SMA filter
        sma_period: Period for SMA filter
        target_horizon: Days ahead for prediction
        **model_kwargs: Additional model parameters

    Returns:
        Trained MLStrategy
    """
    # Create feature engineering
    fe = FeatureEngineering()

    # Combine all training data
    all_features = []
    all_labels = []

    for ticker, df in train_data.items():
        feature_set = fe.create_features(df, target_horizon=target_horizon)
        if not feature_set.X.empty and feature_set.y is not None:
            # Convert to classification
            labels = (feature_set.y > 0).astype(int) * 2 - 1
            all_features.append(feature_set.X)
            all_labels.append(labels)

    if not all_features:
        raise ValueError("No valid training data")

    # Combine
    X = pd.concat(all_features, axis=0)
    y = pd.concat(all_labels, axis=0)

    combined_set = FeatureSet(X=X, y=y, feature_names=list(X.columns))

    # Train model
    model = model_class(**model_kwargs)
    model.fit(combined_set)

    # Create filter
    filter_strategy = SMAFilterStrategy(period=sma_period) if use_sma_filter else None

    return MLStrategy(
        model=model,
        feature_engineering=fe,
        filter_strategy=filter_strategy,
    )
