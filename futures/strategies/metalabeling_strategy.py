"""Metalabeling strategy that combines technical signals with ML filtering."""

from typing import Optional

import pandas as pd

from futures.ml.models import ModelWrapper
from futures.ml.features import FeatureSet
from futures.metalabeling.signals import PrimarySignalGenerator
from futures.metalabeling.features import MetaFeatureEngineering
from .base import Strategy, Signal


class MetalabelingStrategy(Strategy):
    """
    Two-stage trading strategy using metalabeling.

    1. Primary signal generator produces trade candidates from technical indicators
    2. Meta-model predicts which candidates are likely to be profitable
    3. Only trades with high meta-model confidence are executed

    This approach lets simple indicators generate many candidates while
    using ML to filter down to high-quality trades.
    """

    def __init__(
        self,
        meta_model: ModelWrapper,
        signal_generator: Optional[PrimarySignalGenerator] = None,
        feature_engineering: Optional[MetaFeatureEngineering] = None,
        confidence_threshold: float = 0.6,
        context_tickers: Optional[list[str]] = None,
        feature_names: Optional[list[str]] = None,
    ):
        """
        Initialize the metalabeling strategy.

        Args:
            meta_model: Trained model that predicts signal profitability
            signal_generator: Primary signal generator (uses default if None)
            feature_engineering: Feature engineering pipeline (uses default if None)
            confidence_threshold: Minimum meta-model confidence to take a trade
            context_tickers: ETF tickers for market context features
            feature_names: List of feature names from training (for alignment)
        """
        super().__init__(
            confidence_threshold=confidence_threshold,
            context_tickers=context_tickers,
        )
        self.meta_model = meta_model
        self.signal_generator = signal_generator or PrimarySignalGenerator()
        self.feature_engineering = feature_engineering or MetaFeatureEngineering(
            context_tickers=context_tickers
        )
        self.confidence_threshold = confidence_threshold
        self.context_tickers = context_tickers or [
            "SPY", "QQQ", "VXX", "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY",
            "TLT", "HYG", "GLD",
        ]
        # Feature names from training for alignment
        self.feature_names = feature_names or meta_model.feature_names

        # Cache for precomputed indicators
        self._indicator_cache: dict[str, pd.DataFrame] = {}

    @property
    def name(self) -> str:
        return f"metalabeling_conf{self.confidence_threshold}"

    @property
    def required_history(self) -> int:
        # Need enough history for 50-SMA plus buffer
        return max(60, self.signal_generator.required_history)

    def precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns needed for signal generation and features."""
        return self.signal_generator.compute_indicators(df)

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """
        Generate trading signals using metalabeling approach.

        Args:
            data: Dict mapping ticker to OHLCV DataFrame
                  Should include both tradeable stocks and context ETFs

        Returns:
            Dict mapping ticker to Signal
        """
        signals = {}

        # Separate tradeable stocks from context ETFs
        context_tickers_set = set(self.context_tickers)
        tradeable_data = {
            t: df for t, df in data.items() if t not in context_tickers_set
        }
        context_data = {
            t: df for t, df in data.items() if t in context_tickers_set
        }

        # Get primary signals for current bar
        primary_signals = self.signal_generator.get_current_signals(tradeable_data)

        for ticker, df in tradeable_data.items():
            # Default to HOLD
            signals[ticker] = Signal.HOLD

            if len(df) < self.required_history:
                continue

            # Check if primary generator has a signal
            primary_signal, source_indicators = primary_signals.get(ticker, (None, []))

            if primary_signal is None:
                continue

            # Compute features for this signal
            try:
                # Ensure indicators are computed
                if ticker in self._indicator_cache:
                    df_with_indicators = self._indicator_cache[ticker]
                else:
                    df_with_indicators = self.feature_engineering.compute_indicators_for_df(df)

                signal_date = df_with_indicators.index[-1]

                features = self.feature_engineering.create_features_for_signal(
                    df_with_indicators,
                    context_data,
                    ticker,
                    signal_date,
                    primary_signal.value,
                    source_indicators,
                )

                if features.empty:
                    continue

                # Get meta-model prediction
                feature_df = pd.DataFrame([features]).fillna(0)

                # Align features to training order
                if self.feature_names:
                    aligned_df = pd.DataFrame(index=feature_df.index)
                    for feat in self.feature_names:
                        if feat in feature_df.columns:
                            aligned_df[feat] = feature_df[feat]
                        else:
                            aligned_df[feat] = 0  # Missing features get 0
                    feature_df = aligned_df

                # Create FeatureSet for model
                feature_input = FeatureSet(
                    X=feature_df,
                    feature_names=list(feature_df.columns),
                )

                # Get probability of profitable trade
                proba = self.meta_model.predict_proba(feature_input)

                # Binary classification: probability of class 1 (profitable)
                if proba.shape[1] == 2:
                    confidence = proba[0, 1]
                else:
                    confidence = proba[0, 1] if proba.shape[1] > 1 else proba[0, 0]

                # Only take the trade if confidence exceeds threshold
                if confidence >= self.confidence_threshold:
                    # Convert from metalabeling Signal enum to strategies Signal enum
                    if primary_signal.value == 1:
                        signals[ticker] = Signal.BUY
                    elif primary_signal.value == -1:
                        signals[ticker] = Signal.SELL

            except Exception as e:
                # Log error but don't crash - just skip this ticker
                # In production, you'd want proper logging here
                continue

        # Context ETFs always HOLD (we don't trade them)
        for ticker in context_tickers_set:
            if ticker in data:
                signals[ticker] = Signal.HOLD

        return signals

    def get_signal_details(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, dict]:
        """
        Get detailed signal information including confidence scores.

        Useful for analysis and debugging.

        Args:
            data: Dict mapping ticker to OHLCV DataFrame

        Returns:
            Dict mapping ticker to signal details
        """
        details = {}

        context_tickers_set = set(self.context_tickers)
        tradeable_data = {
            t: df for t, df in data.items() if t not in context_tickers_set
        }
        context_data = {
            t: df for t, df in data.items() if t in context_tickers_set
        }

        primary_signals = self.signal_generator.get_current_signals(tradeable_data)

        for ticker, df in tradeable_data.items():
            primary_signal, source_indicators = primary_signals.get(ticker, (None, []))

            detail = {
                "primary_signal": primary_signal.name if primary_signal else None,
                "source_indicators": source_indicators,
                "confidence": None,
                "final_signal": Signal.HOLD.name,
                "reason": "no_primary_signal" if primary_signal is None else None,
            }

            if primary_signal is not None and len(df) >= self.required_history:
                try:
                    df_with_indicators = self.feature_engineering.compute_indicators_for_df(df)
                    signal_date = df_with_indicators.index[-1]

                    features = self.feature_engineering.create_features_for_signal(
                        df_with_indicators,
                        context_data,
                        ticker,
                        signal_date,
                        primary_signal.value,
                        source_indicators,
                    )

                    if not features.empty:
                        feature_df = pd.DataFrame([features]).fillna(0)

                        # Align features to training order
                        if self.feature_names:
                            aligned_df = pd.DataFrame(index=feature_df.index)
                            for feat in self.feature_names:
                                if feat in feature_df.columns:
                                    aligned_df[feat] = feature_df[feat]
                                else:
                                    aligned_df[feat] = 0
                            feature_df = aligned_df

                        feature_input = FeatureSet(
                            X=feature_df,
                            feature_names=list(feature_df.columns),
                        )
                        proba = self.meta_model.predict_proba(feature_input)
                        confidence = proba[0, 1] if proba.shape[1] == 2 else proba[0, 0]

                        detail["confidence"] = float(confidence)

                        if confidence >= self.confidence_threshold:
                            detail["final_signal"] = primary_signal.name
                            detail["reason"] = "confidence_above_threshold"
                        else:
                            detail["reason"] = "confidence_below_threshold"

                except Exception as e:
                    detail["reason"] = f"error: {str(e)}"

            details[ticker] = detail

        return details


def create_metalabeling_strategy(
    meta_model: ModelWrapper,
    confidence_threshold: float = 0.6,
    context_tickers: Optional[list[str]] = None,
    **signal_generator_params,
) -> MetalabelingStrategy:
    """
    Factory function to create a MetalabelingStrategy.

    Args:
        meta_model: Trained meta-model
        confidence_threshold: Minimum confidence to take trades
        context_tickers: ETF tickers for context features
        **signal_generator_params: Parameters for PrimarySignalGenerator

    Returns:
        Configured MetalabelingStrategy
    """
    signal_generator = PrimarySignalGenerator(**signal_generator_params)
    feature_engineering = MetaFeatureEngineering(context_tickers=context_tickers)

    return MetalabelingStrategy(
        meta_model=meta_model,
        signal_generator=signal_generator,
        feature_engineering=feature_engineering,
        confidence_threshold=confidence_threshold,
        context_tickers=context_tickers,
    )
