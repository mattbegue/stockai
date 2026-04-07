"""Metalabeling strategy that combines technical signals with ML filtering."""

from typing import Optional

import pandas as pd

from futures.ml.models import ModelWrapper
from futures.ml.features import FeatureSet
from futures.metalabeling.signals import PrimarySignalGenerator
from futures.metalabeling.features import MetaFeatureEngineering
from futures.regime.classifier import MarketRegimeClassifier, MarketRegime
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
        regime_classifier: Optional[MarketRegimeClassifier] = None,
        base_position_size: float = 0.10,
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
            regime_classifier: Market regime classifier. When provided:
                - BEAR: BUY threshold raised to max(threshold, 0.80); capital preserved
                - BULL: BUY threshold lowered by 10% (capture more edge)
                - NEUTRAL: threshold unchanged
            base_position_size: Base position size as fraction of portfolio.
                Actual size scales with confidence: base × min(conf/threshold, 2.0).
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
        self.feature_names = feature_names or meta_model.feature_names
        self.regime_classifier = regime_classifier
        self.base_position_size = base_position_size

        # Cache for precomputed indicators
        self._indicator_cache: dict[str, pd.DataFrame] = {}
        # Populated during generate_signals(); read by get_position_sizes()
        self._last_confidences: dict[str, float] = {}
        # Source indicators for BUY signals; read by get_position_holding_days()
        self._last_source_indicators: dict[str, list[str]] = {}

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

    def _effective_threshold(self, regime: MarketRegime) -> float:
        """Adjust confidence threshold based on market regime."""
        if regime == MarketRegime.BEAR:
            # In bear markets, only take BUY trades we're very confident about
            return max(self.confidence_threshold, 0.80)
        elif regime == MarketRegime.BULL:
            # In bull markets, capture slightly more edge
            return self.confidence_threshold * 0.90
        return self.confidence_threshold

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """
        Generate trading signals using metalabeling approach.

        Signal generation pipeline:
          1. Classify market regime (if regime_classifier provided)
          2. Adjust effective confidence threshold for regime
          3. Run primary signal generator
          4. Score each candidate with meta-model
          5. Emit BUY/SELL only when confidence >= effective threshold

        Confidence scores are stored in self._last_confidences for downstream
        position sizing via get_position_sizes().
        """
        signals = {}
        self._last_confidences = {}
        self._last_source_indicators = {}

        # Separate tradeable stocks from context ETFs
        context_tickers_set = set(self.context_tickers)
        tradeable_data = {t: df for t, df in data.items() if t not in context_tickers_set}
        context_data = {t: df for t, df in data.items() if t in context_tickers_set}

        # --- Regime classification ---
        if self.regime_classifier is not None:
            regime_reading = self.regime_classifier.classify(data)
            current_regime = regime_reading.regime
        else:
            current_regime = MarketRegime.NEUTRAL

        effective_threshold = self._effective_threshold(current_regime)

        # Get primary signals for current bar
        primary_signals = self.signal_generator.get_current_signals(tradeable_data)

        for ticker, df in tradeable_data.items():
            signals[ticker] = Signal.HOLD

            if len(df) < self.required_history:
                continue

            primary_signal, source_indicators = primary_signals.get(ticker, (None, []))
            if primary_signal is None:
                continue

            try:
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

                # Hard earnings filter (P2-4): skip any signal whose holding
                # window straddles an earnings release — jump risk dominates
                if features.get("earnings_within_hold", 0.0) == 1.0:
                    continue

                feature_df = pd.DataFrame([features]).fillna(0)

                if self.feature_names:
                    aligned_df = pd.DataFrame(index=feature_df.index)
                    for feat in self.feature_names:
                        aligned_df[feat] = feature_df[feat] if feat in feature_df.columns else 0
                    feature_df = aligned_df

                feature_input = FeatureSet(
                    X=feature_df,
                    feature_names=list(feature_df.columns),
                )

                proba = self.meta_model.predict_proba(feature_input)
                confidence = float(proba[0, 1] if proba.shape[1] == 2 else proba[0, 0])

                # Store confidence regardless of whether signal fires
                self._last_confidences[ticker] = confidence

                if confidence >= effective_threshold:
                    if primary_signal.value == 1:
                        signals[ticker] = Signal.BUY
                        self._last_source_indicators[ticker] = source_indicators
                    elif primary_signal.value == -1:
                        signals[ticker] = Signal.SELL

            except Exception:
                continue

        for ticker in context_tickers_set:
            if ticker in data:
                signals[ticker] = Signal.HOLD

        return signals

    def get_position_sizes(self, signals: dict[str, Signal]) -> dict[str, float]:
        """
        Return confidence-scaled position sizes for BUY signals.

        Size = base_position_size × clamp(confidence / threshold, 0.5, 2.0)

        A trade at exactly the threshold gets the base size. A trade with
        twice the threshold's confidence gets 2× the base size (capped).
        Trades at 50% above threshold (e.g. 0.90 vs 0.60) get 1.5× base.
        """
        sizes = {}
        for ticker, signal in signals.items():
            if signal == Signal.BUY and ticker in self._last_confidences:
                confidence = self._last_confidences[ticker]
                scale = max(0.5, min(confidence / self.confidence_threshold, 2.0))
                sizes[ticker] = self.base_position_size * scale
        return sizes

    # Holding period (in trading days) by signal type (P2-10)
    _MOMENTUM_SIGNALS = {"sma_crossover", "volume_breakout", "roc_reversal"}
    _REVERSION_SIGNALS = {"rsi_oversold", "rsi_overbought", "bb_lower", "bb_upper", "vwap_cross"}
    # Everything else (macd_crossover, stoch_crossover, obv_cross, …) → 5 days

    def get_position_holding_days(self, signals: dict[str, "Signal"]) -> dict[str, int]:
        """
        Return asymmetric holding periods based on the primary signal type.

        Mean-reversion signals exit quickly before the snap-back fades.
        Momentum and neutral signals use 5 days (matching training labels).
        Extending momentum beyond 5 days risks holding past the triple-barrier
        horizon that the model was trained on.

        Classification (trading days):
          - Mean-reversion (rsi_*, bb_*, vwap_cross)                → 3
          - Momentum (sma_crossover, volume_breakout, roc_reversal) → 5
          - Neutral / mixed (macd_crossover, stoch_crossover, …)    → 5
        """
        holding_days: dict[str, int] = {}
        for ticker, signal in signals.items():
            if signal != Signal.BUY:
                continue
            indicators = self._last_source_indicators.get(ticker, [])
            if any(ind in self._REVERSION_SIGNALS for ind in indicators):
                holding_days[ticker] = 3
            else:
                holding_days[ticker] = 5
        return holding_days

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
