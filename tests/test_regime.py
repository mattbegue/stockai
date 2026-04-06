"""Tests for market regime classifier and regime-aware strategy."""

import numpy as np
import pandas as pd
import pytest

from futures.regime.classifier import MarketRegime, MarketRegimeClassifier, RegimeReading
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bull_market(n: int = 300) -> dict[str, pd.DataFrame]:
    """SPY in a steady uptrend, VXX low, HYG outperforming TLT."""
    dates = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(0)

    def trending_ohlcv(start, drift, seed):
        r = np.random.default_rng(seed)
        close = start + np.cumsum(r.normal(drift, 1.0, n))
        noise = r.uniform(0.3, 1.5, n)
        df = pd.DataFrame(
            {"open": close - r.uniform(0, 0.5, n), "high": close + noise,
             "low": close - noise, "close": close, "volume": r.integers(1e6, 5e6, n)},
            index=dates,
        )
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        return df

    return {
        "SPY": trending_ohlcv(300, +0.3, 1),   # Rising
        "VXX": trending_ohlcv(20, -0.05, 2),   # Falling VXX = low vol
        "TLT": trending_ohlcv(100, -0.1, 3),   # Falling TLT
        "HYG": trending_ohlcv(80, +0.2, 4),    # Rising HYG = risk-on
    }


def make_bear_market(n: int = 300) -> dict[str, pd.DataFrame]:
    """SPY in a downtrend, VXX high, TLT outperforming HYG."""
    dates = pd.bdate_range("2020-01-02", periods=n)

    def trending_ohlcv(start, drift, seed):
        r = np.random.default_rng(seed)
        close = np.maximum(start + np.cumsum(r.normal(drift, 1.5, n)), 1.0)
        noise = r.uniform(0.5, 2.5, n)
        df = pd.DataFrame(
            {"open": close - r.uniform(0, 1.0, n), "high": close + noise,
             "low": close - noise, "close": close, "volume": r.integers(1e6, 5e6, n)},
            index=dates,
        )
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        return df

    return {
        "SPY": trending_ohlcv(300, -0.4, 10),  # Falling
        "VXX": trending_ohlcv(20, +0.15, 11),  # Rising VXX = high vol
        "TLT": trending_ohlcv(100, +0.2, 12),  # Rising TLT = risk-off
        "HYG": trending_ohlcv(80, -0.15, 13),  # Falling HYG
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMarketRegimeClassifier:
    def test_bull_market_classified_as_bull(self):
        data = make_bull_market(300)
        clf = MarketRegimeClassifier()
        reading = clf.classify(data)
        assert reading.regime == MarketRegime.BULL, (
            f"Expected BULL, got {reading.regime.value} (score={reading.score})"
        )

    def test_bear_market_classified_as_bear(self):
        data = make_bear_market(300)
        clf = MarketRegimeClassifier()
        reading = clf.classify(data)
        assert reading.regime == MarketRegime.BEAR, (
            f"Expected BEAR, got {reading.regime.value} (score={reading.score})"
        )

    def test_score_range(self):
        """Score must stay within [-5, +5]."""
        for data in [make_bull_market(300), make_bear_market(300)]:
            clf = MarketRegimeClassifier()
            reading = clf.classify(data)
            assert -5 <= reading.score <= 5

    def test_score_pct_range(self):
        """Normalised score_pct must be in [0, 1]."""
        clf = MarketRegimeClassifier()
        for data in [make_bull_market(300), make_bear_market(300)]:
            reading = clf.classify(data)
            assert 0.0 <= reading.score_pct <= 1.0

    def test_components_present(self):
        """All expected component keys must be in the reading."""
        clf = MarketRegimeClassifier()
        reading = clf.classify(make_bull_market(300))
        expected_keys = {"spy_above_sma50", "spy_above_sma200", "spy_momentum_20d", "vxx_below_avg63"}
        assert expected_keys.issubset(set(reading.components.keys()))

    def test_fallback_without_vxx(self):
        """Classifier must not crash when VXX is absent."""
        data = make_bull_market(300)
        data.pop("VXX")
        clf = MarketRegimeClassifier()
        reading = clf.classify(data)  # Should not raise
        assert reading.regime in MarketRegime

    def test_fallback_without_tlt_hyg(self):
        """Classifier must not crash when TLT/HYG are absent."""
        data = make_bull_market(300)
        data.pop("TLT")
        data.pop("HYG")
        clf = MarketRegimeClassifier()
        reading = clf.classify(data)
        assert reading.regime in MarketRegime

    def test_insufficient_spy_returns_neutral(self):
        """Too few SPY bars returns NEUTRAL with error in components."""
        data = {"SPY": make_ohlcv(n=5)}
        clf = MarketRegimeClassifier()
        reading = clf.classify(data)
        assert reading.regime == MarketRegime.NEUTRAL
        assert "error" in reading.components

    def test_classify_series_matches_pointwise(self):
        """classify_series must match pointwise classify calls for each date."""
        data = make_bull_market(300)
        clf = MarketRegimeClassifier()
        spy_dates = data["SPY"].index[200:]  # Use last 100 dates (past warmup)

        series = clf.classify_series(data, dates=spy_dates)

        for date in spy_dates[:5]:  # Spot-check first 5
            pointwise = clf.classify(data, as_of_date=date)
            assert series.loc[date, "regime"] == pointwise.regime.value
            assert series.loc[date, "score"] == pointwise.score

    def test_as_of_date_uses_only_past_data(self):
        """
        Classifying at date T must give same result whether or not
        future data exists — regime is purely backward-looking.
        """
        data_full = make_bull_market(300)
        data_short = {k: v.iloc[:200] for k, v in data_full.items()}
        as_of = data_short["SPY"].index[-1]

        clf = MarketRegimeClassifier()
        reading_full = clf.classify(data_full, as_of_date=as_of)
        reading_short = clf.classify(data_short, as_of_date=as_of)

        assert reading_full.regime == reading_short.regime
        assert reading_full.score == reading_short.score


class TestRegimeAwareStrategy:
    """Integration tests verifying regime gates signals correctly."""

    def _make_strategy_with_mock_model(self, confidence: float, regime_classifier=None):
        """Create a MetalabelingStrategy backed by a constant-confidence model."""
        from unittest.mock import MagicMock
        import numpy as np
        from futures.strategies.metalabeling_strategy import MetalabelingStrategy

        model = MagicMock()
        model.feature_names = []
        # predict_proba always returns [[1-conf, conf]]
        model.predict_proba.return_value = np.array([[1 - confidence, confidence]])

        return MetalabelingStrategy(
            meta_model=model,
            confidence_threshold=0.60,
            regime_classifier=regime_classifier,
            base_position_size=0.10,
        )

    def test_effective_threshold_bull(self):
        """In BULL regime, effective threshold drops by 10%."""
        from futures.strategies.metalabeling_strategy import MetalabelingStrategy
        from unittest.mock import MagicMock

        model = MagicMock()
        model.feature_names = []
        strategy = MetalabelingStrategy(meta_model=model, confidence_threshold=0.60)
        assert strategy._effective_threshold(MarketRegime.BULL) == pytest.approx(0.54)

    def test_effective_threshold_bear(self):
        """In BEAR regime, effective threshold raises to at least 0.80."""
        from futures.strategies.metalabeling_strategy import MetalabelingStrategy
        from unittest.mock import MagicMock

        model = MagicMock()
        model.feature_names = []
        strategy = MetalabelingStrategy(meta_model=model, confidence_threshold=0.60)
        assert strategy._effective_threshold(MarketRegime.BEAR) == pytest.approx(0.80)

    def test_effective_threshold_bear_honours_higher_user_threshold(self):
        """If user sets threshold > 0.80, bear regime keeps user's threshold."""
        from futures.strategies.metalabeling_strategy import MetalabelingStrategy
        from unittest.mock import MagicMock

        model = MagicMock()
        model.feature_names = []
        strategy = MetalabelingStrategy(meta_model=model, confidence_threshold=0.85)
        assert strategy._effective_threshold(MarketRegime.BEAR) == pytest.approx(0.85)

    def test_position_sizes_scale_with_confidence(self):
        """Higher confidence → larger position size, capped at 2× base."""
        from futures.strategies.metalabeling_strategy import MetalabelingStrategy
        from unittest.mock import MagicMock
        from futures.strategies.base import Signal

        model = MagicMock()
        model.feature_names = []
        strategy = MetalabelingStrategy(
            meta_model=model, confidence_threshold=0.60, base_position_size=0.10
        )

        # Manually inject confidence scores
        strategy._last_confidences = {"AAPL": 0.90, "MSFT": 0.60, "GOOG": 0.30}
        signals = {"AAPL": Signal.BUY, "MSFT": Signal.BUY, "GOOG": Signal.BUY}
        sizes = strategy.get_position_sizes(signals)

        # AAPL: 0.90/0.60 = 1.5× → 0.15
        assert sizes["AAPL"] == pytest.approx(0.15)
        # MSFT: 0.60/0.60 = 1.0× → 0.10
        assert sizes["MSFT"] == pytest.approx(0.10)
        # GOOG: 0.30/0.60 = 0.5× → 0.05 (clamped at 0.5× minimum)
        assert sizes["GOOG"] == pytest.approx(0.05)

    def test_position_sizes_capped_at_2x(self):
        """Confidence cannot push position size above 2× base."""
        from futures.strategies.metalabeling_strategy import MetalabelingStrategy
        from unittest.mock import MagicMock
        from futures.strategies.base import Signal

        model = MagicMock()
        model.feature_names = []
        strategy = MetalabelingStrategy(
            meta_model=model, confidence_threshold=0.60, base_position_size=0.10
        )
        strategy._last_confidences = {"AAPL": 0.99}
        sizes = strategy.get_position_sizes({"AAPL": Signal.BUY})
        assert sizes["AAPL"] <= 0.20  # 2× cap

    def test_sell_signals_not_sized(self):
        """SELL signals are not included in position sizes (long-only)."""
        from futures.strategies.metalabeling_strategy import MetalabelingStrategy
        from unittest.mock import MagicMock
        from futures.strategies.base import Signal

        model = MagicMock()
        model.feature_names = []
        strategy = MetalabelingStrategy(meta_model=model, confidence_threshold=0.60)
        strategy._last_confidences = {"AAPL": 0.90}
        sizes = strategy.get_position_sizes({"AAPL": Signal.SELL})
        assert "AAPL" not in sizes
