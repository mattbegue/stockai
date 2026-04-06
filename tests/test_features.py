"""Tests for feature engineering — verify no look-ahead bias in feature values."""

import numpy as np
import pandas as pd
import pytest

from futures.metalabeling.features import MetaFeatureEngineering
from futures.metalabeling.signals import CandidateSignal, Signal
from futures.ml.features import FeatureEngineering
from tests.conftest import make_ohlcv


def _make_candidate(ticker, date, direction=Signal.BUY):
    return CandidateSignal(
        ticker=ticker,
        date=date,
        direction=direction,
        source_indicators=["sma_crossover"],
    )


class TestMetaFeatureEngineering:
    def test_features_do_not_use_post_signal_data(self):
        """
        Perturbing price data AFTER the signal date must not change features.

        This is the canonical no-lookahead test: if future data leaks into
        features, features would change when we alter future prices.
        """
        df = make_ohlcv(n=80)
        context = {"SPY": make_ohlcv(n=80, seed=10)}

        signal_date = df.index[50]
        candidate = _make_candidate("AAPL", signal_date)

        fe = MetaFeatureEngineering(context_tickers=["SPY"])
        df_with_ind = fe.compute_indicators_for_df(df)

        # Compute baseline features
        feat_before = fe.create_features_for_signal(
            df_with_ind, context, "AAPL", signal_date, 1, ["sma_crossover"]
        )

        # Corrupt all prices AFTER signal date
        df_corrupted = df.copy()
        df_corrupted.loc[df_corrupted.index > signal_date, "close"] *= 10
        df_corrupted.loc[df_corrupted.index > signal_date, "open"] *= 10
        df_corrupted_with_ind = fe.compute_indicators_for_df(df_corrupted)

        feat_after = fe.create_features_for_signal(
            df_corrupted_with_ind, context, "AAPL", signal_date, 1, ["sma_crossover"]
        )

        # Features should be identical (future data must not matter)
        common = feat_before.index.intersection(feat_after.index)
        assert len(common) > 0, "No common features to compare"
        pd.testing.assert_series_equal(
            feat_before[common].fillna(-999),
            feat_after[common].fillna(-999),
            check_names=False,
            atol=1e-8,
        )

    def test_feature_vector_non_empty(self):
        """Feature computation must return a non-empty Series for a valid signal."""
        df = make_ohlcv(n=80)
        context = {"SPY": make_ohlcv(n=80, seed=10), "VXX": make_ohlcv(n=80, seed=11)}

        fe = MetaFeatureEngineering(context_tickers=["SPY", "VXX"])
        df_with_ind = fe.compute_indicators_for_df(df)

        feat = fe.create_features_for_signal(
            df_with_ind, context, "AAPL", df.index[60], 1, ["sma_crossover"]
        )
        assert len(feat) > 0

    def test_feature_matrix_row_count(self):
        """Feature matrix has one row per labeled signal."""
        from futures.metalabeling.labels import create_metalabels

        df = make_ohlcv(n=80)
        context_df = make_ohlcv(n=80, seed=10)
        data = {"AAPL": df, "SPY": context_df}

        candidates = [_make_candidate("AAPL", df.index[i]) for i in range(10, 50)]
        from futures.metalabeling.labels import create_metalabels

        labeled = create_metalabels(data, candidates, holding_period=5)

        fe = MetaFeatureEngineering(context_tickers=["SPY"])
        matrix = fe.create_feature_matrix(labeled, data, show_progress=False)

        assert len(matrix) == len(labeled)

    def test_spy_return_uses_current_bar(self):
        """spy_return_5d must reflect current-bar close, not prior bar."""
        df = make_ohlcv(n=80)

        # Build two SPY DataFrames: identical except on the signal date
        spy_a = make_ohlcv(n=80, seed=10).copy()
        spy_b = spy_a.copy()

        signal_date = spy_a.index[60]
        spy_b.loc[signal_date, "close"] = spy_a.loc[signal_date, "close"] * 1.10

        fe = MetaFeatureEngineering(context_tickers=["SPY"])
        df_ind = fe.compute_indicators_for_df(df)

        feat_a = fe.create_features_for_signal(df_ind, {"SPY": spy_a}, "AAPL", signal_date, 1, [])
        feat_b = fe.create_features_for_signal(df_ind, {"SPY": spy_b}, "AAPL", signal_date, 1, [])

        # If we're using current bar, changing signal-date SPY close changes spy_return_5d
        assert feat_a.get("spy_return_5d") != feat_b.get("spy_return_5d"), (
            "spy_return_5d must use current bar's close price"
        )


class TestMLFeatureEngineering:
    def test_target_not_in_feature_columns(self):
        """The forward return target must not appear as a column in X."""
        df = make_ohlcv(n=100)
        fe = FeatureEngineering()
        fs = fe.create_features(df, target_horizon=5)

        assert "target" not in fs.X.columns
        assert "forward_return" not in fs.X.columns

    def test_no_future_data_in_features(self):
        """
        Appending future rows must not change features for any existing row.

        If features at row T depend only on data through T, adding rows T+1..T+N
        must leave row T's features unchanged.

        We use a large df so both the full and truncated versions have valid
        rows (after SMA50 warmup and target_horizon NaN tail removal).
        """
        df = make_ohlcv(n=200)
        df_short = df.iloc[:150]  # 150 bars: ~100 valid rows after warmup and 5-day target tail

        fe = FeatureEngineering(lookback_periods=[5, 10, 20, 50], forward_return_periods=[5])
        fs_full = fe.create_features(df, target_horizon=5)
        fs_short = fe.create_features(df_short, target_horizon=5)

        common_idx = fs_full.X.index.intersection(fs_short.X.index)
        assert len(common_idx) > 0, (
            "Both feature sets must have valid rows with a large enough df"
        )

        pd.testing.assert_frame_equal(
            fs_full.X.loc[common_idx].reset_index(drop=True),
            fs_short.X.loc[common_idx].reset_index(drop=True),
            atol=1e-8,
        )

    def test_create_labels_surfaces_nan_at_tail(self):
        """create_labels must return NaN for the last `horizon` rows."""
        df = make_ohlcv(n=30)
        fe = FeatureEngineering()
        labels = fe.create_labels(df, horizon=5)

        assert labels.iloc[-5:].isna().all(), (
            "Last 5 rows must be NaN — no forward data available"
        )
        assert labels.iloc[:-5].notna().all(), "All earlier rows should have valid labels"

    def test_feature_set_y_length_matches_x(self):
        """X and y must have the same length (NaN rows are jointly dropped)."""
        df = make_ohlcv(n=100)
        fe = FeatureEngineering()
        fs = fe.create_features(df, target_horizon=5)

        assert len(fs.X) == len(fs.y)
