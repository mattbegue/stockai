"""Tests for metalabeling label creation."""

import numpy as np
import pandas as pd
import pytest

from futures.metalabeling.labels import create_metalabels, get_label_statistics
from futures.metalabeling.signals import CandidateSignal, Signal
from tests.conftest import make_ohlcv


def _make_candidate(ticker: str, date: pd.Timestamp, direction: Signal = Signal.BUY):
    return CandidateSignal(
        ticker=ticker,
        date=date,
        direction=direction,
        source_indicators=["sma_crossover"],
    )


class TestCreateMetalabels:
    def test_entry_is_next_day_open(self):
        """Entry price must be the OPEN of the day after the signal, not signal-day close."""
        df = make_ohlcv(n=30)
        signal_date = df.index[10]
        entry_idx = 11  # Next bar

        candidate = _make_candidate("AAPL", signal_date)
        result = create_metalabels({"AAPL": df}, [candidate], holding_period=5)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["entry_price"] == pytest.approx(df["open"].iloc[entry_idx])
        assert row["entry_price"] != pytest.approx(df["close"].iloc[10]), (
            "Entry price must NOT be signal-day close (next-open execution)"
        )

    def test_exit_is_close_after_holding_period(self):
        """Exit price must be the close holding_period days after entry, not after signal."""
        df = make_ohlcv(n=30)
        signal_date = df.index[10]
        # entry at idx=11, exit at idx=11+5=16
        expected_exit_idx = 16
        holding = 5

        candidate = _make_candidate("AAPL", signal_date)
        result = create_metalabels({"AAPL": df}, [candidate], holding_period=holding)

        row = result.iloc[0]
        assert row["exit_price"] == pytest.approx(df["close"].iloc[expected_exit_idx])

    def test_buy_forward_return_direction(self):
        """BUY label: positive return when price rises."""
        df = make_ohlcv(n=30)
        # Force a rising path: manually set open/close for the relevant window
        df = df.copy()
        signal_date = df.index[10]
        entry_idx = 11
        exit_idx = 16

        df.loc[df.index[entry_idx], "open"] = 100.0
        df.loc[df.index[exit_idx], "close"] = 110.0  # +10% gain
        df.loc[df.index[entry_idx], "high"] = max(df["high"].iloc[entry_idx], 110.1)

        candidate = _make_candidate("AAPL", signal_date, direction=Signal.BUY)
        result = create_metalabels({"AAPL": df}, [candidate], holding_period=5, min_return=0.0)

        row = result.iloc[0]
        assert row["forward_return"] > 0
        assert row["label"] == 1

    def test_sell_forward_return_direction(self):
        """SELL label: positive return when price falls (short logic)."""
        df = make_ohlcv(n=30)
        df = df.copy()
        signal_date = df.index[10]
        entry_idx = 11
        exit_idx = 16

        df.loc[df.index[entry_idx], "open"] = 100.0
        df.loc[df.index[exit_idx], "close"] = 90.0  # Price fell: profitable for short

        candidate = _make_candidate("AAPL", signal_date, direction=Signal.SELL)
        result = create_metalabels({"AAPL": df}, [candidate], holding_period=5, min_return=0.0)

        row = result.iloc[0]
        assert row["forward_return"] > 0
        assert row["label"] == 1

    def test_insufficient_forward_data_skipped(self):
        """Signals near the end of data are skipped (no room for entry + exit)."""
        df = make_ohlcv(n=20)
        # Signal at second-to-last bar: entry would be last bar, exit out of bounds
        signal_date = df.index[-2]

        candidate = _make_candidate("AAPL", signal_date)
        result = create_metalabels({"AAPL": df}, [candidate], holding_period=5)

        assert len(result) == 0, "Signal with no room for exit must be dropped"

    def test_missing_ticker_skipped(self):
        """Candidates for tickers not in data are silently skipped."""
        df = make_ohlcv(n=30)
        candidate = _make_candidate("MISSING", df.index[10])
        result = create_metalabels({"AAPL": df}, [candidate])
        assert len(result) == 0

    def test_min_return_threshold(self):
        """Label is 0 if forward return doesn't exceed min_return."""
        df = make_ohlcv(n=30)
        df = df.copy()
        signal_date = df.index[10]
        df.loc[df.index[11], "open"] = 100.0
        df.loc[df.index[16], "close"] = 100.1  # 0.1% — below 0.3% threshold

        candidate = _make_candidate("AAPL", signal_date)
        result = create_metalabels({"AAPL": df}, [candidate], holding_period=5, min_return=0.003)

        assert result.iloc[0]["label"] == 0

    def test_statistics_match_labels(self):
        """get_label_statistics counts match the actual label column."""
        df = make_ohlcv(n=60)
        candidates = [_make_candidate("AAPL", df.index[i]) for i in range(5, 45)]
        result = create_metalabels({"AAPL": df}, candidates, holding_period=5)

        stats = get_label_statistics(result)
        assert stats["total"] == len(result)
        assert stats["profitable"] == result["label"].sum()
