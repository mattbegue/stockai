"""Tests for triple-barrier metalabeling label creation."""

import numpy as np
import pandas as pd
import pytest

from futures.metalabeling.labels import create_metalabels, get_label_statistics
from futures.metalabeling.signals import CandidateSignal, Signal
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(ticker: str, date: pd.Timestamp, direction: Signal = Signal.BUY):
    return CandidateSignal(
        ticker=ticker,
        date=date,
        direction=direction,
        source_indicators=["sma_crossover"],
    )


def _flat_ohlcv(n: int = 30, price: float = 100.0) -> pd.DataFrame:
    """OHLCV where open=high=low=close=price — barriers never trigger intraday."""
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1_000_000,
        },
        index=dates,
    )


def _rising_ohlcv(n: int = 30, start: float = 100.0, daily_gain: float = 0.005) -> pd.DataFrame:
    """Price rises by daily_gain each bar. High = close, low = open."""
    dates = pd.bdate_range("2022-01-03", periods=n)
    closes = start * (1 + daily_gain) ** np.arange(n)
    opens = np.roll(closes, 1)
    opens[0] = start
    return pd.DataFrame(
        {"open": opens, "high": closes, "low": opens, "close": closes, "volume": 1_000_000},
        index=dates,
    )


def _falling_ohlcv(n: int = 30, start: float = 100.0, daily_drop: float = 0.005) -> pd.DataFrame:
    """Price falls by daily_drop each bar. Low = close, high = open."""
    dates = pd.bdate_range("2022-01-03", periods=n)
    closes = start * (1 - daily_drop) ** np.arange(n)
    opens = np.roll(closes, 1)
    opens[0] = start
    return pd.DataFrame(
        {"open": opens, "high": opens, "low": closes, "close": closes, "volume": 1_000_000},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Entry / exit mechanics
# ---------------------------------------------------------------------------

class TestEntryExitMechanics:
    def test_entry_is_next_day_open(self):
        """Entry price is the OPEN of the bar after the signal date."""
        df = _flat_ohlcv(n=30, price=100.0)
        signal_date = df.index[10]
        entry_idx = 11

        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", signal_date)],
            holding_period=5, profit_target=0.50, stop_loss=0.50,  # wide — won't trigger
        )

        assert len(result) == 1
        assert result.iloc[0]["entry_price"] == pytest.approx(df["open"].iloc[entry_idx])

    def test_entry_is_not_signal_day_close(self):
        """Entry must NOT equal the signal day's close (no same-bar execution)."""
        df = make_ohlcv(n=30)
        signal_date = df.index[10]

        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", signal_date)],
            holding_period=5, profit_target=0.50, stop_loss=0.50,
        )

        assert result.iloc[0]["entry_price"] != pytest.approx(df["close"].iloc[10])

    def test_time_exit_uses_close_of_holding_period_bar(self):
        """When no barrier is hit, exit is the close of entry + holding_period."""
        df = _flat_ohlcv(n=30, price=100.0)
        signal_date = df.index[10]
        holding = 5
        expected_exit_idx = 11 + holding  # entry at 11, exit at 16

        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", signal_date)],
            holding_period=holding, profit_target=0.50, stop_loss=0.50,
        )

        row = result.iloc[0]
        assert row["exit_type"] == "time"
        assert row["exit_price"] == pytest.approx(df["close"].iloc[expected_exit_idx])
        assert row["bars_held"] == holding

    def test_insufficient_forward_data_skipped(self):
        """Signals at the very end of data are dropped (no room for exit)."""
        df = make_ohlcv(n=20)
        signal_date = df.index[-2]  # entry = last bar, exit out of bounds

        result = create_metalabels({"AAPL": df}, [_make_candidate("AAPL", signal_date)])
        assert len(result) == 0

    def test_missing_ticker_skipped(self):
        """Candidates for unknown tickers are silently dropped."""
        df = make_ohlcv(n=30)
        result = create_metalabels({"AAPL": df}, [_make_candidate("MISSING", df.index[10])])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Triple barrier logic
# ---------------------------------------------------------------------------

class TestTripleBarrier:
    def test_profit_target_triggers_label_1(self):
        """Hitting the upper barrier labels the trade as profitable."""
        # Price rises 2% per day; profit_target=0.015 fires on day 1
        df = _rising_ohlcv(n=30, start=100.0, daily_gain=0.02)
        signal_date = df.index[5]

        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", signal_date)],
            holding_period=5, profit_target=0.015, stop_loss=0.50,
        )

        row = result.iloc[0]
        assert row["exit_type"] == "profit_target"
        assert row["label"] == 1
        assert row["bars_held"] < 5  # exited early

    def test_stop_loss_triggers_label_0(self):
        """Hitting the lower barrier labels the trade as unprofitable."""
        # Price falls 2% per day; stop_loss=0.015 fires on day 1
        df = _falling_ohlcv(n=30, start=100.0, daily_drop=0.02)
        signal_date = df.index[5]

        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", signal_date)],
            holding_period=5, profit_target=0.50, stop_loss=0.015,
        )

        row = result.iloc[0]
        assert row["exit_type"] == "stop_loss"
        assert row["label"] == 0
        assert row["bars_held"] < 5

    def test_time_exit_positive_return_label_1(self):
        """Time exit with return > min_return → label 1."""
        # Flat data → time exit, return = 0
        # Use rising data just above min_return threshold
        df = _rising_ohlcv(n=30, start=100.0, daily_gain=0.001)  # 0.1%/day
        signal_date = df.index[5]

        # Wide barriers so time-exit fires; 5 * 0.1% = 0.5% > min_return=0.003
        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", signal_date)],
            holding_period=5, profit_target=0.50, stop_loss=0.50, min_return=0.003,
        )

        row = result.iloc[0]
        assert row["exit_type"] == "time"
        assert row["forward_return"] > 0.003
        assert row["label"] == 1

    def test_time_exit_below_min_return_label_0(self):
        """Time exit with return < min_return → label 0."""
        df = _flat_ohlcv(n=30, price=100.0)
        signal_date = df.index[5]

        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", signal_date)],
            holding_period=5, profit_target=0.50, stop_loss=0.50, min_return=0.003,
        )

        row = result.iloc[0]
        assert row["exit_type"] == "time"
        assert row["forward_return"] == pytest.approx(0.0)
        assert row["label"] == 0

    def test_stop_loss_takes_priority_over_profit_target_same_bar(self):
        """When both barriers breach on the same bar, stop loss wins (pessimistic)."""
        df = _flat_ohlcv(n=30, price=100.0).copy()
        signal_date = df.index[5]
        entry_idx = 6

        # On day after entry: high well above profit target, low well below stop loss
        df.loc[df.index[entry_idx], "open"] = 100.0
        next_bar = df.index[entry_idx]  # check happens from entry bar onward
        df.loc[next_bar, "high"] = 120.0  # +20%: above any profit target
        df.loc[next_bar, "low"] = 80.0   # -20%: below any stop loss

        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", signal_date)],
            holding_period=5, profit_target=0.05, stop_loss=0.05,
        )

        assert result.iloc[0]["exit_type"] == "stop_loss"
        assert result.iloc[0]["label"] == 0

    def test_exit_type_column_present(self):
        """Output DataFrame must contain exit_type and bars_held columns."""
        df = make_ohlcv(n=30)
        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", df.index[10])],
        )
        assert "exit_type" in result.columns
        assert "bars_held" in result.columns

    def test_exit_price_equals_barrier_on_profit_target(self):
        """Exit price is exactly entry * (1 + profit_target) when upper barrier fires."""
        df = _rising_ohlcv(n=30, start=100.0, daily_gain=0.02)
        signal_date = df.index[5]
        entry_price = df["open"].iloc[6]
        profit_target = 0.015

        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", signal_date)],
            holding_period=5, profit_target=profit_target, stop_loss=0.50,
        )

        row = result.iloc[0]
        assert row["exit_type"] == "profit_target"
        assert row["exit_price"] == pytest.approx(entry_price * (1 + profit_target))

    def test_exit_price_equals_barrier_on_stop_loss(self):
        """Exit price is exactly entry * (1 - stop_loss) when lower barrier fires."""
        df = _falling_ohlcv(n=30, start=100.0, daily_drop=0.02)
        signal_date = df.index[5]
        entry_price = df["open"].iloc[6]
        stop_loss = 0.015

        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", signal_date)],
            holding_period=5, profit_target=0.50, stop_loss=stop_loss,
        )

        row = result.iloc[0]
        assert row["exit_type"] == "stop_loss"
        assert row["exit_price"] == pytest.approx(entry_price * (1 - stop_loss))


# ---------------------------------------------------------------------------
# Direction (BUY vs SELL)
# ---------------------------------------------------------------------------

class TestDirection:
    def test_buy_profits_on_rising_price(self):
        """BUY: profit target fires when price rises."""
        df = _rising_ohlcv(n=30, daily_gain=0.02)
        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", df.index[5], Signal.BUY)],
            holding_period=5, profit_target=0.015, stop_loss=0.50,
        )
        assert result.iloc[0]["label"] == 1
        assert result.iloc[0]["forward_return"] > 0

    def test_sell_profits_on_falling_price(self):
        """SELL: profit target fires when price falls."""
        df = _falling_ohlcv(n=30, daily_drop=0.02)
        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", df.index[5], Signal.SELL)],
            holding_period=5, profit_target=0.015, stop_loss=0.50,
        )
        assert result.iloc[0]["label"] == 1
        assert result.iloc[0]["forward_return"] > 0

    def test_buy_stopped_on_falling_price(self):
        """BUY: stop loss fires when price falls."""
        df = _falling_ohlcv(n=30, daily_drop=0.02)
        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", df.index[5], Signal.BUY)],
            holding_period=5, profit_target=0.50, stop_loss=0.015,
        )
        assert result.iloc[0]["label"] == 0
        assert result.iloc[0]["exit_type"] == "stop_loss"

    def test_sell_stopped_on_rising_price(self):
        """SELL: stop loss fires when price rises."""
        df = _rising_ohlcv(n=30, daily_gain=0.02)
        result = create_metalabels(
            {"AAPL": df}, [_make_candidate("AAPL", df.index[5], Signal.SELL)],
            holding_period=5, profit_target=0.50, stop_loss=0.015,
        )
        assert result.iloc[0]["label"] == 0
        assert result.iloc[0]["exit_type"] == "stop_loss"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestLabelStatistics:
    def test_counts_match_labels(self):
        """get_label_statistics totals match the actual label column."""
        df = make_ohlcv(n=60)
        candidates = [_make_candidate("AAPL", df.index[i]) for i in range(5, 45)]
        result = create_metalabels({"AAPL": df}, candidates, holding_period=5)

        stats = get_label_statistics(result)
        assert stats["total"] == len(result)
        assert stats["profitable"] == result["label"].sum()

    def test_exit_type_breakdown_present(self):
        """Statistics include exit type counts when exit_type column exists."""
        df = make_ohlcv(n=60)
        candidates = [_make_candidate("AAPL", df.index[i]) for i in range(5, 45)]
        result = create_metalabels({"AAPL": df}, candidates)

        stats = get_label_statistics(result)
        assert "exit_profit_target_count" in stats
        assert "exit_stop_loss_count" in stats
        assert "exit_time_count" in stats
        total_exits = (
            stats["exit_profit_target_count"]
            + stats["exit_stop_loss_count"]
            + stats["exit_time_count"]
        )
        assert total_exits == stats["total"]
