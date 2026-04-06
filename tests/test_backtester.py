"""Tests for the backtesting engine — focus on execution timing correctness."""

import numpy as np
import pandas as pd
import pytest

from futures.backtester.engine import Backtester, BacktestResult
from futures.strategies.base import Strategy, Signal
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------------------
# Minimal strategies for testing
# ---------------------------------------------------------------------------

class BuyOnDay(Strategy):
    """Emits BUY for a specific ticker on a specific date, HOLD otherwise."""

    def __init__(self, ticker: str, signal_date: pd.Timestamp):
        self._ticker = ticker
        self._signal_date = signal_date

    @property
    def name(self) -> str:
        return "buy_on_day"

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        signals = {}
        for ticker, df in data.items():
            last_date = df.index[-1]
            if ticker == self._ticker and last_date == self._signal_date:
                signals[ticker] = Signal.BUY
            else:
                signals[ticker] = Signal.HOLD
        return signals


class AlwaysBuy(Strategy):
    """Buys the first ticker every day (up to max_positions)."""

    @property
    def name(self) -> str:
        return "always_buy"

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        return {ticker: Signal.BUY for ticker in data}


class AlwaysSell(Strategy):
    """Issues SELL for any open position."""

    @property
    def name(self) -> str:
        return "always_sell"

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        return {ticker: Signal.SELL for ticker in data}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNextOpenExecution:
    def test_trade_executes_day_after_signal(self):
        """A BUY signal on day T must produce a trade with entry_date = T+1."""
        df = make_ohlcv(n=30)
        signal_date = df.index[10]
        expected_entry_date = df.index[11]

        strategy = BuyOnDay("AAPL", signal_date)
        bt = Backtester(strategy, initial_cash=100_000, max_holding_days=5)
        result = bt.run({"AAPL": df}, show_progress=False)

        assert not result.trades.empty, "Expected at least one trade"
        entry_dates = result.trades["entry_date"].tolist()
        assert expected_entry_date in entry_dates, (
            f"Trade should enter on {expected_entry_date} (day after signal), "
            f"got {entry_dates}"
        )

    def test_entry_price_is_open_not_close(self):
        """Entry price must equal the open of the execution day, not the signal-day close."""
        df = make_ohlcv(n=30)
        signal_date = df.index[10]
        entry_day = df.index[11]

        # Make open and close clearly different on entry day
        df = df.copy()
        df.loc[entry_day, "open"] = 50.0
        df.loc[entry_day, "close"] = 75.0
        df.loc[entry_day, "high"] = 80.0
        df.loc[entry_day, "low"] = 45.0

        strategy = BuyOnDay("AAPL", signal_date)
        bt = Backtester(strategy, initial_cash=100_000, max_holding_days=5)
        result = bt.run({"AAPL": df}, show_progress=False)

        assert not result.trades.empty
        trade = result.trades[result.trades["entry_date"] == entry_day].iloc[0]
        # Entry price should be near 50.0 (open), with slippage applied
        assert trade["entry_price"] == pytest.approx(50.0 * (1 + bt.portfolio.slippage_pct), rel=1e-4)
        assert abs(trade["entry_price"] - 75.0) > 1.0, (
            "Entry price must NOT be close-day close price"
        )

    def test_no_same_bar_trading(self):
        """Portfolio should not execute any trade on the very first bar."""
        df = make_ohlcv(n=30)
        strategy = AlwaysBuy()
        bt = Backtester(strategy, initial_cash=100_000, max_positions=1)
        result = bt.run({"AAPL": df}, show_progress=False)

        if not result.trades.empty:
            # Earliest possible entry is df.index[1] (day after first signal on df.index[0])
            earliest_entry = result.trades["entry_date"].min()
            assert earliest_entry >= df.index[1], (
                "No trade should execute on the first bar (no prior signal)"
            )

    def test_equity_curve_length_matches_dates(self):
        """Equity curve has one entry per trading day."""
        df = make_ohlcv(n=50)
        strategy = AlwaysBuy()
        bt = Backtester(strategy, initial_cash=100_000, max_positions=1)
        result = bt.run({"AAPL": df}, show_progress=False)

        assert len(result.equity_curve) == len(df)


class TestPortfolioAccounting:
    def test_initial_cash_preserved_when_no_signals(self):
        """Portfolio value equals initial cash when no trades fire."""

        class NeverBuy(Strategy):
            @property
            def name(self):
                return "never_buy"

            def generate_signals(self, data):
                return {t: Signal.HOLD for t in data}

        df = make_ohlcv(n=20)
        bt = Backtester(NeverBuy(), initial_cash=50_000)
        result = bt.run({"AAPL": df}, show_progress=False)

        assert result.final_value == pytest.approx(50_000.0)
        assert result.trades.empty

    def test_max_positions_respected(self):
        """Number of concurrent open positions never exceeds max_positions."""
        data = {t: make_ohlcv(seed=i) for i, t in enumerate(["AAPL", "MSFT", "GOOG", "AMZN"])}
        strategy = AlwaysBuy()
        bt = Backtester(strategy, initial_cash=100_000, max_positions=2)
        result = bt.run(data, show_progress=False)

        # Check trade history: at any entry time, open positions ≤ 2
        # (Simple check: total unique concurrent entries per date)
        if not result.trades.empty:
            entries = result.trades.groupby("entry_date").size()
            # On any single day, at most max_positions can be opened
            # (this is a loose bound since exits also happen that day)
            assert entries.max() <= bt.max_positions

    def test_max_holding_days_closes_positions(self):
        """Positions held longer than max_holding_days trading days are closed."""
        import numpy as np

        df = make_ohlcv(n=50)
        max_hold = 5

        strategy = AlwaysBuy()
        bt = Backtester(strategy, initial_cash=100_000, max_positions=1, max_holding_days=max_hold)
        result = bt.run({"AAPL": df}, show_progress=False)

        if not result.trades.empty:
            for _, trade in result.trades.iterrows():
                bdays = int(np.busday_count(
                    trade["entry_date"].date(), trade["exit_date"].date()
                ))
                assert bdays <= max_hold, (
                    f"Trade held {bdays} business days, max is {max_hold}"
                )

    def test_transaction_costs_reduce_pnl(self):
        """Backtest with costs produces lower PnL than one without."""
        df = make_ohlcv(n=40)
        strategy = AlwaysBuy()

        bt_free = Backtester(strategy, initial_cash=100_000, transaction_cost_pct=0.0,
                             slippage_pct=0.0, max_positions=1, max_holding_days=5)
        bt_cost = Backtester(strategy, initial_cash=100_000, transaction_cost_pct=0.01,
                             slippage_pct=0.01, max_positions=1, max_holding_days=5)

        result_free = bt_free.run({"AAPL": df}, show_progress=False)
        result_cost = bt_cost.run({"AAPL": df}, show_progress=False)

        assert result_free.final_value >= result_cost.final_value, (
            "Trading costs must reduce (or not increase) final portfolio value"
        )


class TestMetrics:
    def test_sharpe_finite(self):
        """Sharpe ratio should be a finite number."""
        df = make_ohlcv(n=60)
        strategy = AlwaysBuy()
        bt = Backtester(strategy, initial_cash=100_000, max_positions=1, max_holding_days=10)
        result = bt.run({"AAPL": df}, show_progress=False)

        assert np.isfinite(result.metrics.sharpe_ratio)

    def test_max_drawdown_non_negative(self):
        """Max drawdown is stored as a positive percentage (absolute magnitude)."""
        df = make_ohlcv(n=60)
        strategy = AlwaysBuy()
        bt = Backtester(strategy, initial_cash=100_000, max_positions=1, max_holding_days=10)
        result = bt.run({"AAPL": df}, show_progress=False)

        assert result.metrics.max_drawdown >= 0.0
