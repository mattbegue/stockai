"""Base strategy class and signal types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Signal(Enum):
    """Trading signals."""

    BUY = 1
    SELL = -1
    HOLD = 0

    def __str__(self) -> str:
        return self.name


@dataclass
class Position:
    """Represents a trading position."""

    ticker: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost of the position."""
        return self.shares * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    A strategy takes price data and generates trading signals.
    Strategies should be stateless - all state is managed by the backtester.
    """

    def __init__(self, **params):
        """Initialize strategy with parameters."""
        self.params = params

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this strategy configuration."""
        pass

    @property
    def required_history(self) -> int:
        """
        Minimum number of bars required before generating signals.

        Override this if your strategy needs warmup data.
        Default is 0 (no warmup needed).
        """
        return 0

    @abstractmethod
    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """
        Generate trading signals for each ticker.

        Args:
            data: Dict mapping ticker to DataFrame with OHLCV data
                  Each DataFrame has columns: open, high, low, close, volume
                  and a DatetimeIndex

        Returns:
            Dict mapping ticker to Signal (BUY, SELL, or HOLD)
        """
        pass

    def precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Precompute any indicators needed by the strategy.

        Override this to add indicator columns to the dataframe.
        Called once before backtesting starts.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional indicator columns
        """
        return df

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


class FilterStrategy(Strategy):
    """
    Strategy that acts as a filter/screen.

    Use this to implement first-pass filters that identify
    candidates for more sophisticated analysis.
    """

    @abstractmethod
    def screen(self, data: dict[str, pd.DataFrame]) -> list[str]:
        """
        Screen tickers and return those passing the filter.

        Args:
            data: Dict mapping ticker to DataFrame

        Returns:
            List of tickers that pass the screen
        """
        pass

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """Generate HOLD for all - filtering only, no trading signals."""
        return {ticker: Signal.HOLD for ticker in data.keys()}


class CompositeStrategy(Strategy):
    """
    Combine multiple strategies.

    Useful for layering filters with ML strategies.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        mode: str = "unanimous",  # 'unanimous', 'majority', 'any'
    ):
        """
        Initialize composite strategy.

        Args:
            strategies: List of strategies to combine
            mode: How to combine signals
                - 'unanimous': All must agree (AND)
                - 'majority': Majority vote
                - 'any': Any signal wins (OR)
        """
        super().__init__(strategies=strategies, mode=mode)
        self.strategies = strategies
        self.mode = mode

    @property
    def name(self) -> str:
        names = [s.name for s in self.strategies]
        return f"composite_{self.mode}_{'_'.join(names)}"

    @property
    def required_history(self) -> int:
        return max(s.required_history for s in self.strategies)

    def precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Precompute indicators for all sub-strategies."""
        result = df.copy()
        for strategy in self.strategies:
            result = strategy.precompute_indicators(result)
        return result

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """Combine signals from all strategies."""
        all_signals = [s.generate_signals(data) for s in self.strategies]
        tickers = list(data.keys())

        combined = {}
        for ticker in tickers:
            ticker_signals = [
                signals.get(ticker, Signal.HOLD) for signals in all_signals
            ]
            combined[ticker] = self._combine_signals(ticker_signals)

        return combined

    def _combine_signals(self, signals: list[Signal]) -> Signal:
        """Combine multiple signals into one."""
        values = [s.value for s in signals]

        if self.mode == "unanimous":
            if all(v == Signal.BUY.value for v in values):
                return Signal.BUY
            elif all(v == Signal.SELL.value for v in values):
                return Signal.SELL
            return Signal.HOLD

        elif self.mode == "majority":
            avg = sum(values) / len(values)
            if avg > 0.5:
                return Signal.BUY
            elif avg < -0.5:
                return Signal.SELL
            return Signal.HOLD

        elif self.mode == "any":
            if Signal.BUY.value in values:
                return Signal.BUY
            elif Signal.SELL.value in values:
                return Signal.SELL
            return Signal.HOLD

        return Signal.HOLD
