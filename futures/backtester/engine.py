"""Main backtesting engine."""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd
from tqdm import tqdm

from futures.config import get_settings
from futures.strategies.base import Strategy, Signal
from .portfolio import Portfolio
from .metrics import calculate_metrics, MetricsSummary


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    strategy_name: str
    start_date: date
    end_date: date
    initial_cash: float
    final_value: float
    equity_curve: pd.Series
    daily_returns: pd.Series
    trades: pd.DataFrame
    metrics: MetricsSummary
    signals_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def total_return(self) -> float:
        """Total return percentage."""
        return ((self.final_value - self.initial_cash) / self.initial_cash) * 100

    def summary(self) -> str:
        """Generate a text summary of results."""
        lines = [
            f"Strategy: {self.strategy_name}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Initial Capital: ${self.initial_cash:,.2f}",
            f"Final Value: ${self.final_value:,.2f}",
            f"Total Return: {self.total_return:.2f}%",
            f"",
            f"Performance Metrics:",
            f"  Sharpe Ratio: {self.metrics.sharpe_ratio:.3f}",
            f"  Sortino Ratio: {self.metrics.sortino_ratio:.3f}",
            f"  Max Drawdown: {self.metrics.max_drawdown:.2f}%",
            f"  Win Rate: {self.metrics.win_rate:.1f}%",
            f"  Profit Factor: {self.metrics.profit_factor:.2f}",
            f"  Total Trades: {self.metrics.total_trades}",
            f"  Avg Trade P&L: ${self.metrics.avg_trade_pnl:.2f}",
        ]
        return "\n".join(lines)


class Backtester:
    """
    Main backtesting engine.

    Simulates trading a strategy over historical data.
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_cash: Optional[float] = None,
        transaction_cost_pct: Optional[float] = None,
        slippage_pct: Optional[float] = None,
        position_size: float = 0.1,  # 10% of portfolio per position
        max_positions: int = 10,
        max_holding_days: Optional[int] = None,
        profit_target: Optional[float] = None,
        stop_loss: Optional[float] = None,
        sector_map: Optional[dict[str, str]] = None,
        max_sector_positions: int = 3,
        correlation_limit: Optional[float] = None,
        correlation_lookback: int = 30,
    ):
        """
        Initialize backtester.

        Args:
            strategy: Trading strategy to test
            initial_cash: Starting capital
            transaction_cost_pct: Transaction cost as percentage
            slippage_pct: Slippage as percentage
            position_size: Size of each position as fraction of portfolio
            max_positions: Maximum number of concurrent positions
            max_holding_days: Maximum days to hold a position (None = no limit)
            profit_target: Exit long when price rises this fraction above entry
                           (mirrors the triple barrier upper barrier). None = disabled.
            stop_loss: Exit long when price falls this fraction below entry
                       (mirrors the triple barrier lower barrier). None = disabled.
            sector_map: Dict mapping ticker → sector name. When provided, limits
                        concurrent positions to max_sector_positions per sector.
            max_sector_positions: Max concurrent positions in a single sector (default 3).
            correlation_limit: Reject new positions that have rolling correlation
                               above this threshold with any existing position.
                               None = disabled. Typical value: 0.7.
            correlation_lookback: Number of trading days for correlation window (default 30).
        """
        settings = get_settings()

        self.strategy = strategy
        self.position_size = position_size
        self.max_positions = max_positions
        self.max_holding_days = max_holding_days
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.sector_map = sector_map or {}
        self.max_sector_positions = max_sector_positions
        self.correlation_limit = correlation_limit
        self.correlation_lookback = correlation_lookback

        self.portfolio = Portfolio(
            initial_cash=initial_cash or settings.default_cash,
            transaction_cost_pct=transaction_cost_pct or settings.transaction_cost_pct,
            slippage_pct=slippage_pct or settings.slippage_pct,
        )

        # Populated in run() for correlation filtering; keyed by ticker, indexed by date
        self._returns_df: pd.DataFrame = pd.DataFrame()

    def run(
        self,
        data: dict[str, pd.DataFrame],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        show_progress: bool = True,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: Dict mapping ticker to DataFrame with OHLCV data
            start_date: Start date for backtest (uses data start if not specified)
            end_date: End date for backtest (uses data end if not specified)
            show_progress: Show progress bar

        Returns:
            BacktestResult with all metrics and history
        """
        self.portfolio.reset()

        # Precompute indicators
        processed_data = {}
        for ticker, df in data.items():
            processed_data[ticker] = self.strategy.precompute_indicators(df)

        # Precompute daily returns for correlation filtering (P2-6)
        if self.correlation_limit is not None:
            close_prices = {
                ticker: df["close"]
                for ticker, df in processed_data.items()
                if "close" in df.columns
            }
            self._returns_df = pd.DataFrame(close_prices).pct_change()
        else:
            self._returns_df = pd.DataFrame()

        # Get all unique dates
        all_dates = set()
        for df in processed_data.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)

        # Apply date filters
        if start_date:
            all_dates = [d for d in all_dates if d.date() >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d.date() <= end_date]

        if len(all_dates) < self.strategy.required_history:
            raise ValueError(
                f"Not enough data. Need {self.strategy.required_history} bars, "
                f"got {len(all_dates)}"
            )

        # Track equity curve
        equity_curve = []
        signals_history = []

        # Signals generated at day T's close are executed at day T+1's open.
        pending_signals: dict[str, Signal] = {}
        pending_sizes: dict[str, float] = {}  # Per-ticker position sizes (fractions)
        pending_holding_days: dict[str, int] = {}  # Per-ticker max holding periods (P2-10)

        iterator = tqdm(all_dates, desc="Backtesting") if show_progress else all_dates

        for current_date in iterator:
            # Collect open and close prices; build data slice up to current_date.
            current_data = {}
            open_prices: dict[str, float] = {}
            close_prices: dict[str, float] = {}
            high_prices: dict[str, float] = {}
            low_prices: dict[str, float] = {}

            for ticker, df in processed_data.items():
                mask = df.index <= current_date
                if mask.sum() >= self.strategy.required_history:
                    current_data[ticker] = df[mask]
                if current_date in df.index:
                    open_prices[ticker] = df.loc[current_date, "open"]
                    close_prices[ticker] = df.loc[current_date, "close"]
                    high_prices[ticker] = df.loc[current_date, "high"]
                    low_prices[ticker] = df.loc[current_date, "low"]

            # --- AT OPEN: execute yesterday's signals and all exit checks ---
            if open_prices:
                self.portfolio.update_prices(open_prices)
                # Time-based exits: always call so per-position limits fire even
                # when global max_holding_days is None (P2-10)
                self._close_aged_positions(open_prices, current_date)
                # Barrier exits (profit target / stop loss) — consistent with triple barrier labels
                if self.profit_target is not None or self.stop_loss is not None:
                    self._close_barrier_positions(
                        open_prices, high_prices, low_prices, current_date
                    )
                if pending_signals:
                    self._execute_signals(
                        pending_signals, open_prices, current_date,
                        pending_sizes, pending_holding_days,
                    )

            # --- AT CLOSE: mark-to-market for equity curve ---
            if close_prices:
                self.portfolio.update_prices(close_prices)

            # Generate signals based on today's full bar; they execute tomorrow.
            if current_data:
                signals = self.strategy.generate_signals(current_data)
                # Per-ticker sizes (empty dict → use global self.position_size)
                pending_sizes = self.strategy.get_position_sizes(signals)
                # Per-ticker holding periods (empty dict → use global max_holding_days)
                pending_holding_days = self.strategy.get_position_holding_days(signals)
                pending_signals = signals

                for ticker, signal in signals.items():
                    if signal != Signal.HOLD:
                        signals_history.append({
                            "date": current_date,
                            "ticker": ticker,
                            "signal": signal.name,
                        })
            else:
                pending_signals = {}
                pending_sizes = {}
                pending_holding_days = {}

            # Record equity at close
            equity_curve.append({
                "date": current_date,
                "equity": self.portfolio.total_value,
                "cash": self.portfolio.cash,
                "positions": self.portfolio.position_value,
            })

        # Close all positions at end
        final_prices = {
            ticker: df["close"].iloc[-1]
            for ticker, df in processed_data.items()
            if not df.empty
        }
        self.portfolio.close_all(final_prices, date=all_dates[-1])

        # Build result
        equity_df = pd.DataFrame(equity_curve).set_index("date")
        equity_series = equity_df["equity"]

        daily_returns = equity_series.pct_change().dropna()

        trades_df = self.portfolio.get_trade_history()
        signals_df = pd.DataFrame(signals_history) if signals_history else pd.DataFrame()

        metrics = calculate_metrics(
            equity_curve=equity_series,
            trades=trades_df,
            initial_cash=self.portfolio.initial_cash,
        )

        return BacktestResult(
            strategy_name=self.strategy.name,
            start_date=all_dates[0].date() if hasattr(all_dates[0], 'date') else all_dates[0],
            end_date=all_dates[-1].date() if hasattr(all_dates[-1], 'date') else all_dates[-1],
            initial_cash=self.portfolio.initial_cash,
            final_value=self.portfolio.total_value,
            equity_curve=equity_series,
            daily_returns=daily_returns,
            trades=trades_df,
            metrics=metrics,
            signals_history=signals_df,
        )

    def _execute_signals(
        self,
        signals: dict[str, Signal],
        prices: dict[str, float],
        current_date: pd.Timestamp,
        position_sizes: Optional[dict[str, float]] = None,
        holding_days: Optional[dict[str, int]] = None,
    ):
        """
        Execute trading signals.

        Args:
            signals: Ticker → Signal mapping
            prices: Ticker → execution price (next-bar open)
            current_date: Execution date
            position_sizes: Optional per-ticker position sizes as portfolio fractions.
                            Falls back to self.position_size when not provided.
            holding_days: Optional per-ticker max holding periods (P2-10).
                          Falls back to self.max_holding_days when not provided.
        """
        position_sizes = position_sizes or {}
        holding_days = holding_days or {}

        # First, handle sell signals
        for ticker, signal in signals.items():
            if signal == Signal.SELL and ticker in self.portfolio.positions:
                if ticker in prices:
                    self.portfolio.sell(ticker, prices[ticker], date=current_date)

        # Then, handle buy signals (sorted by confidence-weighted size, highest first)
        buy_candidates = [
            (ticker, prices[ticker], position_sizes.get(ticker, self.position_size))
            for ticker, signal in signals.items()
            if signal == Signal.BUY
            and ticker not in self.portfolio.positions
            and ticker in prices
        ]

        # Sort by position size descending (highest-conviction trades first)
        buy_candidates.sort(key=lambda x: x[2], reverse=True)

        # Limit new positions
        available_slots = self.max_positions - len(self.portfolio.positions)
        buy_candidates = buy_candidates[:available_slots]

        # --- Sector cap filter (P2-6) ---
        if self.sector_map:
            sector_counts: dict[str, int] = {}
            for held_ticker in self.portfolio.positions:
                sector = self.sector_map.get(held_ticker, "Unknown")
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            filtered = []
            for ticker, price, size_fraction in buy_candidates:
                sector = self.sector_map.get(ticker, "Unknown")
                if sector_counts.get(sector, 0) < self.max_sector_positions:
                    filtered.append((ticker, price, size_fraction))
                    # Speculatively increment so same-day candidates don't pile into one sector
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
            buy_candidates = filtered

        # --- Correlation filter (P2-6) ---
        if self.correlation_limit is not None and not self._returns_df.empty and self.portfolio.positions:
            held_tickers = list(self.portfolio.positions.keys())
            # Slice to the lookback window ending at current_date
            mask = self._returns_df.index <= current_date
            window = self._returns_df[mask].tail(self.correlation_lookback)

            filtered = []
            for ticker, price, size_fraction in buy_candidates:
                if ticker not in window.columns:
                    filtered.append((ticker, price, size_fraction))
                    continue
                candidate_returns = window[ticker].dropna()
                too_correlated = False
                for held in held_tickers:
                    if held not in window.columns:
                        continue
                    held_returns = window[held].dropna()
                    common = candidate_returns.index.intersection(held_returns.index)
                    if len(common) < 10:
                        continue
                    corr = candidate_returns[common].corr(held_returns[common])
                    if corr > self.correlation_limit:
                        too_correlated = True
                        break
                if not too_correlated:
                    filtered.append((ticker, price, size_fraction))
            buy_candidates = filtered

        for ticker, price, size_fraction in buy_candidates:
            position_value = self.portfolio.total_value * size_fraction
            if self.portfolio.cash < position_value * 0.5:
                break
            # Per-position holding period (P2-10): strategy override → global fallback
            pos_hold_days = holding_days.get(ticker, self.max_holding_days)
            self.portfolio.buy(
                ticker,
                price,
                value=min(position_value, self.portfolio.cash * 0.95),
                date=current_date,
                max_holding_days=pos_hold_days,
            )

    def _close_aged_positions(
        self,
        prices: dict[str, float],
        current_date: pd.Timestamp,
    ):
        """Close positions that have exceeded their max_holding_days (in trading days).

        Each position can carry its own max_holding_days (set at entry, P2-10).
        If a position has no per-position limit, the backtester's global
        max_holding_days is used as fallback. If neither is set, no time exit.
        """
        import numpy as np

        positions_to_close = []
        for ticker, position in self.portfolio.positions.items():
            # Effective limit: per-position first, then global fallback
            effective_max = position.max_holding_days
            if effective_max is None:
                effective_max = self.max_holding_days
            if effective_max is None:
                continue  # No limit for this position

            # Count business days held so results are independent of weekend placement
            days_held = int(np.busday_count(
                position.entry_date.date(), current_date.date()
            ))
            if days_held >= effective_max:
                positions_to_close.append(ticker)

        for ticker in positions_to_close:
            if ticker in prices:
                self.portfolio.sell(ticker, prices[ticker], date=current_date)


    def _close_barrier_positions(
        self,
        open_prices: dict[str, float],
        high_prices: dict[str, float],
        low_prices: dict[str, float],
        current_date: pd.Timestamp,
    ):
        """Close positions that hit a profit target or stop loss.

        Barrier priority matches labels.py (pessimistic / conservative):
          1. Stop loss takes priority when both barriers breach the same bar.
          2. Gap-through open: exit at open (no better fill available).
          3. Intraday breach: exit at exact barrier price.

        Only long positions are supported (strategy is long-only).
        """
        positions_to_close: list[tuple[str, float]] = []

        for ticker, position in self.portfolio.positions.items():
            entry = position.entry_price
            target = entry * (1.0 + self.profit_target) if self.profit_target is not None else float("inf")
            stop = entry * (1.0 - self.stop_loss) if self.stop_loss is not None else 0.0

            current_open = open_prices.get(ticker)
            if current_open is None:
                continue

            # Gap scenarios: open already beyond a barrier
            if current_open <= stop:
                positions_to_close.append((ticker, current_open))
                continue
            if current_open >= target:
                positions_to_close.append((ticker, current_open))
                continue

            # Intraday check — stop loss first (pessimistic)
            bar_low = low_prices.get(ticker)
            bar_high = high_prices.get(ticker)

            if bar_low is not None and bar_low <= stop:
                positions_to_close.append((ticker, stop))
                continue
            if bar_high is not None and bar_high >= target:
                positions_to_close.append((ticker, target))

        for ticker, exit_price in positions_to_close:
            self.portfolio.sell(ticker, exit_price, date=current_date)


def run_backtest(
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
    **kwargs,
) -> BacktestResult:
    """Convenience function to run a backtest."""
    backtester = Backtester(strategy, **kwargs)
    return backtester.run(data)
