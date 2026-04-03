"""Performance metrics calculation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MetricsSummary:
    """Summary of backtest performance metrics."""

    # Returns
    total_return: float
    annualized_return: float
    volatility: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int  # days

    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_trade_pnl: float
    avg_winning_trade: float
    avg_losing_trade: float
    profit_factor: float
    expectancy: float

    # Other
    trading_days: int
    best_day: float
    worst_day: float


def calculate_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    initial_cash: float,
    risk_free_rate: float = 0.04,  # 4% annual
    trading_days_per_year: int = 252,
) -> MetricsSummary:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity_curve: Series of portfolio values indexed by date
        trades: DataFrame of completed trades
        initial_cash: Starting capital
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        trading_days_per_year: Trading days per year

    Returns:
        MetricsSummary with all metrics
    """
    daily_returns = equity_curve.pct_change().dropna()

    # Basic return metrics
    total_return = (equity_curve.iloc[-1] - initial_cash) / initial_cash * 100

    trading_days = len(daily_returns)
    years = trading_days / trading_days_per_year

    if years > 0:
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
    else:
        annualized_return = total_return

    volatility = daily_returns.std() * np.sqrt(trading_days_per_year) * 100

    # Sharpe Ratio
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
    excess_returns = daily_returns - daily_rf

    if daily_returns.std() > 0:
        sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(trading_days_per_year)
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio (only downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino_ratio = (excess_returns.mean() / downside_returns.std()) * np.sqrt(trading_days_per_year)
    else:
        sortino_ratio = sharpe_ratio

    # Drawdown analysis
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100

    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Max drawdown duration
    in_drawdown = drawdown < 0
    drawdown_periods = (~in_drawdown).cumsum()
    if in_drawdown.any():
        drawdown_lengths = in_drawdown.groupby(drawdown_periods).sum()
        max_drawdown_duration = int(drawdown_lengths.max())
    else:
        max_drawdown_duration = 0

    # Calmar Ratio
    if max_drawdown > 0:
        calmar_ratio = annualized_return / max_drawdown
    else:
        calmar_ratio = 0.0

    # Trade statistics
    total_trades = len(trades) if not trades.empty else 0

    if total_trades > 0:
        winning_trades = len(trades[trades["pnl"] > 0])
        losing_trades = len(trades[trades["pnl"] < 0])
        win_rate = (winning_trades / total_trades) * 100

        avg_trade_pnl = trades["pnl"].mean()

        winners = trades[trades["pnl"] > 0]["pnl"]
        losers = trades[trades["pnl"] < 0]["pnl"]

        avg_winning_trade = winners.mean() if len(winners) > 0 else 0.0
        avg_losing_trade = losers.mean() if len(losers) > 0 else 0.0

        # Profit factor
        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 0

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0

        # Expectancy
        expectancy = (win_rate / 100 * avg_winning_trade) + ((1 - win_rate / 100) * avg_losing_trade)
    else:
        winning_trades = 0
        losing_trades = 0
        win_rate = 0.0
        avg_trade_pnl = 0.0
        avg_winning_trade = 0.0
        avg_losing_trade = 0.0
        profit_factor = 0.0
        expectancy = 0.0

    # Best/worst days
    best_day = daily_returns.max() * 100 if len(daily_returns) > 0 else 0.0
    worst_day = daily_returns.min() * 100 if len(daily_returns) > 0 else 0.0

    return MetricsSummary(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        avg_trade_pnl=avg_trade_pnl,
        avg_winning_trade=avg_winning_trade,
        avg_losing_trade=avg_losing_trade,
        profit_factor=profit_factor,
        expectancy=expectancy,
        trading_days=trading_days,
        best_day=best_day,
        worst_day=worst_day,
    )


def compare_strategies(results: list["BacktestResult"]) -> pd.DataFrame:
    """
    Compare multiple backtest results.

    Args:
        results: List of BacktestResult objects

    Returns:
        DataFrame comparing key metrics across strategies
    """
    records = []

    for result in results:
        records.append({
            "Strategy": result.strategy_name,
            "Total Return (%)": result.total_return,
            "Sharpe Ratio": result.metrics.sharpe_ratio,
            "Sortino Ratio": result.metrics.sortino_ratio,
            "Max Drawdown (%)": result.metrics.max_drawdown,
            "Win Rate (%)": result.metrics.win_rate,
            "Profit Factor": result.metrics.profit_factor,
            "Total Trades": result.metrics.total_trades,
            "Volatility (%)": result.metrics.volatility,
        })

    df = pd.DataFrame(records)
    df = df.set_index("Strategy")

    return df
