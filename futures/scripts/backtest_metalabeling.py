"""Backtest the metalabeling strategy and compare to buy-and-hold."""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from futures.config import get_universe, TickerUniverse, get_sector_map
from futures.data.fetcher import DataManager
from futures.data.earnings import EarningsCalendar
from futures.strategies import MetalabelingStrategy
from futures.metalabeling import PrimarySignalGenerator, MetaFeatureEngineering
from futures.backtester import Backtester
from futures.regime.classifier import MarketRegimeClassifier


# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')


def load_model():
    """Load the active trained model from registry."""
    from futures.models.registry import ModelRegistry

    registry = ModelRegistry()
    return registry.load_active()


def create_run_directory() -> Path:
    """Create timestamped run directory."""
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / timestamp
    run_dir.mkdir(exist_ok=True)

    return run_dir


def save_config(run_dir: Path, config: dict):
    """Save configuration for reproducibility."""
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"  Config saved to: {config_path}")


def save_results(run_dir: Path, results: dict):
    """Save results summary."""
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {results_path}")


def calculate_metrics(equity_curve: pd.Series) -> dict:
    """Calculate performance metrics from equity curve."""
    if len(equity_curve) < 2:
        return {}

    returns = equity_curve.pct_change().dropna()
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    volatility = returns.std() * np.sqrt(252)
    sharpe = annualized_return / volatility if volatility > 0 else 0

    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    return {
        "total_return_pct": total_return * 100,
        "annualized_return_pct": annualized_return * 100,
        "volatility_pct": volatility * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 100,
        "win_rate_pct": win_rate * 100,
        "trading_days": len(equity_curve),
    }


def plot_equity_curves(run_dir: Path, strategy_equity: pd.Series, spy_equity: pd.Series,
                       strategy_name: str = "Metalabeling"):
    """Plot equity curves comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize to start at 100
    strategy_norm = (strategy_equity / strategy_equity.iloc[0]) * 100
    spy_norm = (spy_equity / spy_equity.iloc[0]) * 100

    ax.plot(strategy_norm.index, strategy_norm.values, label=strategy_name, linewidth=2, color='#2E86AB')
    ax.plot(spy_norm.index, spy_norm.values, label='SPY Buy & Hold', linewidth=2, color='#A23B72', alpha=0.8)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value (Indexed to 100)', fontsize=12)
    ax.set_title('Strategy vs SPY Buy & Hold', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    # Add final values annotation
    strategy_final = strategy_norm.iloc[-1]
    spy_final = spy_norm.iloc[-1]
    ax.annotate(f'{strategy_final:.1f}', xy=(strategy_norm.index[-1], strategy_final),
                xytext=(10, 0), textcoords='offset points', fontsize=10, color='#2E86AB')
    ax.annotate(f'{spy_final:.1f}', xy=(spy_norm.index[-1], spy_final),
                xytext=(10, 0), textcoords='offset points', fontsize=10, color='#A23B72')

    plt.tight_layout()
    plt.savefig(run_dir / "equity_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: equity_curves.png")


def plot_drawdown(run_dir: Path, strategy_equity: pd.Series, spy_equity: pd.Series):
    """Plot drawdown comparison."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Calculate drawdowns
    strategy_rolling_max = strategy_equity.expanding().max()
    strategy_dd = ((strategy_equity - strategy_rolling_max) / strategy_rolling_max) * 100

    spy_rolling_max = spy_equity.expanding().max()
    spy_dd = ((spy_equity - spy_rolling_max) / spy_rolling_max) * 100

    ax.fill_between(strategy_dd.index, strategy_dd.values, 0, alpha=0.4, label='Metalabeling', color='#2E86AB')
    ax.fill_between(spy_dd.index, spy_dd.values, 0, alpha=0.4, label='SPY', color='#A23B72')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    # Add max drawdown annotations
    strategy_max_dd = strategy_dd.min()
    spy_max_dd = spy_dd.min()
    ax.axhline(y=strategy_max_dd, color='#2E86AB', linestyle='--', alpha=0.5)
    ax.axhline(y=spy_max_dd, color='#A23B72', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(run_dir / "drawdown.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: drawdown.png")


def plot_monthly_returns(run_dir: Path, equity_curve: pd.Series):
    """Plot monthly returns heatmap."""
    # Calculate daily returns and resample to monthly
    returns = equity_curve.pct_change().dropna()
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100

    # Create pivot table for heatmap
    monthly_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values
    })

    pivot = monthly_df.pivot(index='year', columns='month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(12, 4))

    # Create heatmap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=-10, vmax=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Return (%)', fontsize=11)

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > 5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        color=text_color, fontsize=9)

    ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)

    plt.tight_layout()
    plt.savefig(run_dir / "monthly_returns.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: monthly_returns.png")


def plot_rolling_sharpe(run_dir: Path, equity_curve: pd.Series, window: int = 63):
    """Plot rolling Sharpe ratio."""
    returns = equity_curve.pct_change().dropna()

    # Adjust window if not enough data (need at least window + 10 points for meaningful plot)
    min_points_needed = window + 10
    if len(returns) < min_points_needed:
        if len(returns) < 20:
            print(f"  Skipped: rolling_sharpe.png (only {len(returns)} trading days, need at least 20)")
            return
        # Use smaller window: half the data length, minimum 10 days
        window = max(10, len(returns) // 2)
        print(f"  Note: Using {window}-day window (limited data)")

    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std

    # Check if we have valid data to plot
    valid_sharpe = rolling_sharpe.dropna()
    if len(valid_sharpe) < 5:
        print(f"  Skipped: rolling_sharpe.png (insufficient data for rolling calculation)")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5, color='#2E86AB')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
    ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Sharpe = -1')

    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe.values > 0, alpha=0.3, color='green')
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe.values < 0, alpha=0.3, color='red')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
    ax.set_title(f'Rolling {window}-Day Sharpe Ratio', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(run_dir / "rolling_sharpe.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: rolling_sharpe.png")


def plot_trade_analysis(run_dir: Path, trades_df: pd.DataFrame):
    """Plot trade analysis charts."""
    if trades_df.empty or 'pnl' not in trades_df.columns:
        print("  Skipped: trade_analysis.png (no trades)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. P&L Distribution
    ax1 = axes[0, 0]
    pnl = trades_df['pnl'].dropna()
    if len(pnl) > 0:
        colors = ['green' if x > 0 else 'red' for x in pnl]
        ax1.hist(pnl, bins=30, color='#2E86AB', edgecolor='white', alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax1.axvline(x=pnl.mean(), color='orange', linestyle='--', label=f'Mean: ${pnl.mean():.2f}')
        ax1.set_xlabel('P&L ($)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
        ax1.legend()

    # 2. Cumulative P&L
    ax2 = axes[0, 1]
    if len(pnl) > 0:
        cumulative_pnl = pnl.cumsum()
        ax2.plot(range(len(cumulative_pnl)), cumulative_pnl.values, linewidth=2, color='#2E86AB')
        ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl.values, 0,
                         where=cumulative_pnl.values > 0, alpha=0.3, color='green')
        ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl.values, 0,
                         where=cumulative_pnl.values < 0, alpha=0.3, color='red')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Trade Number', fontsize=11)
        ax2.set_ylabel('Cumulative P&L ($)', fontsize=11)
        ax2.set_title('Cumulative P&L by Trade', fontsize=12, fontweight='bold')

    # 3. Win/Loss by Ticker (top 10)
    ax3 = axes[1, 0]
    if 'ticker' in trades_df.columns and len(pnl) > 0:
        ticker_pnl = trades_df.groupby('ticker')['pnl'].sum().sort_values()
        top_bottom = pd.concat([ticker_pnl.head(5), ticker_pnl.tail(5)])
        colors = ['red' if x < 0 else 'green' for x in top_bottom.values]
        ax3.barh(top_bottom.index, top_bottom.values, color=colors, alpha=0.7)
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('Total P&L ($)', fontsize=11)
        ax3.set_ylabel('Ticker', fontsize=11)
        ax3.set_title('P&L by Ticker (Top/Bottom 5)', fontsize=12, fontweight='bold')

    # 4. Trade Duration (if available)
    ax4 = axes[1, 1]
    wins = (pnl > 0).sum()
    losses = (pnl <= 0).sum()
    ax4.pie([wins, losses], labels=['Winners', 'Losers'],
            colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%',
            startangle=90, explode=(0.02, 0.02))
    ax4.set_title(f'Win Rate: {wins}/{wins+losses} trades', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(run_dir / "trade_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: trade_analysis.png")


def analyze_by_confidence(
    run_dir: Path,
    trades_df: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    model,
    feature_eng,
    universe,
    model_info: dict,
):
    """Analyze trade performance by model confidence level."""
    if trades_df.empty or 'entry_date' not in trades_df.columns:
        print("  Skipped: confidence_analysis.png (no trades)")
        return None

    print("  Computing confidence scores for trades...")

    # Re-compute confidence for each trade
    confidences = []
    context_data = {t: data[t] for t in universe.context if t in data}

    for _, trade in trades_df.iterrows():
        ticker = trade['ticker']
        entry_date = pd.Timestamp(trade['entry_date'])

        if ticker not in data:
            confidences.append(np.nan)
            continue

        try:
            df = data[ticker]
            df_with_ind = feature_eng.compute_indicators_for_df(df)

            if entry_date not in df_with_ind.index:
                # Find closest date
                idx = df_with_ind.index.get_indexer([entry_date], method='nearest')[0]
                entry_date = df_with_ind.index[idx]

            # Determine direction from trade side
            direction = 1 if trade.get('side', 'long') == 'long' else -1

            features = feature_eng.create_features_for_signal(
                df_with_ind, context_data, ticker, entry_date, direction, []
            )

            if features.empty:
                confidences.append(np.nan)
                continue

            feature_df = pd.DataFrame([features]).fillna(0)

            # Align to training features
            feature_names = model_info.get('feature_names', [])
            if feature_names:
                aligned_df = pd.DataFrame(index=feature_df.index)
                for feat in feature_names:
                    aligned_df[feat] = feature_df.get(feat, 0)
                feature_df = aligned_df

            from futures.ml.features import FeatureSet
            feature_input = FeatureSet(X=feature_df, feature_names=list(feature_df.columns))
            proba = model.predict_proba(feature_input)
            confidence = proba[0, 1] if proba.shape[1] == 2 else proba[0, 0]
            confidences.append(confidence)

        except Exception as e:
            confidences.append(np.nan)

    trades_df = trades_df.copy()
    trades_df['confidence'] = confidences

    # Remove trades without confidence
    trades_with_conf = trades_df.dropna(subset=['confidence', 'pnl'])

    if len(trades_with_conf) < 5:
        print("  Skipped: confidence_analysis.png (not enough trades with confidence)")
        return trades_df

    # Create confidence analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. P&L by confidence bucket
    ax1 = axes[0, 0]
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1.0]
    trades_with_conf['conf_bucket'] = pd.cut(trades_with_conf['confidence'], bins=bins)
    bucket_stats = trades_with_conf.groupby('conf_bucket', observed=True).agg({
        'pnl': ['sum', 'mean', 'count'],
    }).round(2)
    bucket_stats.columns = ['total_pnl', 'avg_pnl', 'count']
    bucket_stats = bucket_stats.reset_index()

    x_labels = [f"{b.left:.2f}-{b.right:.2f}" for b in bucket_stats['conf_bucket']]
    colors = ['green' if p > 0 else 'red' for p in bucket_stats['total_pnl']]
    bars = ax1.bar(range(len(x_labels)), bucket_stats['total_pnl'], color=colors, alpha=0.7)
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.set_xlabel('Confidence Range')
    ax1.set_ylabel('Total P&L ($)')
    ax1.set_title('Total P&L by Confidence Level', fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, bucket_stats['count'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'n={int(count)}', ha='center', va='bottom', fontsize=9)

    # 2. Win rate by confidence
    ax2 = axes[0, 1]
    win_rates = trades_with_conf.groupby('conf_bucket', observed=True).apply(
        lambda x: (x['pnl'] > 0).mean() * 100
    )
    ax2.bar(range(len(x_labels)), win_rates.values, color='#2E86AB', alpha=0.7)
    ax2.set_xticks(range(len(x_labels)))
    ax2.set_xticklabels(x_labels, rotation=45)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    ax2.set_xlabel('Confidence Range')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate by Confidence Level', fontweight='bold')
    ax2.legend()

    # 3. Scatter: Confidence vs P&L
    ax3 = axes[1, 0]
    colors = ['green' if p > 0 else 'red' for p in trades_with_conf['pnl']]
    ax3.scatter(trades_with_conf['confidence'], trades_with_conf['pnl'],
                c=colors, alpha=0.6, edgecolors='white', linewidth=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Model Confidence')
    ax3.set_ylabel('Trade P&L ($)')
    ax3.set_title('Confidence vs Trade P&L', fontweight='bold')

    # 4. Confidence distribution
    ax4 = axes[1, 1]
    winners = trades_with_conf[trades_with_conf['pnl'] > 0]['confidence']
    losers = trades_with_conf[trades_with_conf['pnl'] <= 0]['confidence']
    ax4.hist(winners, bins=15, alpha=0.6, label=f'Winners (n={len(winners)})', color='green')
    ax4.hist(losers, bins=15, alpha=0.6, label=f'Losers (n={len(losers)})', color='red')
    ax4.set_xlabel('Model Confidence')
    ax4.set_ylabel('Count')
    ax4.set_title('Confidence Distribution: Winners vs Losers', fontweight='bold')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(run_dir / "confidence_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: confidence_analysis.png")

    # Print summary
    print(f"\n  Confidence Analysis Summary:")
    for _, row in bucket_stats.iterrows():
        bucket = row['conf_bucket']
        wr = win_rates.get(bucket, 0)
        print(f"    {bucket.left:.2f}-{bucket.right:.2f}: {int(row['count']):3d} trades, "
              f"${row['total_pnl']:8.2f} total, {wr:.1f}% win rate")

    return trades_df


def analyze_by_market_regime(
    run_dir: Path,
    trades_df: pd.DataFrame,
    spy_data: pd.DataFrame,
    vxx_data: pd.DataFrame = None,
):
    """Analyze trade performance by market regime."""
    if trades_df.empty or 'entry_date' not in trades_df.columns or spy_data is None:
        print("  Skipped: market_regime_analysis.png (insufficient data)")
        return

    print("  Analyzing performance by market regime...")

    trades_df = trades_df.copy()

    # Compute market regime indicators
    spy_data = spy_data.copy()
    spy_data['return_5d'] = spy_data['close'].pct_change(5)
    spy_data['return_20d'] = spy_data['close'].pct_change(20)
    spy_data['volatility_20d'] = spy_data['close'].pct_change().rolling(20).std() * np.sqrt(252)
    spy_data['sma_50'] = spy_data['close'].rolling(50).mean()
    spy_data['above_sma50'] = spy_data['close'] > spy_data['sma_50']

    # Map trades to market conditions at entry
    market_conditions = []
    for _, trade in trades_df.iterrows():
        entry_date = pd.Timestamp(trade['entry_date'])
        if entry_date in spy_data.index:
            row = spy_data.loc[entry_date]
        else:
            idx = spy_data.index.get_indexer([entry_date], method='nearest')[0]
            row = spy_data.iloc[idx]

        conditions = {
            'spy_return_5d': row.get('return_5d', np.nan),
            'spy_return_20d': row.get('return_20d', np.nan),
            'volatility': row.get('volatility_20d', np.nan),
            'above_sma50': row.get('above_sma50', np.nan),
        }
        market_conditions.append(conditions)

    conditions_df = pd.DataFrame(market_conditions)
    trades_df = pd.concat([trades_df.reset_index(drop=True), conditions_df], axis=1)

    # Define regimes
    trades_df['trend_regime'] = trades_df['above_sma50'].map({True: 'Uptrend', False: 'Downtrend'})
    trades_df['momentum_regime'] = pd.cut(
        trades_df['spy_return_20d'],
        bins=[-np.inf, -0.02, 0.02, np.inf],
        labels=['Bear', 'Neutral', 'Bull']
    )
    trades_df['vol_regime'] = pd.cut(
        trades_df['volatility'],
        bins=[-np.inf, 0.12, 0.20, np.inf],
        labels=['Low Vol', 'Normal Vol', 'High Vol']
    )

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. P&L by trend regime
    ax1 = axes[0, 0]
    regime_pnl = trades_df.groupby('trend_regime')['pnl'].agg(['sum', 'mean', 'count'])
    colors = ['green' if p > 0 else 'red' for p in regime_pnl['sum']]
    bars = ax1.bar(regime_pnl.index, regime_pnl['sum'], color=colors, alpha=0.7)
    ax1.set_ylabel('Total P&L ($)')
    ax1.set_title('P&L by Trend Regime (SPY vs 50-SMA)', fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for bar, count in zip(bars, regime_pnl['count']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'n={int(count)}', ha='center', va='bottom', fontsize=10)

    # 2. P&L by momentum regime
    ax2 = axes[0, 1]
    mom_pnl = trades_df.groupby('momentum_regime', observed=True)['pnl'].agg(['sum', 'mean', 'count'])
    colors = ['green' if p > 0 else 'red' for p in mom_pnl['sum']]
    bars = ax2.bar(range(len(mom_pnl)), mom_pnl['sum'], color=colors, alpha=0.7)
    ax2.set_xticks(range(len(mom_pnl)))
    ax2.set_xticklabels(mom_pnl.index)
    ax2.set_ylabel('Total P&L ($)')
    ax2.set_title('P&L by Momentum Regime (20-day SPY return)', fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for bar, count in zip(bars, mom_pnl['count']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'n={int(count)}', ha='center', va='bottom', fontsize=10)

    # 3. P&L by volatility regime
    ax3 = axes[1, 0]
    vol_pnl = trades_df.groupby('vol_regime', observed=True)['pnl'].agg(['sum', 'mean', 'count'])
    colors = ['green' if p > 0 else 'red' for p in vol_pnl['sum']]
    bars = ax3.bar(range(len(vol_pnl)), vol_pnl['sum'], color=colors, alpha=0.7)
    ax3.set_xticks(range(len(vol_pnl)))
    ax3.set_xticklabels(vol_pnl.index)
    ax3.set_ylabel('Total P&L ($)')
    ax3.set_title('P&L by Volatility Regime', fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for bar, count in zip(bars, vol_pnl['count']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'n={int(count)}', ha='center', va='bottom', fontsize=10)

    # 4. Win rate by regime (combined)
    ax4 = axes[1, 1]
    regimes = ['Uptrend', 'Downtrend', 'Bull', 'Bear', 'Low Vol', 'High Vol']
    win_rates = []
    counts = []
    for regime in regimes:
        if regime in ['Uptrend', 'Downtrend']:
            mask = trades_df['trend_regime'] == regime
        elif regime in ['Bull', 'Bear']:
            mask = trades_df['momentum_regime'] == regime
        else:
            mask = trades_df['vol_regime'] == regime

        if mask.sum() > 0:
            wr = (trades_df.loc[mask, 'pnl'] > 0).mean() * 100
            win_rates.append(wr)
            counts.append(mask.sum())
        else:
            win_rates.append(0)
            counts.append(0)

    colors = ['green' if wr > 50 else 'red' for wr in win_rates]
    bars = ax4.bar(range(len(regimes)), win_rates, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(regimes)))
    ax4.set_xticklabels(regimes, rotation=45, ha='right')
    ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_title('Win Rate by Market Regime', fontweight='bold')
    for bar, count in zip(bars, counts):
        if count > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'n={int(count)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(run_dir / "market_regime_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: market_regime_analysis.png")

    # Print summary
    print(f"\n  Market Regime Summary:")
    for regime_col, regime_name in [('trend_regime', 'Trend'), ('momentum_regime', 'Momentum'), ('vol_regime', 'Volatility')]:
        print(f"    {regime_name}:")
        for regime in trades_df[regime_col].dropna().unique():
            mask = trades_df[regime_col] == regime
            total_pnl = trades_df.loc[mask, 'pnl'].sum()
            wr = (trades_df.loc[mask, 'pnl'] > 0).mean() * 100
            n = mask.sum()
            print(f"      {regime}: {n} trades, ${total_pnl:.2f}, {wr:.1f}% win rate")


def plot_summary_dashboard(run_dir: Path, strategy_metrics: dict, spy_metrics: dict,
                           result, strategy_equity: pd.Series):
    """Create a summary dashboard."""
    fig = plt.figure(figsize=(16, 10))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Key Metrics Table (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')

    metrics_text = [
        ['Metric', 'Strategy', 'SPY'],
        ['Total Return', f"{strategy_metrics.get('total_return_pct', 0):.2f}%",
         f"{spy_metrics.get('total_return', 0):.2f}%"],
        ['Annualized Return', f"{strategy_metrics.get('annualized_return_pct', 0):.2f}%",
         f"{spy_metrics.get('annualized_return', 0):.2f}%"],
        ['Sharpe Ratio', f"{strategy_metrics.get('sharpe_ratio', 0):.2f}",
         f"{spy_metrics.get('sharpe', 0):.2f}"],
        ['Max Drawdown', f"{strategy_metrics.get('max_drawdown_pct', 0):.2f}%",
         f"{spy_metrics.get('max_dd', 0):.2f}%"],
        ['Total Trades', f"{result.metrics.total_trades}", '-'],
        ['Win Rate', f"{result.metrics.win_rate:.1f}%", '-'],
    ]

    table = ax1.table(cellText=metrics_text, loc='center', cellLoc='center',
                      colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color header row
    for j in range(3):
        table[(0, j)].set_facecolor('#2E86AB')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax1.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

    # 2. Equity Curve (top middle and right)
    ax2 = fig.add_subplot(gs[0, 1:])
    strategy_norm = (strategy_equity / strategy_equity.iloc[0]) * 100
    ax2.plot(strategy_norm.index, strategy_norm.values, linewidth=2, color='#2E86AB')
    ax2.set_title('Portfolio Value (Indexed)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 3. Monthly Returns (middle row)
    ax3 = fig.add_subplot(gs[1, :])
    returns = strategy_equity.pct_change().dropna()
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
    colors = ['green' if x > 0 else 'red' for x in monthly.values]
    ax3.bar(monthly.index, monthly.values, color=colors, alpha=0.7, width=20)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('Monthly Returns (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Return (%)')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 4. Trade P&L Distribution (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    if not result.trades.empty and 'pnl' in result.trades.columns:
        pnl = result.trades['pnl'].dropna()
        if len(pnl) > 0:
            ax4.hist(pnl, bins=20, color='#2E86AB', edgecolor='white', alpha=0.7)
            ax4.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax4.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('P&L ($)')

    # 5. Drawdown (bottom middle and right)
    ax5 = fig.add_subplot(gs[2, 1:])
    rolling_max = strategy_equity.expanding().max()
    drawdown = ((strategy_equity - rolling_max) / rolling_max) * 100
    ax5.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='red')
    ax5.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Drawdown (%)')
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.suptitle('Metalabeling Strategy Backtest Report', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(run_dir / "dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dashboard.png")


def main():
    parser = argparse.ArgumentParser(description="Backtest metalabeling strategy")
    parser.add_argument(
        "--universe",
        choices=["small", "medium", "large"],
        default=None,
        help="Universe size (default: use universe from trained model)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.60,
        help="Confidence threshold for trade signals (default: 0.60)",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Record this run in the backtest registry (backtest_registry.json)",
    )
    parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="Comment to attach when registering (describes what changed)",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        default=[],
        help="Tag for this run (can repeat: --tag baseline --tag phase2)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Metalabeling Strategy Backtest")
    print("=" * 70)

    # Create run directory
    run_dir = create_run_directory()
    print(f"\nRun directory: {run_dir}")

    # Configuration
    CONFIDENCE_THRESHOLD = args.confidence
    INITIAL_CASH = 100_000
    TEST_PERIOD_DAYS = 365
    MAX_HOLDING_DAYS = 5  # Time-based exit to match training labels (set to None to disable)

    # Load model
    print("\nLoading trained model...")
    model, model_info = load_model()
    print(f"  Training date: {model_info.get('training_date', 'unknown')}")
    print(f"  Training samples: {model_info.get('n_samples', 'unknown')}")
    print(f"  Trained universe: {model_info.get('universe_size', 'unknown')}")

    # Display embargo metadata if available
    embargo_end = model_info.get("embargo_end_date")
    training_end = model_info.get("training_end_date")

    if embargo_end:
        print(f"  Training ended: {training_end}")
        print(f"  Embargo ends: {embargo_end}")
    else:
        print(f"  WARNING: Legacy model - no embargo metadata")

    # Determine which universe to use
    if args.universe:
        # User specified universe
        universe = get_universe(args.universe)
        print(f"\nUsing user-specified universe: {args.universe}")
    elif "universe_tradeable" in model_info and "universe_context" in model_info:
        # Use universe from model
        universe = TickerUniverse(
            tradeable=model_info["universe_tradeable"],
            context=model_info["universe_context"],
            name=model_info.get("universe_size", "from_model"),
        )
        print(f"\nUsing universe from trained model: {universe}")
    else:
        # Fallback to small (for older models)
        universe = get_universe("small")
        print(f"\nNo universe in model, falling back to: small")

    # Load data
    print("\nLoading market data...")
    dm = DataManager()
    data = dm.get_multi(universe.all_tickers, refresh=False, show_progress=False)

    # Determine test period
    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    end_date = all_dates[-1]
    test_start_intended = end_date - pd.Timedelta(days=TEST_PERIOD_DAYS)

    # Validate against training period (backtest must start AFTER training data ends)
    if training_end:
        training_end_date = pd.Timestamp(training_end)
        if test_start_intended <= training_end_date:
            print(f"\n  Adjusting backtest to start after training data ends")
            print(f"  Training ended: {training_end_date.date()}")
            print(f"  Backtest will start: {(training_end_date + pd.Timedelta(days=1)).date()}")
            test_start = training_end_date + pd.Timedelta(days=1)
        else:
            test_start = test_start_intended
            print(f"\n  Backtest start {test_start.date()} is after training end (OK)")
    elif embargo_end:
        # Fallback to embargo_end if no training_end (shouldn't happen with new models)
        embargo_date = pd.Timestamp(embargo_end)
        if test_start_intended <= embargo_date:
            print(f"\n  WARNING: Using embargo_end as cutoff (no training_end_date)")
            test_start = embargo_date + pd.Timedelta(days=1)
        else:
            test_start = test_start_intended
    else:
        # Legacy model: use training_date as conservative cutoff
        training_date = model_info.get("training_date")
        if training_date:
            training_date_ts = pd.Timestamp(training_date)
            if test_start_intended <= training_date_ts:
                print(f"\n  WARNING: Backtest may overlap with training period!")
                print(f"  Adjusting start to after training date: {(training_date_ts + pd.Timedelta(days=1)).date()}")
                test_start = training_date_ts + pd.Timedelta(days=1)
            else:
                test_start = test_start_intended
        else:
            test_start = test_start_intended
            print(f"\n  WARNING: Cannot validate train/test separation (no date metadata)")

    print(f"\nBacktest period: {test_start.date()} to {end_date.date()}")

    # Filter data
    lookback_start = test_start - pd.Timedelta(days=100)
    test_data = {}
    for ticker, df in data.items():
        filtered = df.loc[lookback_start:]
        if len(filtered) > 60:
            test_data[ticker] = filtered

    print(f"  Tickers with sufficient data: {len(test_data)}")

    # Load barrier params from model metadata (triple barrier consistency)
    MODEL_PROFIT_TARGET = model_info.get("profit_target")
    MODEL_STOP_LOSS = model_info.get("stop_loss")

    # Create strategy
    print("\nInitializing strategy...")
    signal_gen = PrimarySignalGenerator()

    # Load earnings calendar (P2-4)
    print("  Loading earnings calendar...")
    earnings_cal = EarningsCalendar()
    earnings_cal.load(universe.tradeable)
    tickers_with_earnings = sum(1 for t in universe.tradeable if earnings_cal.has_data(t))
    print(f"  Earnings data: {tickers_with_earnings}/{len(universe.tradeable)} tickers")

    feature_eng = MetaFeatureEngineering(
        context_tickers=universe.context,
        earnings_calendar=earnings_cal,
        holding_period=MAX_HOLDING_DAYS or 5,
    )

    regime_classifier = MarketRegimeClassifier()

    strategy = MetalabelingStrategy(
        meta_model=model,
        signal_generator=signal_gen,
        feature_engineering=feature_eng,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        context_tickers=universe.context,
        feature_names=model_info.get("feature_names"),
        regime_classifier=regime_classifier,
    )

    # Save configuration
    config = {
        "run_timestamp": datetime.now().isoformat(),
        "strategy": {
            "name": "metalabeling",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "primary_indicators": ["sma_crossover", "rsi_extremes", "macd_crossover", "bb_touch"],
            "regime_gating": True,
            "regime_bear_threshold": max(CONFIDENCE_THRESHOLD, 0.80),
            "regime_bull_threshold": CONFIDENCE_THRESHOLD * 0.90,
            "earnings_filter": True,
            "earnings_tickers_loaded": tickers_with_earnings,
        },
        "backtest": {
            "initial_cash": INITIAL_CASH,
            "test_start": str(test_start.date()),
            "test_end": str(end_date.date()),
            "test_period_days": TEST_PERIOD_DAYS,
            "max_holding_days": MAX_HOLDING_DAYS,
            "profit_target": MODEL_PROFIT_TARGET,
            "stop_loss": MODEL_STOP_LOSS,
            "n_tickers": len(test_data),
            "sector_cap": 2,
            "correlation_limit": 0.70,
            "correlation_lookback": 30,
        },
        "model": {
            "training_date": model_info.get("training_date"),
            "n_training_samples": model_info.get("n_samples"),
            "n_features": len(model_info.get("feature_names", [])),
            "holding_period": model_info.get("holding_period"),
            "min_return_threshold": model_info.get("min_return"),
            "profit_target": MODEL_PROFIT_TARGET,
            "stop_loss": MODEL_STOP_LOSS,
            "is_calibrated": model_info.get("is_calibrated", False),
        },
        "universe": {
            "tradeable": universe.tradeable,
            "context": universe.context,
        }
    }
    save_config(run_dir, config)

    # Run backtest
    print("\nRunning backtest...")
    if MAX_HOLDING_DAYS:
        print(f"  Time-based exit: {MAX_HOLDING_DAYS} days")
    else:
        print("  Time-based exit: DISABLED")
    if MODEL_PROFIT_TARGET:
        print(f"  Profit target:   {MODEL_PROFIT_TARGET:.1%}")
    if MODEL_STOP_LOSS:
        print(f"  Stop loss:       {MODEL_STOP_LOSS:.1%}")
    print(f"  Regime gating:   ENABLED (BULL={CONFIDENCE_THRESHOLD*0.90:.2f}, NEUTRAL={CONFIDENCE_THRESHOLD:.2f}, BEAR={max(CONFIDENCE_THRESHOLD,0.80):.2f})")
    sector_map = get_sector_map(universe.tradeable)
    print(f"  Sector cap:      {len(set(sector_map.values()))} sectors, max 2 positions each")
    print(f"  Corr filter:     r > 0.70 over 30-day window")
    bt = Backtester(
        strategy=strategy,
        initial_cash=INITIAL_CASH,
        max_holding_days=MAX_HOLDING_DAYS,
        profit_target=MODEL_PROFIT_TARGET,
        stop_loss=MODEL_STOP_LOSS,
        sector_map=sector_map,
        max_sector_positions=2,
        correlation_limit=0.70,
        correlation_lookback=30,
    )
    result = bt.run(test_data, show_progress=True)

    print(f"\nSignals generated: {len(result.signals_history)}")

    # Calculate metrics
    equity = result.equity_curve.loc[test_start:]
    strategy_metrics = calculate_metrics(equity)

    # SPY benchmark
    spy_data = data.get("SPY")
    spy_equity = None
    spy_metrics = {}
    if spy_data is not None:
        spy_filtered = spy_data.loc[test_start:end_date]
        spy_equity = spy_filtered["close"]
        spy_return = (spy_equity.iloc[-1] / spy_equity.iloc[0]) - 1
        spy_returns = spy_equity.pct_change().dropna()
        spy_vol = spy_returns.std() * np.sqrt(252)
        spy_years = len(spy_filtered) / 252
        spy_annualized = (1 + spy_return) ** (1 / spy_years) - 1 if spy_years > 0 else 0
        spy_sharpe = spy_annualized / spy_vol if spy_vol > 0 else 0
        spy_rolling_max = spy_equity.expanding().max()
        spy_dd = ((spy_equity - spy_rolling_max) / spy_rolling_max).min()

        spy_metrics = {
            "total_return": spy_return * 100,
            "annualized_return": spy_annualized * 100,
            "volatility": spy_vol * 100,
            "sharpe": spy_sharpe,
            "max_dd": spy_dd * 100,
        }

    # Save results
    results = {
        "strategy_metrics": strategy_metrics,
        "spy_metrics": spy_metrics,
        "trade_stats": {
            "total_trades": result.metrics.total_trades,
            "win_rate": result.metrics.win_rate,
            "profit_factor": result.metrics.profit_factor,
            "avg_trade_pnl": result.metrics.avg_trade_pnl,
            "total_signals": len(result.signals_history),
        },
        "final_value": result.final_value,
    }
    save_results(run_dir, results)

    # Save trades and signals
    if not result.trades.empty:
        trades_df = result.trades.copy()
        # Add additional analysis columns
        if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
            trades_df['holding_days'] = (
                pd.to_datetime(trades_df['exit_date']) - pd.to_datetime(trades_df['entry_date'])
            ).dt.days
            # Determine exit reason
            trades_df['exit_reason'] = trades_df['holding_days'].apply(
                lambda d: 'time_based' if MAX_HOLDING_DAYS and d >= MAX_HOLDING_DAYS else 'signal_or_eod'
            )
        if 'pnl' in trades_df.columns:
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            trades_df['is_winner'] = trades_df['pnl'] > 0

        # Reorder columns for readability
        col_order = [
            'ticker', 'side', 'entry_date', 'exit_date', 'holding_days', 'exit_reason',
            'entry_price', 'exit_price', 'shares', 'pnl', 'pnl_pct', 'commission',
            'cumulative_pnl', 'is_winner'
        ]
        col_order = [c for c in col_order if c in trades_df.columns]
        trades_df = trades_df[col_order]

        trades_df.to_csv(run_dir / "trades.csv", index=False)
        print(f"  Saved: trades.csv ({len(trades_df)} trades)")

        # Save Excel version with formatting if openpyxl available
        try:
            trades_df.to_excel(run_dir / "trades.xlsx", index=False)
            print(f"  Saved: trades.xlsx")
        except Exception:
            pass  # openpyxl not installed

    if not result.signals_history.empty:
        result.signals_history.to_csv(run_dir / "signals.csv", index=False)
        print(f"  Saved: signals.csv ({len(result.signals_history)} signals)")

    # Save equity curve
    equity.to_csv(run_dir / "equity_curve.csv")
    print(f"  Saved: equity_curve.csv")

    # Generate visualizations
    print("\nGenerating visualizations...")
    if spy_equity is not None:
        plot_equity_curves(run_dir, equity, spy_equity)
        plot_drawdown(run_dir, equity, spy_equity)

    plot_monthly_returns(run_dir, equity)
    plot_rolling_sharpe(run_dir, equity)
    plot_trade_analysis(run_dir, result.trades)
    plot_summary_dashboard(run_dir, strategy_metrics, spy_metrics, result, equity)

    # Advanced analysis: confidence and market regime
    print("\nGenerating advanced analysis...")
    if not result.trades.empty:
        # Confidence analysis
        trades_with_conf = analyze_by_confidence(
            run_dir, result.trades, data, model, feature_eng, universe, model_info
        )

        # Market regime analysis
        spy_df = data.get("SPY")
        vxx_df = data.get("VXX")
        analyze_by_market_regime(run_dir, result.trades, spy_df, vxx_df)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Metalabeling':>15} {'SPY B&H':>15}")
    print("-" * 60)
    print(f"{'Total Return':<30} {strategy_metrics.get('total_return_pct', 0):>14.2f}% {spy_metrics.get('total_return', 0):>14.2f}%")
    print(f"{'Annualized Return':<30} {strategy_metrics.get('annualized_return_pct', 0):>14.2f}% {spy_metrics.get('annualized_return', 0):>14.2f}%")
    print(f"{'Volatility':<30} {strategy_metrics.get('volatility_pct', 0):>14.2f}% {spy_metrics.get('volatility', 0):>14.2f}%")
    print(f"{'Sharpe Ratio':<30} {strategy_metrics.get('sharpe_ratio', 0):>15.2f} {spy_metrics.get('sharpe', 0):>15.2f}")
    print(f"{'Max Drawdown':<30} {strategy_metrics.get('max_drawdown_pct', 0):>14.2f}% {spy_metrics.get('max_dd', 0):>14.2f}%")

    print(f"\n{'TRADE STATISTICS':^60}")
    print("-" * 60)
    print(f"{'Total Trades':<30} {result.metrics.total_trades:>15}")
    print(f"{'Win Rate':<30} {result.metrics.win_rate:>14.2f}%")
    print(f"{'Profit Factor':<30} {result.metrics.profit_factor:>15.2f}")
    print(f"{'Avg Trade P&L':<30} ${result.metrics.avg_trade_pnl:>14.2f}")
    print(f"{'Final Portfolio Value':<30} ${result.final_value:>14,.2f}")

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {run_dir}")
    print(f"{'=' * 70}")

    # Optionally record to the backtest registry
    if args.register:
        from futures.scripts.backtest_registry import record, print_comparison_table

        registry_metrics = {
            **strategy_metrics,
            "spy_total_return_pct": spy_metrics.get("total_return", None),
            "spy_sharpe_ratio":     spy_metrics.get("sharpe", None),
            "total_trades":         result.metrics.total_trades,
            "win_rate_pct":         result.metrics.win_rate,
            "profit_factor":        result.metrics.profit_factor,
        }
        registry_params = {
            "universe":             universe.name,
            "model_name":           model_info.get("training_date", "unknown"),
            "start_date":           str(test_start.date()),
            "end_date":             str(end_date.date()),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "max_holding_days":     MAX_HOLDING_DAYS,
            "initial_cash":         INITIAL_CASH,
            "n_tickers":            len(test_data),
            "run_dir":              str(run_dir),
            "regime_gating":        True,
            "earnings_filter":      True,
            "profit_target":        MODEL_PROFIT_TARGET,
            "stop_loss":            MODEL_STOP_LOSS,
            "sector_cap":           2,
            "correlation_limit":    0.70,
        }
        run_id = record(
            metrics=registry_metrics,
            params=registry_params,
            comment=args.comment,
            tags=args.tags or [],
        )
        print(f"\n✓ Registered as: {run_id}")
        print_comparison_table()

    return result, strategy_metrics, run_dir


if __name__ == "__main__":
    main()
