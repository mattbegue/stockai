"""
Diagnostic script to detect survivorship bias and other backtest flaws.

Run after a backtest to check for:
1. Survivorship bias indicators
2. Look-ahead bias signals
3. Market regime dependency
4. Baseline alpha from universe selection
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from futures.config import get_universe
from futures.data.storage import Storage


def load_run(run_dir: Path) -> dict:
    """Load all data from a backtest run."""
    data = {}

    # Load config
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            data["config"] = json.load(f)

    # Load results
    results_path = run_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            data["results"] = json.load(f)

    # Load trades
    trades_path = run_dir / "trades.csv"
    if trades_path.exists():
        data["trades"] = pd.read_csv(trades_path)

    # Load equity curve
    equity_path = run_dir / "equity_curve.csv"
    if equity_path.exists():
        data["equity"] = pd.read_csv(equity_path, index_col=0, parse_dates=True)

    return data


def load_cached_data(storage: Storage, tickers: list) -> dict:
    """Load data from SQLite cache."""
    data = {}
    for ticker in tickers:
        df = storage.load_prices(ticker)
        if df is not None and len(df) > 0:
            data[ticker] = df
    return data


def check_survivorship_bias(storage: Storage, universe, test_start: str, test_end: str) -> dict:
    """
    Check if the universe stocks outperformed the market.

    If survivorship bias exists, our universe stocks should have
    significantly outperformed SPY on average.
    """
    print("\n" + "=" * 70)
    print("SURVIVORSHIP BIAS CHECK")
    print("=" * 70)

    # Get data for all tradeable stocks
    print("Loading cached market data...")
    data = load_cached_data(storage, universe.tradeable + ["SPY"])

    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    # Calculate buy-and-hold returns for each stock
    stock_returns = {}
    for ticker, df in data.items():
        if ticker == "SPY":
            continue
        try:
            filtered = df.loc[test_start:test_end]
            if len(filtered) > 20:
                ret = (filtered["close"].iloc[-1] / filtered["close"].iloc[0]) - 1
                stock_returns[ticker] = ret * 100
        except Exception:
            pass

    # SPY return
    spy_df = data.get("SPY")
    spy_return = None
    if spy_df is not None:
        spy_filtered = spy_df.loc[test_start:test_end]
        spy_return = (spy_filtered["close"].iloc[-1] / spy_filtered["close"].iloc[0]) - 1
        spy_return *= 100

    returns_series = pd.Series(stock_returns)

    print(f"\nTest period: {test_start.date()} to {test_end.date()}")
    print(f"Universe stocks analyzed: {len(returns_series)}")

    print(f"\n{'Metric':<35} {'Value':>15}")
    print("-" * 50)
    print(f"{'SPY Buy & Hold Return':<35} {spy_return:>14.2f}%")
    print(f"{'Universe Mean Return':<35} {returns_series.mean():>14.2f}%")
    print(f"{'Universe Median Return':<35} {returns_series.median():>14.2f}%")
    print(f"{'Universe Std Dev':<35} {returns_series.std():>14.2f}%")
    print(f"{'% Stocks Beating SPY':<35} {(returns_series > spy_return).mean() * 100:>14.1f}%")
    print(f"{'% Stocks Positive':<35} {(returns_series > 0).mean() * 100:>14.1f}%")

    # Survivorship bias indicator
    excess_return = returns_series.mean() - spy_return
    print(f"\n{'Universe Excess Return vs SPY':<35} {excess_return:>14.2f}%")

    if excess_return > 5:
        print("\n⚠️  WARNING: Universe significantly outperforms SPY!")
        print("   This suggests survivorship bias - you're trading stocks")
        print("   that you already know performed well in this period.")
        bias_severity = "HIGH"
    elif excess_return > 2:
        print("\n⚠️  CAUTION: Universe moderately outperforms SPY.")
        print("   Some survivorship bias may be present.")
        bias_severity = "MODERATE"
    else:
        print("\n✓  Universe performance close to SPY.")
        print("   Survivorship bias appears limited.")
        bias_severity = "LOW"

    # Top and bottom performers
    print(f"\nTop 5 performers in universe:")
    for ticker, ret in returns_series.nlargest(5).items():
        print(f"  {ticker}: {ret:+.1f}%")

    print(f"\nBottom 5 performers in universe:")
    for ticker, ret in returns_series.nsmallest(5).items():
        print(f"  {ticker}: {ret:+.1f}%")

    return {
        "spy_return": spy_return,
        "universe_mean_return": returns_series.mean(),
        "universe_median_return": returns_series.median(),
        "excess_return": excess_return,
        "pct_beating_spy": (returns_series > spy_return).mean() * 100,
        "bias_severity": bias_severity,
        "stock_returns": stock_returns,
    }


def check_trade_distribution(trades_df: pd.DataFrame) -> dict:
    """
    Analyze trade P&L distribution for signs of bias.

    In a biased backtest:
    - Losing trades tend to be small (stocks recovered)
    - Winning trades tend to be large (we picked winners)
    """
    print("\n" + "=" * 70)
    print("TRADE P&L DISTRIBUTION ANALYSIS")
    print("=" * 70)

    if trades_df.empty or "pnl" not in trades_df.columns:
        print("No trades to analyze.")
        return {}

    pnl = trades_df["pnl"].dropna()
    winners = pnl[pnl > 0]
    losers = pnl[pnl <= 0]

    print(f"\n{'Metric':<35} {'Value':>15}")
    print("-" * 50)
    print(f"{'Total Trades':<35} {len(pnl):>15}")
    print(f"{'Winners':<35} {len(winners):>15}")
    print(f"{'Losers':<35} {len(losers):>15}")
    print(f"{'Win Rate':<35} {len(winners)/len(pnl)*100:>14.1f}%")

    print(f"\n{'Average Win':<35} ${winners.mean():>14.2f}")
    print(f"{'Average Loss':<35} ${losers.mean():>14.2f}")
    print(f"{'Largest Win':<35} ${winners.max():>14.2f}")
    print(f"{'Largest Loss':<35} ${losers.min():>14.2f}")

    # Win/Loss ratio
    if len(losers) > 0 and losers.mean() != 0:
        win_loss_ratio = abs(winners.mean() / losers.mean())
        print(f"{'Win/Loss Ratio':<35} {win_loss_ratio:>15.2f}")

    # Asymmetry check
    print(f"\n{'Median Win':<35} ${winners.median():>14.2f}")
    print(f"{'Median Loss':<35} ${losers.median():>14.2f}")

    # Check for suspicious patterns
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0

    if avg_win > avg_loss * 2:
        print("\n⚠️  WARNING: Average win is 2x+ larger than average loss!")
        print("   This asymmetry could indicate survivorship bias -")
        print("   losing positions may have recovered because we're trading survivors.")

    # P&L by ticker
    if "ticker" in trades_df.columns:
        ticker_pnl = trades_df.groupby("ticker")["pnl"].agg(["sum", "count", "mean"])
        ticker_pnl.columns = ["total_pnl", "num_trades", "avg_pnl"]
        ticker_pnl = ticker_pnl.sort_values("total_pnl", ascending=False)

        print(f"\nTop 5 tickers by total P&L:")
        for ticker, row in ticker_pnl.head(5).iterrows():
            print(f"  {ticker}: ${row['total_pnl']:+,.0f} ({row['num_trades']:.0f} trades, avg ${row['avg_pnl']:+.0f})")

        print(f"\nBottom 5 tickers by total P&L:")
        for ticker, row in ticker_pnl.tail(5).iterrows():
            print(f"  {ticker}: ${row['total_pnl']:+,.0f} ({row['num_trades']:.0f} trades, avg ${row['avg_pnl']:+.0f})")

        # Concentration check
        top5_pnl = ticker_pnl.head(5)["total_pnl"].sum()
        total_pnl = ticker_pnl["total_pnl"].sum()
        if total_pnl > 0:
            concentration = top5_pnl / total_pnl * 100
            print(f"\n{'Top 5 tickers % of total P&L':<35} {concentration:>14.1f}%")

            if concentration > 50:
                print("\n⚠️  WARNING: Returns highly concentrated in few stocks!")
                print("   Strategy may just be riding a few big winners.")

    return {
        "total_trades": len(pnl),
        "win_rate": len(winners) / len(pnl) * 100,
        "avg_win": winners.mean() if len(winners) > 0 else 0,
        "avg_loss": losers.mean() if len(losers) > 0 else 0,
    }


def check_regime_performance(storage: Storage, equity_curve: pd.DataFrame) -> dict:
    """
    Check strategy performance during different market regimes.
    """
    print("\n" + "=" * 70)
    print("MARKET REGIME ANALYSIS")
    print("=" * 70)

    if equity_curve.empty:
        print("No equity curve to analyze.")
        return {}

    # Get SPY data for regime detection
    spy_data = storage.load_prices("SPY")
    if spy_data is None:
        print("Could not load SPY data for regime analysis.")
        return {}

    # Align dates
    equity = equity_curve["equity"] if "equity" in equity_curve.columns else equity_curve.iloc[:, 0]

    # Calculate rolling metrics
    equity_returns = equity.pct_change().dropna()

    # Identify drawdown periods in SPY
    spy_aligned = spy_data.loc[equity.index[0]:equity.index[-1]]["close"]
    spy_rolling_max = spy_aligned.expanding().max()
    spy_drawdown = (spy_aligned - spy_rolling_max) / spy_rolling_max

    # Split into regimes
    bull_mask = spy_drawdown > -0.05  # Less than 5% from highs
    correction_mask = (spy_drawdown <= -0.05) & (spy_drawdown > -0.10)
    bear_mask = spy_drawdown <= -0.10  # 10%+ drawdown

    # Calculate strategy returns in each regime
    aligned_returns = equity_returns.reindex(spy_drawdown.index).dropna()

    regimes = {
        "Bull (SPY <5% from high)": bull_mask,
        "Correction (SPY 5-10% down)": correction_mask,
        "Bear (SPY >10% down)": bear_mask,
    }

    print(f"\n{'Regime':<30} {'Days':>8} {'Avg Daily':>12} {'Total':>12}")
    print("-" * 65)

    regime_stats = {}
    for regime_name, mask in regimes.items():
        regime_returns = aligned_returns[mask.reindex(aligned_returns.index).fillna(False)]
        if len(regime_returns) > 0:
            avg_ret = regime_returns.mean() * 100
            total_ret = (1 + regime_returns).prod() - 1
            total_ret *= 100
            print(f"{regime_name:<30} {len(regime_returns):>8} {avg_ret:>11.3f}% {total_ret:>11.2f}%")
            regime_stats[regime_name] = {
                "days": len(regime_returns),
                "avg_daily_return": avg_ret,
                "total_return": total_ret,
            }

    # Check if strategy only works in bull markets
    if "Bull (SPY <5% from high)" in regime_stats and "Bear (SPY >10% down)" in regime_stats:
        bull_ret = regime_stats["Bull (SPY <5% from high)"]["avg_daily_return"]
        bear_ret = regime_stats["Bear (SPY >10% down)"]["avg_daily_return"]

        if bull_ret > 0 and bear_ret < 0:
            print("\n⚠️  WARNING: Strategy profitable in bull markets, loses in bear markets.")
            print("   This is a common sign of a long-biased strategy with no real edge.")
        elif bull_ret > 0 and bear_ret > 0:
            print("\n✓  Strategy profitable in both bull and bear regimes.")
            print("   This is a positive sign (though sample size matters).")

    return regime_stats


def check_alpha_vs_beta(trades_df: pd.DataFrame, stock_returns: dict) -> dict:
    """
    Check if strategy P&L correlates with underlying stock performance.

    High correlation = just riding beta, not generating alpha.
    """
    print("\n" + "=" * 70)
    print("ALPHA VS BETA ANALYSIS")
    print("=" * 70)

    if trades_df.empty or "ticker" not in trades_df.columns:
        print("No trade data available.")
        return {}

    if not stock_returns:
        print("No stock returns data available.")
        return {}

    # Get P&L by ticker
    ticker_pnl = trades_df.groupby("ticker")["pnl"].sum()

    # Match with stock returns
    matched = []
    for ticker in ticker_pnl.index:
        if ticker in stock_returns:
            matched.append({
                "ticker": ticker,
                "strategy_pnl": ticker_pnl[ticker],
                "stock_return": stock_returns[ticker],
            })

    if len(matched) < 10:
        print(f"Only {len(matched)} matched tickers - insufficient for correlation analysis.")
        return {}

    matched_df = pd.DataFrame(matched)

    # Calculate correlation
    correlation = matched_df["strategy_pnl"].corr(matched_df["stock_return"])

    print(f"\nCorrelation between strategy P&L and stock returns: {correlation:.3f}")

    if correlation > 0.7:
        print("\n⚠️  WARNING: High correlation with underlying stock returns!")
        print("   Strategy may just be capturing beta (market exposure)")
        print("   rather than generating true alpha.")
    elif correlation > 0.4:
        print("\n⚠️  CAUTION: Moderate correlation with stock returns.")
        print("   Some portion of returns may be from beta exposure.")
    else:
        print("\n✓  Low correlation with underlying stock returns.")
        print("   Strategy appears to generate returns independent of stock direction.")

    # Check if we're just long good stocks
    print(f"\nStocks where we made money:")
    profitable_tickers = matched_df[matched_df["strategy_pnl"] > 0]
    avg_return_winners = profitable_tickers["stock_return"].mean()
    print(f"  Count: {len(profitable_tickers)}")
    print(f"  Avg underlying stock return: {avg_return_winners:.1f}%")

    print(f"\nStocks where we lost money:")
    unprofitable_tickers = matched_df[matched_df["strategy_pnl"] <= 0]
    avg_return_losers = unprofitable_tickers["stock_return"].mean()
    print(f"  Count: {len(unprofitable_tickers)}")
    print(f"  Avg underlying stock return: {avg_return_losers:.1f}%")

    if avg_return_winners > avg_return_losers + 10:
        print("\n⚠️  Our winning trades were on stocks that went up more overall.")
        print("   This suggests we may be benefiting from stock selection luck,")
        print("   not timing skill.")

    return {
        "correlation": correlation,
        "avg_return_profitable_stocks": avg_return_winners,
        "avg_return_unprofitable_stocks": avg_return_losers,
    }


def generate_diagnostic_report(run_dir: Path, results: dict):
    """Generate a diagnostic visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Stock returns distribution
    ax1 = axes[0, 0]
    if "survivorship" in results and "stock_returns" in results["survivorship"]:
        returns = list(results["survivorship"]["stock_returns"].values())
        spy_ret = results["survivorship"]["spy_return"]
        ax1.hist(returns, bins=30, color="#2E86AB", edgecolor="white", alpha=0.7)
        ax1.axvline(x=spy_ret, color="red", linestyle="--", linewidth=2, label=f"SPY: {spy_ret:.1f}%")
        ax1.axvline(x=np.mean(returns), color="orange", linestyle="--", linewidth=2,
                    label=f"Universe Mean: {np.mean(returns):.1f}%")
        ax1.set_xlabel("Buy & Hold Return (%)")
        ax1.set_ylabel("Number of Stocks")
        ax1.set_title("Universe Stock Returns vs SPY\n(Survivorship Bias Check)")
        ax1.legend()

    # 2. Trade P&L distribution
    ax2 = axes[0, 1]
    if "trades" in results and "pnl_data" in results["trades"]:
        pnl = results["trades"]["pnl_data"]
        colors = ["green" if x > 0 else "red" for x in pnl]
        ax2.hist(pnl, bins=30, color="#2E86AB", edgecolor="white", alpha=0.7)
        ax2.axvline(x=0, color="black", linestyle="-", linewidth=2)
        ax2.axvline(x=np.mean(pnl), color="orange", linestyle="--",
                    label=f"Mean: ${np.mean(pnl):.0f}")
        ax2.set_xlabel("Trade P&L ($)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Trade P&L Distribution")
        ax2.legend()

    # 3. Regime performance
    ax3 = axes[1, 0]
    if "regimes" in results and results["regimes"]:
        regimes = results["regimes"]
        names = list(regimes.keys())
        returns = [regimes[r]["total_return"] for r in names]
        colors = ["green" if r > 0 else "red" for r in returns]
        bars = ax3.bar(range(len(names)), returns, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels([n.split("(")[0].strip() for n in names], rotation=15)
        ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax3.set_ylabel("Total Return (%)")
        ax3.set_title("Strategy Performance by Market Regime")

        # Add value labels
        for bar, ret in zip(bars, returns):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{ret:.1f}%", ha="center", va="bottom", fontsize=10)

    # 4. Summary scorecard
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Build scorecard text
    scorecard = []
    scorecard.append("DIAGNOSTIC SUMMARY")
    scorecard.append("=" * 40)

    if "survivorship" in results:
        severity = results["survivorship"].get("bias_severity", "UNKNOWN")
        excess = results["survivorship"].get("excess_return", 0)
        scorecard.append(f"\nSurvivorship Bias: {severity}")
        scorecard.append(f"  Universe excess return: {excess:+.1f}%")

    if "alpha_beta" in results and "correlation" in results["alpha_beta"]:
        corr = results["alpha_beta"]["correlation"]
        scorecard.append(f"\nBeta Correlation: {corr:.2f}")
        if corr > 0.7:
            scorecard.append("  ⚠️ High - mostly beta exposure")
        elif corr > 0.4:
            scorecard.append("  ⚠️ Moderate - some beta")
        else:
            scorecard.append("  ✓ Low - appears to be alpha")

    if "trades" in results:
        wr = results["trades"].get("win_rate", 0)
        scorecard.append(f"\nWin Rate: {wr:.1f}%")

    scorecard.append("\n" + "=" * 40)
    scorecard.append("\nRECOMMENDATIONS:")

    if "survivorship" in results and results["survivorship"].get("bias_severity") in ["HIGH", "MODERATE"]:
        scorecard.append("• Use point-in-time index constituents")
        scorecard.append("• Include delisted stocks in backtest")

    if "alpha_beta" in results and results["alpha_beta"].get("correlation", 0) > 0.5:
        scorecard.append("• Add short positions to reduce beta")
        scorecard.append("• Test with beta-hedged returns")

    ax4.text(0.1, 0.95, "\n".join(scorecard), transform=ax4.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Backtest Diagnostic Report", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(run_dir / "diagnostic_report.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {run_dir / 'diagnostic_report.png'}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose backtest for potential biases")
    parser.add_argument(
        "run_dir",
        type=str,
        nargs="?",
        help="Path to run directory (default: latest in runs/)",
    )
    args = parser.parse_args()

    # Find run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        runs_dir = Path("runs")
        if not runs_dir.exists():
            print("No runs/ directory found. Run a backtest first.")
            return

        # Get latest run
        run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
        if not run_dirs:
            print("No runs found in runs/ directory.")
            return
        run_dir = run_dirs[-1]

    print("=" * 70)
    print("BACKTEST DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    print(f"\nAnalyzing run: {run_dir}")

    # Load run data
    run_data = load_run(run_dir)

    if "config" not in run_data:
        print("Could not load run config.")
        return

    config = run_data["config"]

    # Get universe from config
    universe_tickers = config.get("universe", {})
    if "tradeable" in universe_tickers:
        from futures.config import TickerUniverse
        universe = TickerUniverse(
            tradeable=universe_tickers["tradeable"],
            context=universe_tickers.get("context", []),
            name="from_config",
        )
    else:
        universe = get_universe("small")

    # Load market data from cache
    storage = Storage()

    # Get test period from config
    backtest_config = config.get("backtest", {})
    test_start = backtest_config.get("test_start", "2023-01-01")
    test_end = backtest_config.get("test_end", "2024-01-01")

    # Run diagnostics
    results = {}

    # 1. Survivorship bias check
    survivorship = check_survivorship_bias(storage, universe, test_start, test_end)
    results["survivorship"] = survivorship

    # 2. Trade distribution analysis
    if "trades" in run_data:
        trade_analysis = check_trade_distribution(run_data["trades"])
        trade_analysis["pnl_data"] = run_data["trades"]["pnl"].dropna().tolist() if "pnl" in run_data["trades"].columns else []
        results["trades"] = trade_analysis

    # 3. Regime analysis
    if "equity" in run_data:
        regime_analysis = check_regime_performance(storage, run_data["equity"])
        results["regimes"] = regime_analysis

    # 4. Alpha vs Beta analysis
    if "trades" in run_data and "stock_returns" in survivorship:
        alpha_beta = check_alpha_vs_beta(run_data["trades"], survivorship["stock_returns"])
        results["alpha_beta"] = alpha_beta

    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING DIAGNOSTIC REPORT")
    print("=" * 70)
    generate_diagnostic_report(run_dir, results)

    # Save results
    diagnostic_path = run_dir / "diagnostic_results.json"

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results_json = convert_numpy(results)

    # Remove large data from JSON
    if "survivorship" in results_json and "stock_returns" in results_json["survivorship"]:
        del results_json["survivorship"]["stock_returns"]
    if "trades" in results_json and "pnl_data" in results_json["trades"]:
        del results_json["trades"]["pnl_data"]

    with open(diagnostic_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved: {diagnostic_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)

    issues = []
    if survivorship.get("bias_severity") == "HIGH":
        issues.append("• HIGH survivorship bias detected")
    elif survivorship.get("bias_severity") == "MODERATE":
        issues.append("• MODERATE survivorship bias detected")

    if "alpha_beta" in results and results["alpha_beta"].get("correlation", 0) > 0.7:
        issues.append("• High correlation with stock returns (beta exposure)")

    if results.get("regimes"):
        bear_regime = [r for r in results["regimes"] if "Bear" in r]
        if bear_regime and results["regimes"][bear_regime[0]].get("total_return", 0) < -5:
            issues.append("• Strategy loses significantly in bear markets")

    if issues:
        print("\n⚠️  POTENTIAL ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
        print("\n   Consider these factors when evaluating live deployment.")
    else:
        print("\n✓  No major red flags detected.")
        print("   However, always validate with true out-of-sample testing.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
