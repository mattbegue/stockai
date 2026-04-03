"""
Paper Trading Tracker

Run daily before market open to:
1. Log new predictions
2. Check outcomes for signals from 5 days ago
3. Calculate rolling forward metrics
4. Alert on performance deviation

Usage:
    # Log today's signals (run before market open)
    python -m futures.scripts.paper_trade

    # Check outcomes and update metrics (run after market close)
    python -m futures.scripts.paper_trade --check-outcomes

    # View current performance summary
    python -m futures.scripts.paper_trade --summary

    # Full daily workflow (log + check)
    python -m futures.scripts.paper_trade --full
"""

import argparse
import json
import pickle
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from futures.config import get_universe, TickerUniverse
from futures.data.storage import Storage
from futures.strategies import MetalabelingStrategy
from futures.metalabeling import PrimarySignalGenerator, MetaFeatureEngineering


# ============================================================================
# CONFIGURATION
# ============================================================================
PAPER_TRADE_DIR = Path("paper_trading")
PREDICTIONS_FILE = PAPER_TRADE_DIR / "predictions.csv"
OUTCOMES_FILE = PAPER_TRADE_DIR / "outcomes.csv"
METRICS_FILE = PAPER_TRADE_DIR / "forward_metrics.json"

HOLDING_PERIOD = 5  # Days
DEFAULT_CONFIDENCE_THRESHOLD = 0.55  # Default, can be overridden via --confidence

# Alert thresholds
ALERT_WIN_RATE_MIN = 0.50  # Alert if win rate drops below 50%
ALERT_SHARPE_MIN = 0.0  # Alert if Sharpe goes negative
ALERT_DRAWDOWN_MAX = -0.10  # Alert if drawdown exceeds 10%


# ============================================================================
# SETUP
# ============================================================================
def setup_directories():
    """Create paper trading directory structure."""
    PAPER_TRADE_DIR.mkdir(exist_ok=True)

    # Initialize predictions CSV if doesn't exist
    if not PREDICTIONS_FILE.exists():
        pd.DataFrame(columns=[
            "prediction_date", "prediction_time", "ticker", "signal",
            "confidence", "entry_price", "expected_exit_date", "outcome_checked"
        ]).to_csv(PREDICTIONS_FILE, index=False)

    # Initialize outcomes CSV if doesn't exist
    if not OUTCOMES_FILE.exists():
        pd.DataFrame(columns=[
            "prediction_date", "ticker", "signal", "confidence",
            "entry_price", "exit_price", "exit_date",
            "return_pct", "profitable", "holding_days"
        ]).to_csv(OUTCOMES_FILE, index=False)


def load_model():
    """Load the active trained model."""
    from futures.models.registry import ModelRegistry

    registry = ModelRegistry()
    return registry.load_active()


def load_market_data(tickers: list) -> dict:
    """Load latest market data from cache."""
    storage = Storage()
    data = {}
    for ticker in tickers:
        df = storage.load_prices(ticker)
        if df is not None and len(df) > 0:
            data[ticker] = df
    return data


# ============================================================================
# PREDICTION LOGGING
# ============================================================================
def log_predictions(confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
    """Generate and log today's predictions."""
    print("=" * 70)
    print(f"PAPER TRADING - Logging Predictions")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Confidence threshold: {confidence_threshold:.2f}")
    print("=" * 70)

    setup_directories()

    # Load model and get universe
    print("\nLoading model...")
    model, model_info = load_model()

    # Get universe from model
    if "universe_tradeable" in model_info:
        universe = TickerUniverse(
            tradeable=model_info["universe_tradeable"],
            context=model_info.get("universe_context", []),
            name=model_info.get("universe_size", "from_model"),
        )
    else:
        universe = get_universe("small")

    print(f"Universe: {universe}")

    # Load market data
    print("\nLoading market data...")
    data = load_market_data(universe.all_tickers)
    print(f"Loaded {len(data)} tickers")

    # Check data freshness
    latest_dates = {ticker: df.index[-1] for ticker, df in data.items() if len(df) > 0}
    if latest_dates:
        most_recent = max(latest_dates.values())
        print(f"Most recent data: {most_recent.date()}")

    # Create strategy
    print("\nGenerating signals...")
    signal_gen = PrimarySignalGenerator()
    feature_eng = MetaFeatureEngineering(context_tickers=universe.context)

    strategy = MetalabelingStrategy(
        meta_model=model,
        signal_generator=signal_gen,
        feature_engineering=feature_eng,
        confidence_threshold=confidence_threshold,
        context_tickers=universe.context,
        feature_names=model_info.get("feature_names"),
    )

    # Get signal details
    signal_details = strategy.get_signal_details(data)

    # Filter to actionable signals
    predictions = []
    today = date.today()
    prediction_time = datetime.now().strftime("%H:%M:%S")
    expected_exit = today + timedelta(days=HOLDING_PERIOD)

    for ticker, detail in signal_details.items():
        if ticker in universe.context:
            continue

        final_signal = detail.get("final_signal")
        if final_signal in ["BUY", "SELL"]:
            entry_price = data[ticker]["close"].iloc[-1] if ticker in data else None

            predictions.append({
                "prediction_date": today.isoformat(),
                "prediction_time": prediction_time,
                "ticker": ticker,
                "signal": final_signal,
                "confidence": detail.get("confidence"),
                "entry_price": entry_price,
                "expected_exit_date": expected_exit.isoformat(),
                "outcome_checked": False,
            })

    if not predictions:
        print("\nNo signals generated today.")
        return

    # Sort by confidence
    predictions.sort(key=lambda x: x["confidence"] or 0, reverse=True)

    # Log to CSV
    existing = pd.read_csv(PREDICTIONS_FILE)
    new_df = pd.DataFrame(predictions)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(PREDICTIONS_FILE, index=False)

    # Display
    print(f"\n{'='*70}")
    print(f"PREDICTIONS LOGGED: {len(predictions)} signals")
    print(f"{'='*70}")

    print(f"\n{'Ticker':<8} {'Signal':<6} {'Confidence':>10} {'Entry Price':>12}")
    print("-" * 40)
    for p in predictions[:20]:  # Show top 20
        conf_str = f"{p['confidence']:.1%}" if p['confidence'] else "N/A"
        price_str = f"${p['entry_price']:.2f}" if p['entry_price'] else "N/A"
        print(f"{p['ticker']:<8} {p['signal']:<6} {conf_str:>10} {price_str:>12}")

    if len(predictions) > 20:
        print(f"... and {len(predictions) - 20} more")

    print(f"\nPredictions saved to: {PREDICTIONS_FILE}")
    print(f"Expected exit date: {expected_exit}")


# ============================================================================
# OUTCOME CHECKING
# ============================================================================
def check_outcomes():
    """Check outcomes for predictions that have reached their holding period."""
    print("=" * 70)
    print(f"PAPER TRADING - Checking Outcomes")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    setup_directories()

    # Load predictions
    predictions = pd.read_csv(PREDICTIONS_FILE)
    if predictions.empty:
        print("\nNo predictions to check.")
        return

    # Find unchecked predictions past their exit date
    today = date.today()
    predictions["expected_exit_date"] = pd.to_datetime(predictions["expected_exit_date"]).dt.date
    predictions["prediction_date"] = pd.to_datetime(predictions["prediction_date"]).dt.date

    to_check = predictions[
        (predictions["outcome_checked"] == False) &
        (predictions["expected_exit_date"] <= today)
    ]

    if to_check.empty:
        print("\nNo outcomes ready to check.")
        pending = predictions[predictions["outcome_checked"] == False]
        if not pending.empty:
            next_check = pending["expected_exit_date"].min()
            print(f"Next outcome check: {next_check}")
        return

    print(f"\nChecking {len(to_check)} predictions...")

    # Load market data
    tickers_to_check = to_check["ticker"].unique().tolist()
    data = load_market_data(tickers_to_check)

    # Check each prediction
    outcomes = []
    checked_indices = []

    for idx, row in to_check.iterrows():
        ticker = row["ticker"]
        if ticker not in data:
            print(f"  {ticker}: No data available")
            continue

        df = data[ticker]
        entry_date = pd.Timestamp(row["prediction_date"])
        expected_exit = pd.Timestamp(row["expected_exit_date"])

        # Find actual entry and exit prices
        # Entry: close on prediction date (or next available)
        entry_mask = df.index >= entry_date
        if not entry_mask.any():
            continue
        actual_entry_date = df.index[entry_mask][0]
        entry_price = row["entry_price"] or df.loc[actual_entry_date, "close"]

        # Exit: close on expected exit date (or next available)
        exit_mask = df.index >= expected_exit
        if not exit_mask.any():
            # Use latest available
            actual_exit_date = df.index[-1]
        else:
            actual_exit_date = df.index[exit_mask][0]
        exit_price = df.loc[actual_exit_date, "close"]

        # Calculate return
        if row["signal"] == "BUY":
            return_pct = (exit_price / entry_price - 1) * 100
        else:  # SELL signal
            return_pct = (entry_price / exit_price - 1) * 100

        profitable = return_pct > 0
        holding_days = (actual_exit_date - actual_entry_date).days

        outcomes.append({
            "prediction_date": row["prediction_date"],
            "ticker": ticker,
            "signal": row["signal"],
            "confidence": row["confidence"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_date": actual_exit_date.date().isoformat(),
            "return_pct": return_pct,
            "profitable": profitable,
            "holding_days": holding_days,
        })

        checked_indices.append(idx)

        status = "✓" if profitable else "✗"
        print(f"  {status} {ticker}: {return_pct:+.2f}% ({row['signal']})")

    if not outcomes:
        print("\nNo outcomes could be calculated.")
        return

    # Save outcomes
    existing_outcomes = pd.read_csv(OUTCOMES_FILE)
    new_outcomes = pd.DataFrame(outcomes)
    combined_outcomes = pd.concat([existing_outcomes, new_outcomes], ignore_index=True)
    combined_outcomes.to_csv(OUTCOMES_FILE, index=False)

    # Mark predictions as checked
    predictions.loc[checked_indices, "outcome_checked"] = True
    predictions.to_csv(PREDICTIONS_FILE, index=False)

    # Summary
    print(f"\n{'='*70}")
    print("OUTCOME SUMMARY")
    print(f"{'='*70}")

    new_df = pd.DataFrame(outcomes)
    winners = new_df[new_df["profitable"] == True]
    losers = new_df[new_df["profitable"] == False]

    print(f"\nThis batch:")
    print(f"  Checked: {len(outcomes)}")
    print(f"  Winners: {len(winners)} ({len(winners)/len(outcomes)*100:.1f}%)")
    print(f"  Losers: {len(losers)} ({len(losers)/len(outcomes)*100:.1f}%)")
    print(f"  Avg return: {new_df['return_pct'].mean():+.2f}%")

    print(f"\nOutcomes saved to: {OUTCOMES_FILE}")

    # Update and check metrics
    update_metrics()


# ============================================================================
# METRICS AND ALERTS
# ============================================================================
def update_metrics():
    """Calculate rolling forward metrics and check for alerts."""
    outcomes = pd.read_csv(OUTCOMES_FILE)

    if outcomes.empty or len(outcomes) < 5:
        print("\nNot enough outcomes for metrics (need at least 5).")
        return

    # Calculate metrics
    total_trades = len(outcomes)
    winners = outcomes[outcomes["profitable"] == True]
    win_rate = len(winners) / total_trades

    avg_return = outcomes["return_pct"].mean()
    std_return = outcomes["return_pct"].std()

    # Approximate Sharpe (assuming 0 risk-free rate, daily returns)
    if std_return > 0:
        sharpe = (avg_return / std_return) * np.sqrt(252 / HOLDING_PERIOD)
    else:
        sharpe = 0

    # Calculate cumulative equity curve
    outcomes_sorted = outcomes.sort_values("exit_date")
    cumulative_returns = (1 + outcomes_sorted["return_pct"] / 100).cumprod()

    # Max drawdown
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Recent performance (last 20 trades)
    recent = outcomes_sorted.tail(20)
    recent_win_rate = (recent["profitable"] == True).mean()
    recent_avg_return = recent["return_pct"].mean()

    metrics = {
        "last_updated": datetime.now().isoformat(),
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_return_pct": avg_return,
        "std_return_pct": std_return,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 100,
        "cumulative_return_pct": (cumulative_returns.iloc[-1] - 1) * 100,
        "recent_20_win_rate": recent_win_rate,
        "recent_20_avg_return": recent_avg_return,
    }

    # Save metrics
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    # Check for alerts
    alerts = []

    if win_rate < ALERT_WIN_RATE_MIN:
        alerts.append(f"⚠️  WIN RATE ALERT: {win_rate:.1%} (threshold: {ALERT_WIN_RATE_MIN:.1%})")

    if sharpe < ALERT_SHARPE_MIN:
        alerts.append(f"⚠️  SHARPE ALERT: {sharpe:.2f} (threshold: {ALERT_SHARPE_MIN:.2f})")

    if max_drawdown < ALERT_DRAWDOWN_MAX:
        alerts.append(f"⚠️  DRAWDOWN ALERT: {max_drawdown:.1%} (threshold: {ALERT_DRAWDOWN_MAX:.1%})")

    if recent_win_rate < ALERT_WIN_RATE_MIN and len(recent) >= 10:
        alerts.append(f"⚠️  RECENT PERFORMANCE ALERT: Last 20 win rate {recent_win_rate:.1%}")

    print(f"\n{'='*70}")
    print("FORWARD METRICS")
    print(f"{'='*70}")
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 45)
    print(f"{'Total Trades':<30} {total_trades:>15}")
    print(f"{'Win Rate':<30} {win_rate:>14.1%}")
    print(f"{'Avg Return per Trade':<30} {avg_return:>14.2f}%")
    print(f"{'Sharpe Ratio (approx)':<30} {sharpe:>15.2f}")
    print(f"{'Max Drawdown':<30} {max_drawdown*100:>14.2f}%")
    print(f"{'Cumulative Return':<30} {(cumulative_returns.iloc[-1]-1)*100:>14.2f}%")
    print(f"\n{'Recent 20 Trades:':<30}")
    print(f"{'  Win Rate':<30} {recent_win_rate:>14.1%}")
    print(f"{'  Avg Return':<30} {recent_avg_return:>14.2f}%")

    if alerts:
        print(f"\n{'='*70}")
        print("⚠️  ALERTS")
        print(f"{'='*70}")
        for alert in alerts:
            print(f"\n{alert}")
        print("\nConsider reviewing strategy performance and market conditions.")
    else:
        print(f"\n✓ All metrics within acceptable ranges.")


def show_summary():
    """Display current paper trading summary."""
    print("=" * 70)
    print("PAPER TRADING SUMMARY")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    setup_directories()

    # Load data
    predictions = pd.read_csv(PREDICTIONS_FILE) if PREDICTIONS_FILE.exists() else pd.DataFrame()
    outcomes = pd.read_csv(OUTCOMES_FILE) if OUTCOMES_FILE.exists() else pd.DataFrame()

    print(f"\n{'Status':<30} {'Count':>10}")
    print("-" * 40)
    print(f"{'Total Predictions Logged':<30} {len(predictions):>10}")

    if not predictions.empty:
        checked = predictions[predictions["outcome_checked"] == True]
        pending = predictions[predictions["outcome_checked"] == False]
        print(f"{'  - Outcomes Checked':<30} {len(checked):>10}")
        print(f"{'  - Pending':<30} {len(pending):>10}")

    print(f"{'Total Outcomes Recorded':<30} {len(outcomes):>10}")

    if not outcomes.empty:
        winners = outcomes[outcomes["profitable"] == True]
        print(f"{'  - Winners':<30} {len(winners):>10}")
        print(f"{'  - Losers':<30} {len(outcomes) - len(winners):>10}")

    # Show metrics if available
    if METRICS_FILE.exists():
        print("\n")
        update_metrics()

    # Show pending predictions
    if not predictions.empty:
        pending = predictions[predictions["outcome_checked"] == False]
        if not pending.empty:
            print(f"\n{'='*70}")
            print("PENDING PREDICTIONS")
            print(f"{'='*70}")

            pending_sorted = pending.sort_values("expected_exit_date")
            print(f"\n{'Ticker':<8} {'Signal':<6} {'Entry Date':<12} {'Exit Date':<12} {'Confidence':>10}")
            print("-" * 55)

            for _, row in pending_sorted.head(15).iterrows():
                conf_str = f"{row['confidence']:.1%}" if pd.notna(row['confidence']) else "N/A"
                print(f"{row['ticker']:<8} {row['signal']:<6} {row['prediction_date']:<12} {row['expected_exit_date']:<12} {conf_str:>10}")

            if len(pending_sorted) > 15:
                print(f"... and {len(pending_sorted) - 15} more")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m futures.scripts.paper_trade              # Log today's predictions
  python -m futures.scripts.paper_trade --check      # Check outcomes
  python -m futures.scripts.paper_trade --summary    # View summary
  python -m futures.scripts.paper_trade --full       # Full daily workflow
        """
    )
    parser.add_argument(
        "--check", "--check-outcomes",
        action="store_true",
        dest="check",
        help="Check outcomes for predictions past their holding period",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show current paper trading summary",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full daily workflow (log predictions + check outcomes)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for signals (default: {DEFAULT_CONFIDENCE_THRESHOLD})",
    )

    args = parser.parse_args()

    if args.summary:
        show_summary()
    elif args.check:
        check_outcomes()
    elif args.full:
        check_outcomes()
        print("\n")
        log_predictions(confidence_threshold=args.confidence)
        print("\n")
        show_summary()
    else:
        log_predictions(confidence_threshold=args.confidence)


if __name__ == "__main__":
    main()
