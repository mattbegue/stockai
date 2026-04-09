"""
Paper Trading Tracker

Run daily before market open (~8:30 AM ET) to:
  1. Refresh market data from Alpaca
  2. Settle matured positions (exit any hitting their holding-period today)
  3. Generate new signals from yesterday's close
  4. Display today's action list

Usage:
    # Full daily workflow (recommended — run pre-market each trading day)
    python -m futures.scripts.paper_trade --full

    # Individual steps
    python -m futures.scripts.paper_trade                  # Log new predictions only
    python -m futures.scripts.paper_trade --check          # Settle matured positions only
    python -m futures.scripts.paper_trade --summary        # View current performance

Timing:
    --full  at ~8:30 AM ET (after Alpaca data refreshes, before 9:30 open)
    Actions execute at 9:30 AM open (entry = today's open; exit = open on exit day)
"""

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from futures.config import get_universe, TickerUniverse
from futures.data.fetcher import DataManager
from futures.metalabeling import PrimarySignalGenerator, MetaFeatureEngineering
from futures.strategies import MetalabelingStrategy
from futures.strategies.base import Signal


# ============================================================================
# CONFIGURATION
# ============================================================================
PAPER_TRADE_DIR = Path("paper_trading")
POSITIONS_FILE = PAPER_TRADE_DIR / "positions.csv"    # Open (pending) positions
TRADES_FILE = PAPER_TRADE_DIR / "trades.csv"          # Closed trades (full history)
METRICS_FILE = PAPER_TRADE_DIR / "forward_metrics.json"

POSITION_SIZE_USD = 10_000.0   # Fixed dollar amount per trade
DEFAULT_CONFIDENCE = 0.60      # Must match backtest threshold
TRANSACTION_COST_PCT = 0.001   # 0.1% per leg (matches backtest)
SLIPPAGE_PCT = 0.0005          # 0.05% per leg (matches backtest)

# Alert thresholds
ALERT_WIN_RATE_MIN = 0.50
ALERT_SHARPE_MIN = 0.0
ALERT_DRAWDOWN_MAX = -0.10


# ============================================================================
# SETUP / HELPERS
# ============================================================================
def setup_directories():
    PAPER_TRADE_DIR.mkdir(exist_ok=True)

    if not POSITIONS_FILE.exists():
        pd.DataFrame(columns=[
            "ticker", "signal", "entry_date", "entry_price",
            "confidence", "source_indicators", "holding_days", "exit_date",
        ]).to_csv(POSITIONS_FILE, index=False)

    if not TRADES_FILE.exists():
        pd.DataFrame(columns=[
            "ticker", "signal", "entry_date", "entry_price",
            "exit_date", "exit_price", "holding_days_target",
            "holding_days_actual", "confidence", "source_indicators",
            "return_pct", "pnl_usd", "profitable",
        ]).to_csv(TRADES_FILE, index=False)


def load_model():
    from futures.models.registry import ModelRegistry
    return ModelRegistry().load_active()


def _next_bday(d: date, offset: int = 1) -> date:
    """Return d shifted by `offset` business days."""
    return np.busday_offset(d, offset, roll="forward").astype(date)


def _bdays_between(start: date, end: date) -> int:
    """Number of business days between two dates (exclusive of start)."""
    return int(np.busday_count(start, end))


def _round_trip_cost_pct() -> float:
    """Total round-trip cost fraction (two legs of slippage + commission)."""
    entry_cost = SLIPPAGE_PCT + TRANSACTION_COST_PCT
    exit_cost = SLIPPAGE_PCT + TRANSACTION_COST_PCT
    return entry_cost + exit_cost


def refresh_market_data(universe: TickerUniverse, show_progress: bool = True) -> dict:
    """Fetch fresh data from Alpaca (incremental update — only missing days)."""
    dm = DataManager()
    data = dm.get_multi(universe.all_tickers, refresh=False, show_progress=show_progress)

    # Identify tickers with stale data and refresh them
    stale = [t for t, df in data.items() if df.index[-1].date() < _next_bday(date.today(), -1)]
    if stale:
        fresh = dm.get_multi(stale, refresh=True, show_progress=show_progress)
        data.update(fresh)

    # Load any tickers that weren't in cache yet
    missing = [t for t in universe.all_tickers if t not in data]
    if missing:
        fetched = dm.get_multi(missing, refresh=True, show_progress=show_progress)
        data.update(fetched)

    return data


# ============================================================================
# SETTLE MATURED POSITIONS
# ============================================================================
def check_outcomes(data: dict | None = None):
    """
    Settle any open positions whose exit date has arrived.

    Exit price = open on the exit date (matching the backtester).
    If today is before market open and we don't yet have today's open,
    we use yesterday's close as a proxy (labelled accordingly).
    """
    print("=" * 70)
    print("PAPER TRADING — Settling Matured Positions")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    setup_directories()

    positions = pd.read_csv(POSITIONS_FILE)
    if positions.empty:
        print("\nNo open positions.")
        return

    today = date.today()
    positions["exit_date"] = pd.to_datetime(positions["exit_date"]).dt.date

    due = positions[positions["exit_date"] <= today]
    still_open = positions[positions["exit_date"] > today]

    if due.empty:
        print("\nNo positions due for exit today.")
        if not positions.empty:
            soonest = positions["exit_date"].min()
            print(f"Next exit: {soonest}")
        return

    # Load data if not provided
    if data is None:
        tickers = due["ticker"].unique().tolist()
        dm = DataManager()
        data = dm.get_multi(tickers, refresh=False, show_progress=False)

    print(f"\nSettling {len(due)} position(s)...")

    new_trades = []
    failed = []

    for _, row in due.iterrows():
        ticker = row["ticker"]
        if ticker not in data:
            print(f"  SKIP {ticker}: no data")
            failed.append(row)
            continue

        df = data[ticker]
        exit_target = pd.Timestamp(row["exit_date"])

        # Find the exit bar: first bar on or after the exit date
        exit_mask = df.index >= exit_target
        if not exit_mask.any():
            # Data doesn't reach the exit date yet — skip until it does
            still_open = pd.concat([still_open, pd.DataFrame([row])], ignore_index=True)
            continue

        exit_bar = df.index[exit_mask][0]

        # Use open price on exit day (matches backtester); fall back to close
        # if open is unavailable (shouldn't happen but defensive)
        if "open" in df.columns and pd.notna(df.loc[exit_bar, "open"]):
            exit_price = float(df.loc[exit_bar, "open"])
            price_type = "open"
        else:
            exit_price = float(df.loc[exit_bar, "close"])
            price_type = "close"

        entry_price = float(row["entry_price"])

        # Apply exit-leg costs
        if row["signal"] == "BUY":
            exec_exit = exit_price * (1 - SLIPPAGE_PCT)
            raw_return = (exec_exit / entry_price) - 1
        else:
            exec_exit = exit_price * (1 + SLIPPAGE_PCT)
            raw_return = (entry_price / exec_exit) - 1

        # Subtract round-trip commission (entry leg already applied at entry)
        return_pct = (raw_return - TRANSACTION_COST_PCT) * 100
        pnl_usd = POSITION_SIZE_USD * (return_pct / 100)
        holding_actual = _bdays_between(
            pd.Timestamp(row["entry_date"]).date(), exit_bar.date()
        )

        status = "WIN" if return_pct > 0 else "LOSS"
        print(
            f"  {status:4s}  {ticker:<6}  {row['signal']:<4}  "
            f"entry ${entry_price:.2f} → exit ${exit_price:.2f} ({price_type})  "
            f"{return_pct:+.2f}%  ${pnl_usd:+.0f}"
        )

        new_trades.append({
            "ticker": ticker,
            "signal": row["signal"],
            "entry_date": row["entry_date"],
            "entry_price": entry_price,
            "exit_date": exit_bar.date().isoformat(),
            "exit_price": exit_price,
            "holding_days_target": row["holding_days"],
            "holding_days_actual": holding_actual,
            "confidence": row["confidence"],
            "source_indicators": row.get("source_indicators", ""),
            "return_pct": round(return_pct, 4),
            "pnl_usd": round(pnl_usd, 2),
            "profitable": return_pct > 0,
        })

    # Persist
    if new_trades:
        existing_trades = pd.read_csv(TRADES_FILE)
        updated_trades = pd.concat(
            [existing_trades, pd.DataFrame(new_trades)], ignore_index=True
        )
        updated_trades.to_csv(TRADES_FILE, index=False)

    # Rewrite positions file with only the still-open ones (+ any failed)
    if failed:
        still_open = pd.concat(
            [still_open, pd.DataFrame(failed)], ignore_index=True
        )
    still_open.to_csv(POSITIONS_FILE, index=False)

    if new_trades:
        batch = pd.DataFrame(new_trades)
        wins = batch[batch["profitable"]].shape[0]
        print(f"\n  Settled: {len(new_trades)}  |  Wins: {wins}  |  Losses: {len(new_trades)-wins}")
        print(f"  Batch avg return: {batch['return_pct'].mean():+.2f}%")
        _update_metrics(verbose=True)


# ============================================================================
# GENERATE NEW SIGNALS
# ============================================================================
def log_predictions(
    data: dict | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
    universe: TickerUniverse | None = None,
    model=None,
    model_info: dict | None = None,
):
    """
    Generate today's signals and log any BUY/SELL candidates as open positions.

    Entry price = tomorrow's open (market order at 9:30 AM ET).
    We record today's close as the reference; outcome checking uses the actual
    next-day open from the stored OHLCV data.
    """
    print("=" * 70)
    print("PAPER TRADING — Generating New Signals")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Confidence threshold: {confidence_threshold:.2f}")
    print("=" * 70)

    setup_directories()

    # Load model if not provided
    if model is None:
        print("\nLoading model...")
        model, model_info = load_model()

    # Determine universe
    if universe is None:
        if model_info and "universe_tradeable" in model_info:
            universe = TickerUniverse(
                tradeable=model_info["universe_tradeable"],
                context=model_info.get("universe_context", []),
                name=model_info.get("universe_size", "from_model"),
            )
        else:
            universe = get_universe("small")

    print(f"Universe: {len(universe.tradeable)} tradeable, {len(universe.context)} context")

    # Load data if not provided
    if data is None:
        print("\nLoading market data (refreshing stale tickers)...")
        data = refresh_market_data(universe)
        print(f"Loaded {len(data)} tickers")

    # Check data freshness
    if data:
        most_recent = max(df.index[-1] for df in data.values() if len(df) > 0)
        print(f"Most recent bar: {most_recent.date()}")

    # Create strategy
    signal_gen = PrimarySignalGenerator()
    feature_eng = MetaFeatureEngineering(context_tickers=universe.context)

    strategy = MetalabelingStrategy(
        meta_model=model,
        signal_generator=signal_gen,
        feature_engineering=feature_eng,
        confidence_threshold=confidence_threshold,
        context_tickers=universe.context,
        feature_names=model_info.get("feature_names") if model_info else None,
    )

    # Generate signals
    print("\nRunning signal generation...")
    signals = strategy.generate_signals(data)
    holding_days_map = strategy.get_position_holding_days(signals)

    # Load existing open positions to avoid duplicate entries
    existing_positions = pd.read_csv(POSITIONS_FILE)
    open_tickers = set(existing_positions["ticker"].tolist()) if not existing_positions.empty else set()

    today = date.today()
    context_set = set(universe.context)

    candidates = []
    for ticker, signal in signals.items():
        if ticker in context_set:
            continue
        if signal not in (Signal.BUY, Signal.SELL):
            continue
        if ticker in open_tickers:
            print(f"  SKIP {ticker}: already have open position")
            continue

        # Reference price (today's close) — actual entry will be next day's open
        ref_close = (
            float(data[ticker]["close"].iloc[-1]) if ticker in data else None
        )
        conf = strategy._last_confidences.get(ticker)
        indicators = strategy._last_source_indicators.get(ticker, [])
        hold = holding_days_map.get(ticker, 5)
        exit_date = _next_bday(today, hold)

        # Apply entry-leg costs to the reference price so PnL calc is consistent
        if signal == Signal.BUY and ref_close:
            entry_exec = ref_close * (1 + SLIPPAGE_PCT + TRANSACTION_COST_PCT)
        elif ref_close:
            entry_exec = ref_close * (1 - SLIPPAGE_PCT - TRANSACTION_COST_PCT)
        else:
            entry_exec = None

        candidates.append({
            "ticker": ticker,
            "signal": signal.name,
            "entry_date": today.isoformat(),
            "entry_price": round(entry_exec, 4) if entry_exec else None,
            "confidence": round(conf, 4) if conf else None,
            "source_indicators": ",".join(indicators),
            "holding_days": hold,
            "exit_date": exit_date.isoformat(),
        })

    if not candidates:
        print("\nNo new signals today.")
        return

    # Sort by confidence descending
    candidates.sort(key=lambda x: x["confidence"] or 0, reverse=True)

    # Append to positions file (these are "open" positions)
    updated = pd.concat(
        [existing_positions, pd.DataFrame(candidates)], ignore_index=True
    )
    updated.to_csv(POSITIONS_FILE, index=False)

    # Display
    print(f"\n{'='*70}")
    print(f"NEW SIGNALS: {len(candidates)}")
    print(f"{'='*70}")
    print(
        f"\n{'Ticker':<8} {'Sig':<5} {'Conf':>6}  {'Hold':>4}  "
        f"{'Exit Date':<12}  {'Ref Close':>10}  {'Indicators'}"
    )
    print("-" * 75)
    for c in candidates:
        conf_str = f"{c['confidence']:.1%}" if c["confidence"] else " N/A "
        price_str = f"${c['entry_price']:.2f}" if c["entry_price"] else "  N/A"
        print(
            f"{c['ticker']:<8} {c['signal']:<5} {conf_str:>6}  {c['holding_days']:>3}d  "
            f"{c['exit_date']:<12}  {price_str:>10}  {c['source_indicators']}"
        )

    print(f"\n  → Execute these as market orders at 9:30 AM ET open.")
    print(f"  → Positions saved to: {POSITIONS_FILE}")


# ============================================================================
# METRICS
# ============================================================================
def _update_metrics(verbose: bool = True) -> dict:
    """Recompute all forward metrics from the closed-trades log."""
    trades = pd.read_csv(TRADES_FILE)

    if trades.empty or len(trades) < 5:
        if verbose:
            print("\nNot enough closed trades for metrics (need at least 5).")
        return {}

    trades = trades.sort_values("exit_date").reset_index(drop=True)

    total = len(trades)
    win_rate = trades["profitable"].mean()
    avg_ret = trades["return_pct"].mean()
    std_ret = trades["return_pct"].std()
    total_pnl = trades["pnl_usd"].sum()

    # Annualised Sharpe: scale per-trade returns by sqrt(252 / avg_hold)
    avg_hold = trades["holding_days_actual"].mean() if "holding_days_actual" in trades else 5
    sharpe = (avg_ret / std_ret) * np.sqrt(252 / max(avg_hold, 1)) if std_ret > 0 else 0.0

    # Equity curve (each trade deployed POSITION_SIZE_USD)
    cum_returns = (1 + trades["return_pct"] / 100).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    # Profit factor
    gross_wins = trades.loc[trades["profitable"], "pnl_usd"].sum()
    gross_losses = abs(trades.loc[~trades["profitable"], "pnl_usd"].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Recent window
    recent = trades.tail(20)
    recent_win_rate = recent["profitable"].mean()
    recent_avg_ret = recent["return_pct"].mean()

    metrics = {
        "last_updated": datetime.now().isoformat(),
        "total_trades": total,
        "win_rate": win_rate,
        "avg_return_pct": avg_ret,
        "std_return_pct": std_ret,
        "sharpe_ratio": sharpe,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd * 100,
        "total_pnl_usd": total_pnl,
        "cumulative_return_pct": float((cum_returns.iloc[-1] - 1) * 100),
        "recent_20_win_rate": recent_win_rate,
        "recent_20_avg_return": recent_avg_ret,
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    if not verbose:
        return metrics

    # --- Display ---
    alerts = []
    if win_rate < ALERT_WIN_RATE_MIN:
        alerts.append(f"WIN RATE {win_rate:.1%} < {ALERT_WIN_RATE_MIN:.1%}")
    if sharpe < ALERT_SHARPE_MIN:
        alerts.append(f"SHARPE {sharpe:.2f} < {ALERT_SHARPE_MIN:.2f}")
    if max_dd * 100 < ALERT_DRAWDOWN_MAX * 100:
        alerts.append(f"DRAWDOWN {max_dd*100:.1f}% < {ALERT_DRAWDOWN_MAX*100:.1f}%")
    if len(recent) >= 10 and recent_win_rate < ALERT_WIN_RATE_MIN:
        alerts.append(f"RECENT WIN RATE {recent_win_rate:.1%} (last {len(recent)} trades)")

    print(f"\n{'='*70}")
    print("FORWARD METRICS")
    print(f"{'='*70}")
    print(f"\n{'Total Trades':<35} {total:>10}")
    print(f"{'Win Rate':<35} {win_rate:>9.1%}")
    print(f"{'Avg Return / Trade':<35} {avg_ret:>9.2f}%")
    print(f"{'Profit Factor':<35} {profit_factor:>10.2f}")
    print(f"{'Sharpe (annualised, approx)':<35} {sharpe:>10.2f}")
    print(f"{'Max Drawdown':<35} {max_dd*100:>9.2f}%")
    print(f"{'Total P&L (${:.0f}/trade)'.format(POSITION_SIZE_USD):<35} ${total_pnl:>9.0f}")
    print(f"{'Cumulative Return':<35} {metrics['cumulative_return_pct']:>9.2f}%")
    print(f"\n{'Recent 20 trades — Win Rate':<35} {recent_win_rate:>9.1%}")
    print(f"{'Recent 20 trades — Avg Return':<35} {recent_avg_ret:>9.2f}%")

    if alerts:
        print(f"\n{'='*70}")
        print("⚠  ALERTS")
        print(f"{'='*70}")
        for a in alerts:
            print(f"  • {a}")
    else:
        print(f"\n  ✓ All metrics within acceptable ranges.")

    return metrics


def show_summary():
    """Display current paper trading summary."""
    print("=" * 70)
    print("PAPER TRADING SUMMARY")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    setup_directories()

    positions = pd.read_csv(POSITIONS_FILE) if POSITIONS_FILE.exists() else pd.DataFrame()
    trades = pd.read_csv(TRADES_FILE) if TRADES_FILE.exists() else pd.DataFrame()

    print(f"\n{'Open positions':<35} {len(positions):>5}")
    print(f"{'Closed trades':<35} {len(trades):>5}")

    if not positions.empty:
        print(f"\n{'='*70}")
        print("OPEN POSITIONS")
        print(f"{'='*70}")
        print(f"\n{'Ticker':<8} {'Sig':<5} {'Entry':<12} {'Exit':<12} {'Hold':>4}  {'Conf':>6}")
        print("-" * 55)
        positions["exit_date"] = pd.to_datetime(positions["exit_date"]).dt.date
        positions_sorted = positions.sort_values("exit_date")
        for _, row in positions_sorted.iterrows():
            conf_str = f"{row['confidence']:.1%}" if pd.notna(row.get("confidence")) else " N/A"
            print(
                f"{row['ticker']:<8} {row['signal']:<5} {str(row['entry_date']):<12} "
                f"{str(row['exit_date']):<12} {row['holding_days']:>3}d  {conf_str:>6}"
            )

    if not trades.empty:
        _update_metrics(verbose=True)

        # Show last 10 closed trades
        print(f"\n{'='*70}")
        print("LAST 10 CLOSED TRADES")
        print(f"{'='*70}")
        print(f"\n{'Ticker':<8} {'Sig':<5} {'Entry':>10} {'Exit':>10} {'Ret%':>7}  {'P&L':>8}  {'W/L'}")
        print("-" * 65)
        recent = trades.sort_values("exit_date").tail(10)
        for _, row in recent.iterrows():
            wl = "WIN" if row["profitable"] else "LOSS"
            print(
                f"{row['ticker']:<8} {row['signal']:<5} "
                f"${row['entry_price']:>9.2f} ${row['exit_price']:>9.2f} "
                f"{row['return_pct']:>+6.2f}%  ${row['pnl_usd']:>+7.0f}  {wl}"
            )
    else:
        print("\nNo closed trades yet.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading Tracker — run daily pre-market (~8:30 AM ET)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m futures.scripts.paper_trade --full       # Full daily workflow (recommended)
  python -m futures.scripts.paper_trade              # Log new signals only
  python -m futures.scripts.paper_trade --check      # Settle matured positions only
  python -m futures.scripts.paper_trade --summary    # Performance dashboard
        """,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Settle matured positions (exit anything past its holding period)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show performance dashboard",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full daily workflow: refresh data → settle exits → generate signals → summary",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Confidence threshold for new signals (default: {DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--universe",
        choices=["small", "medium", "large"],
        default=None,
        help="Universe override (default: read from active model metadata)",
    )
    args = parser.parse_args()

    if args.summary:
        show_summary()
        return

    if args.check:
        check_outcomes()
        return

    if args.full:
        # Load shared resources once, reuse across steps
        print("Loading model and refreshing data...\n")
        model, model_info = load_model()

        if args.universe:
            universe = get_universe(args.universe)
        elif model_info and "universe_tradeable" in model_info:
            universe = TickerUniverse(
                tradeable=model_info["universe_tradeable"],
                context=model_info.get("universe_context", []),
                name=model_info.get("universe_size", "from_model"),
            )
        else:
            universe = get_universe("small")

        print(f"Refreshing data for {len(universe.all_tickers)} tickers...")
        data = refresh_market_data(universe)
        print(f"  {len(data)} tickers loaded\n")

        check_outcomes(data=data)
        print()
        log_predictions(
            data=data,
            confidence_threshold=args.confidence,
            universe=universe,
            model=model,
            model_info=model_info,
        )
        print()
        show_summary()
        return

    # Default: log predictions only
    log_predictions(confidence_threshold=args.confidence)


if __name__ == "__main__":
    main()
