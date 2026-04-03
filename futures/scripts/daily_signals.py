"""
Daily Trading Signal Generator

Run this script daily (after market close) to get:
1. New entry signals for tomorrow
2. Exit signals for current positions (time-based or signal-based)
3. Clear action list with exit criteria

Usage:
    python -m futures.scripts.daily_signals

Position tracking is stored in positions.json in the current directory.
"""

import json
import pickle
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from futures.config import get_default_universe
from futures.data.fetcher import DataManager
from futures.strategies import MetalabelingStrategy
from futures.metalabeling import PrimarySignalGenerator, MetaFeatureEngineering


# ============================================================================
# CONFIGURATION - Edit these as needed
# ============================================================================
POSITIONS_FILE = "positions.json"
CONFIDENCE_THRESHOLD = 0.52
MAX_HOLDING_DAYS = 5
MAX_POSITIONS = 10
POSITION_SIZE_PCT = 10  # % of portfolio per position
STOP_LOSS_PCT = 5.0  # Stop loss threshold as % below entry price


def load_model():
    """Load the active trained model from registry."""
    from futures.models.registry import ModelRegistry

    registry = ModelRegistry()
    return registry.load_active()


def load_positions(filepath: str = POSITIONS_FILE) -> dict:
    """
    Load current positions from JSON file.

    Format:
    {
        "AAPL": {"entry_date": "2024-01-15", "entry_price": 185.50, "shares": 54},
        "MSFT": {"entry_date": "2024-01-16", "entry_price": 390.25, "shares": 25},
    }
    """
    path = Path(filepath)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_positions(positions: dict, filepath: str = POSITIONS_FILE):
    """Save positions to JSON file."""
    with open(filepath, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def calculate_exit_date(entry_date_str: str, holding_days: int) -> date:
    """Calculate the exit date based on entry and holding period."""
    entry = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
    return entry + timedelta(days=holding_days)


def calculate_stop_loss_price(entry_price: float, stop_loss_pct: float) -> float:
    """Calculate the stop loss price based on entry price and stop loss percentage."""
    return entry_price * (1 - stop_loss_pct / 100)


def main():
    today = date.today()

    print("=" * 70)
    print(f"DAILY TRADING SIGNALS - {today}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Load model and configuration
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading model...")
    try:
        model, model_info = load_model()
        feature_names = model_info.get("feature_names", [])
        print(f"  Model loaded (trained: {model_info.get('training_date', 'unknown')})")
    except FileNotFoundError:
        print(f"  ERROR: Model not found at {MODEL_PATH}")
        print("  Run training first: python -m futures.scripts.train_metalabeling")
        return

    # -------------------------------------------------------------------------
    # 2. Load current positions
    # -------------------------------------------------------------------------
    print("\n[2/5] Loading current positions...")
    positions = load_positions()
    if positions:
        print(f"  Found {len(positions)} open position(s):")
        for ticker, pos in positions.items():
            days_held = (today - datetime.strptime(pos['entry_date'], "%Y-%m-%d").date()).days
            print(f"    {ticker}: {pos['shares']} shares @ ${pos['entry_price']:.2f} "
                  f"(held {days_held} days)")
    else:
        print("  No open positions")

    # -------------------------------------------------------------------------
    # 3. Fetch latest market data
    # -------------------------------------------------------------------------
    print("\n[3/5] Fetching market data...")
    universe = get_default_universe()
    dm = DataManager()

    # Fetch with refresh to get latest data
    data = dm.get_multi(
        universe.all_tickers,
        refresh=True,  # Get fresh data
        show_progress=True,
    )
    print(f"  Loaded {len(data)} tickers")

    # Check data freshness
    latest_dates = {ticker: df.index[-1] for ticker, df in data.items() if len(df) > 0}
    most_recent = max(latest_dates.values()) if latest_dates else None
    if most_recent:
        print(f"  Most recent data: {most_recent.date()}")

    # -------------------------------------------------------------------------
    # 4. Create strategy and generate signals
    # -------------------------------------------------------------------------
    print("\n[4/5] Generating signals...")
    signal_gen = PrimarySignalGenerator()
    feature_eng = MetaFeatureEngineering(context_tickers=universe.context)

    strategy = MetalabelingStrategy(
        meta_model=model,
        signal_generator=signal_gen,
        feature_engineering=feature_eng,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        context_tickers=universe.context,
        feature_names=feature_names,
    )

    # Get signal details for all tickers
    signal_details = strategy.get_signal_details(data)

    # -------------------------------------------------------------------------
    # 5. Generate action list
    # -------------------------------------------------------------------------
    print("\n[5/5] Analyzing actions...")

    exits = []
    entries = []
    holds = []

    # Check exits for current positions
    for ticker, pos in positions.items():
        entry_date = datetime.strptime(pos['entry_date'], "%Y-%m-%d").date()
        days_held = (today - entry_date).days
        target_exit_date = calculate_exit_date(pos['entry_date'], MAX_HOLDING_DAYS)

        # Get current price
        current_price = data[ticker]["close"].iloc[-1] if ticker in data else None
        pnl = None
        pnl_pct = None
        if current_price:
            pnl = (current_price - pos['entry_price']) * pos['shares']
            pnl_pct = ((current_price / pos['entry_price']) - 1) * 100

        stop_loss_price = calculate_stop_loss_price(pos['entry_price'], STOP_LOSS_PCT)
        stop_loss_triggered = current_price is not None and current_price <= stop_loss_price

        exit_info = {
            "ticker": ticker,
            "entry_date": pos['entry_date'],
            "entry_price": pos['entry_price'],
            "shares": pos['shares'],
            "days_held": days_held,
            "current_price": current_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "target_exit_date": str(target_exit_date),
            "stop_loss_price": stop_loss_price,
        }

        # Check if stop loss triggered (highest priority - protect capital)
        if stop_loss_triggered:
            exit_info["exit_reason"] = "STOP_LOSS"
            exit_info["action"] = "EXIT"
            exits.append(exit_info)
        # Check if time-based exit
        elif days_held >= MAX_HOLDING_DAYS:
            exit_info["exit_reason"] = "TIME_BASED"
            exit_info["action"] = "EXIT"
            exits.append(exit_info)
        # Check if signal-based exit (SELL signal)
        elif ticker in signal_details:
            detail = signal_details[ticker]
            if detail.get("final_signal") == "SELL":
                exit_info["exit_reason"] = "SELL_SIGNAL"
                exit_info["action"] = "EXIT"
                exits.append(exit_info)
            else:
                exit_info["action"] = "HOLD"
                exit_info["days_until_exit"] = MAX_HOLDING_DAYS - days_held
                holds.append(exit_info)
        else:
            exit_info["action"] = "HOLD"
            exit_info["days_until_exit"] = MAX_HOLDING_DAYS - days_held
            holds.append(exit_info)

    # Check for new entry signals
    available_slots = MAX_POSITIONS - len(positions) + len(exits)  # Account for exits

    for ticker, detail in signal_details.items():
        # Skip if already in position
        if ticker in positions:
            continue
        # Skip context tickers
        if ticker in universe.context:
            continue
        # Check for BUY signal
        if detail.get("final_signal") == "BUY":
            current_price = data[ticker]["close"].iloc[-1] if ticker in data else None
            stop_loss_price = calculate_stop_loss_price(current_price, STOP_LOSS_PCT) if current_price else None
            entries.append({
                "ticker": ticker,
                "action": "BUY",
                "signal_source": detail.get("source_indicators", []),
                "confidence": detail.get("confidence"),
                "current_price": current_price,
                "stop_loss_price": stop_loss_price,
                "exit_criteria": f"Time-based: {MAX_HOLDING_DAYS} days, SELL signal, or stop loss (-{STOP_LOSS_PCT}%)",
                "target_exit_date": str(today + timedelta(days=MAX_HOLDING_DAYS)),
            })

    # Sort entries by confidence
    entries.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    # Limit to available slots
    entries = entries[:available_slots]

    # =========================================================================
    # OUTPUT RESULTS
    # =========================================================================

    print("\n" + "=" * 70)
    print("TODAY'S ACTIONS")
    print("=" * 70)

    # EXITS
    print(f"\n{'EXIT POSITIONS':=^70}")
    if exits:
        for e in exits:
            pnl_str = f"${e['pnl']:+,.2f} ({e['pnl_pct']:+.1f}%)" if e['pnl'] else "N/A"
            reason_detail = ""
            if e['exit_reason'] == "STOP_LOSS":
                reason_detail = f" (triggered at ${e['stop_loss_price']:.2f})"
            print(f"""
  {e['ticker']} - {e['exit_reason']}{reason_detail}
    Entry: {e['entry_date']} @ ${e['entry_price']:.2f}
    Current: ${e['current_price']:.2f} (held {e['days_held']} days)
    Stop loss: ${e['stop_loss_price']:.2f}
    P&L: {pnl_str}
    ACTION: SELL {e['shares']} shares
""")
    else:
        print("\n  No exits today\n")

    # NEW ENTRIES
    print(f"{'NEW ENTRIES':=^70}")
    if entries:
        for i, e in enumerate(entries, 1):
            stop_loss_str = f"${e['stop_loss_price']:.2f}" if e['stop_loss_price'] else "N/A"
            print(f"""
  {i}. {e['ticker']} - CONFIDENCE: {e['confidence']:.1%}
     Signal: {', '.join(e['signal_source'])}
     Price: ${e['current_price']:.2f}
     Stop loss: {stop_loss_str} (-{STOP_LOSS_PCT}%)
     Exit criteria: {e['exit_criteria']}
     Target exit: {e['target_exit_date']}
     ACTION: BUY (allocate {POSITION_SIZE_PCT}% of portfolio)
""")
    else:
        print("\n  No new entries today\n")

    # HOLD POSITIONS
    print(f"{'HOLD POSITIONS':=^70}")
    if holds:
        for h in holds:
            pnl_str = f"${h['pnl']:+,.2f} ({h['pnl_pct']:+.1f}%)" if h['pnl'] else "N/A"
            print(f"""
  {h['ticker']}
    Entry: {h['entry_date']} @ ${h['entry_price']:.2f}
    Current: ${h['current_price']:.2f} (held {h['days_held']} days)
    Stop loss: ${h['stop_loss_price']:.2f} (-{STOP_LOSS_PCT}%)
    P&L: {pnl_str}
    Days until time-exit: {h['days_until_exit']}
    ACTION: HOLD
""")
    else:
        print("\n  No positions to hold\n")

    # SUMMARY
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  Exits:   {len(exits)} position(s) to close
  Entries: {len(entries)} new position(s) to open
  Holds:   {len(holds)} position(s) to maintain

  Open slots after actions: {MAX_POSITIONS - len(positions) + len(exits) - len(entries)}
""")

    # =========================================================================
    # OPTIONAL: Update positions file
    # =========================================================================
    print("=" * 70)
    print("POSITION TRACKING")
    print("=" * 70)

    if exits or entries:
        print(f"\n  To update positions after executing trades, run:")
        print(f"    python -m futures.scripts.daily_signals --update")
        print(f"\n  Or manually edit: {POSITIONS_FILE}")

        # Show what the new positions would look like
        new_positions = {k: v for k, v in positions.items() if k not in [e['ticker'] for e in exits]}
        for e in entries:
            new_positions[e['ticker']] = {
                "entry_date": str(today),
                "entry_price": e['current_price'],
                "shares": "TBD",  # User fills in actual shares
            }

        print(f"\n  New positions.json would be:")
        print(json.dumps(new_positions, indent=2, default=str))

    # Check for --update flag
    import sys
    if "--update" in sys.argv:
        print("\n  Updating positions file...")
        # Remove exited positions
        for e in exits:
            if e['ticker'] in positions:
                del positions[e['ticker']]
        # Add new entries (with placeholder shares)
        for e in entries:
            positions[e['ticker']] = {
                "entry_date": str(today),
                "entry_price": float(e['current_price']),
                "shares": 0,  # User should update with actual shares
            }
        save_positions(positions)
        print(f"  Saved to {POSITIONS_FILE}")
        print("  NOTE: Update 'shares' for new positions with actual quantities!")

    print("\n" + "=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return {
        "exits": exits,
        "entries": entries,
        "holds": holds,
    }


if __name__ == "__main__":
    main()
