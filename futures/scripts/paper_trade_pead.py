"""
PEAD Paper Trading Tracker

Run daily before market open (~8:30 AM ET) to:
  1. Refresh market data from Alpaca
  2. Settle matured positions (exit any hitting their 20-day hold today)
  3. Scan yfinance for qualifying earnings events (EPS surprise >= threshold)
  4. Log new positions and print today's action list

Usage:
    python -m futures.scripts.paper_trade_pead --full       # Full daily workflow
    python -m futures.scripts.paper_trade_pead              # Scan for new signals only
    python -m futures.scripts.paper_trade_pead --check      # Settle exits only
    python -m futures.scripts.paper_trade_pead --summary    # Performance dashboard

Timing:
    Run --full at ~8:30 AM ET (after Alpaca data refreshes, before 9:30 open).
    New positions enter at today's 9:30 AM open.
    Exits execute at the open on the exit date (20 trading days after entry).

Data lag note:
    yfinance sometimes has a 1-day lag on AMC earnings. If a ticker reported
    last night (AMC) but doesn't appear today, it will appear tomorrow and be
    treated as a BMO reporter (1-day late). This matches the backtest's BMO
    handling and is acceptable.
"""

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from futures.config.universes import get_universe
from futures.data.fetcher import DataManager


# ============================================================================
# CONFIGURATION  (must match backtest_pead.py parameters)
# ============================================================================
PAPER_TRADE_DIR = Path("paper_trading_pead")
POSITIONS_FILE = PAPER_TRADE_DIR / "positions.csv"
TRADES_FILE = PAPER_TRADE_DIR / "trades.csv"
METRICS_FILE = PAPER_TRADE_DIR / "forward_metrics.json"

MIN_SURPRISE_PCT = 10.0        # Minimum EPS surprise % to enter
HOLDING_DAYS = 20              # Trading days to hold each position
POSITION_SIZE_USD = 5_000.0   # Fixed dollar amount per trade (5% of $100K)
TRANSACTION_COST_PCT = 0.001   # 0.1% per leg
SLIPPAGE_PCT = 0.0005          # 0.05% per leg

ALERT_WIN_RATE_MIN = 0.50
ALERT_SHARPE_MIN = 0.0
ALERT_DRAWDOWN_MAX = -0.10


# ============================================================================
# HELPERS
# ============================================================================
def setup_directories():
    PAPER_TRADE_DIR.mkdir(exist_ok=True)

    if not POSITIONS_FILE.exists():
        pd.DataFrame(columns=[
            "ticker", "entry_date", "entry_price",
            "surprise_pct", "surprise_bucket", "holding_days", "exit_date",
        ]).to_csv(POSITIONS_FILE, index=False)

    if not TRADES_FILE.exists():
        pd.DataFrame(columns=[
            "ticker", "entry_date", "entry_price",
            "exit_date", "exit_price",
            "surprise_pct", "surprise_bucket",
            "holding_days_target", "holding_days_actual",
            "return_pct", "pnl_usd", "profitable",
        ]).to_csv(TRADES_FILE, index=False)


def _next_bday(d: date, offset: int = 1) -> date:
    return np.busday_offset(d, offset, roll="forward").astype(date)


def _bdays_between(start: date, end: date) -> int:
    return int(np.busday_count(start, end))


def _surprise_bucket(surprise_pct: float) -> str:
    if surprise_pct >= 25.0:
        return "25%+"
    elif surprise_pct >= 15.0:
        return "15-25%"
    else:
        return "10-15%"


def refresh_market_data(universe, show_progress: bool = False) -> dict:
    """Fetch fresh data from Alpaca (incremental — only missing days)."""
    dm = DataManager()
    data = dm.get_multi(universe.all_tickers, refresh=False, show_progress=show_progress)

    stale = [t for t, df in data.items() if df.index[-1].date() < _next_bday(date.today(), -1)]
    if stale:
        fresh = dm.get_multi(stale, refresh=True, show_progress=show_progress)
        data.update(fresh)

    missing = [t for t in universe.all_tickers if t not in data]
    if missing:
        fetched = dm.get_multi(missing, refresh=True, show_progress=show_progress)
        data.update(fetched)

    return data


# ============================================================================
# SCAN EARNINGS (yfinance)
# ============================================================================
def scan_earnings(universe_tickers: list[str], lookback_days: int = 2) -> list[dict]:
    """
    Check yfinance for qualifying earnings events in the last `lookback_days` calendar days.

    Returns a list of dicts: {ticker, report_date, surprise_pct, surprise_bucket}
    sorted by surprise_pct descending.

    Note: yfinance may lag 1 day for AMC reporters. Using lookback_days=2 catches
    both last night's AMC and this morning's BMO reporters.
    """
    cutoff = date.today() - timedelta(days=lookback_days)
    qualifying = []

    tradeable_set = set(universe_tickers)

    for ticker in universe_tickers:
        try:
            df = yf.Ticker(ticker).get_earnings_dates(limit=8)
            if df is None or df.empty:
                continue

            df = df.dropna(subset=["EPS Estimate", "Reported EPS"])

            for idx, row in df.iterrows():
                report_date = idx.date() if hasattr(idx, "date") else idx
                if report_date < cutoff or report_date > date.today():
                    continue

                estimate = float(row["EPS Estimate"])
                actual = float(row["Reported EPS"])

                # Skip when estimate is zero/near-zero (undefined surprise)
                if abs(estimate) < 0.01:
                    continue

                surprise_pct = float(row.get("Surprise(%)", 0))
                # yfinance Surprise(%) is already in percentage form
                if surprise_pct < MIN_SURPRISE_PCT:
                    continue

                qualifying.append({
                    "ticker": ticker,
                    "report_date": report_date,
                    "eps_estimate": round(estimate, 4),
                    "eps_actual": round(actual, 4),
                    "surprise_pct": round(surprise_pct, 2),
                    "surprise_bucket": _surprise_bucket(surprise_pct),
                })

        except Exception:
            continue

    qualifying.sort(key=lambda x: x["surprise_pct"], reverse=True)
    return qualifying


# ============================================================================
# SETTLE MATURED POSITIONS
# ============================================================================
def check_outcomes(data: dict | None = None):
    """Settle any open positions whose 20-day hold has expired."""
    print("=" * 70)
    print("PEAD PAPER TRADING — Settling Matured Positions")
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
            print(f"Next exit: {positions['exit_date'].min()}")
        return

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
            print(f"  SKIP {ticker}: no price data")
            failed.append(row)
            continue

        df = data[ticker]
        exit_target = pd.Timestamp(row["exit_date"])

        exit_mask = df.index >= exit_target
        if not exit_mask.any():
            still_open = pd.concat([still_open, pd.DataFrame([row])], ignore_index=True)
            continue

        exit_bar = df.index[exit_mask][0]

        if "open" in df.columns and pd.notna(df.loc[exit_bar, "open"]):
            exit_price = float(df.loc[exit_bar, "open"])
            price_type = "open"
        else:
            exit_price = float(df.loc[exit_bar, "close"])
            price_type = "close (fallback)"

        entry_price = float(row["entry_price"])

        # Exit-leg costs (entry leg already baked in at entry recording)
        exec_exit = exit_price * (1 - SLIPPAGE_PCT)
        raw_return = (exec_exit / entry_price) - 1
        return_pct = (raw_return - TRANSACTION_COST_PCT) * 100
        pnl_usd = POSITION_SIZE_USD * (return_pct / 100)
        holding_actual = _bdays_between(
            pd.Timestamp(row["entry_date"]).date(), exit_bar.date()
        )

        status = "WIN " if return_pct > 0 else "LOSS"
        print(
            f"  {status}  {ticker:<6}  "
            f"entry ${entry_price:.2f} → exit ${exit_price:.2f} ({price_type})  "
            f"{return_pct:+.2f}%  ${pnl_usd:+.0f}  [{row.get('surprise_bucket', '?')}]"
        )

        new_trades.append({
            "ticker": ticker,
            "entry_date": row["entry_date"],
            "entry_price": entry_price,
            "exit_date": exit_bar.date().isoformat(),
            "exit_price": exit_price,
            "surprise_pct": row.get("surprise_pct"),
            "surprise_bucket": row.get("surprise_bucket"),
            "holding_days_target": row["holding_days"],
            "holding_days_actual": holding_actual,
            "return_pct": round(return_pct, 4),
            "pnl_usd": round(pnl_usd, 2),
            "profitable": return_pct > 0,
        })

    if new_trades:
        existing = pd.read_csv(TRADES_FILE)
        updated = pd.concat([existing, pd.DataFrame(new_trades)], ignore_index=True)
        updated.to_csv(TRADES_FILE, index=False)

    if failed:
        still_open = pd.concat([still_open, pd.DataFrame(failed)], ignore_index=True)
    still_open.to_csv(POSITIONS_FILE, index=False)

    if new_trades:
        batch = pd.DataFrame(new_trades)
        wins = batch[batch["profitable"]].shape[0]
        print(f"\n  Settled: {len(new_trades)}  |  Wins: {wins}  |  Losses: {len(new_trades)-wins}")
        print(f"  Batch avg return: {batch['return_pct'].mean():+.2f}%")
        _update_metrics(verbose=True)


# ============================================================================
# SCAN AND LOG NEW SIGNALS
# ============================================================================
def log_signals(data: dict | None = None, universe=None):
    """
    Scan yfinance for qualifying earnings events and log new PEAD positions.

    Entry price = today's open (execute as market order at 9:30 AM ET).
    We use yesterday's close as a proxy for the entry price until the open
    is available; the actual open is used for P&L when settling.
    """
    print("=" * 70)
    print("PEAD PAPER TRADING — Scanning for New Signals")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Min EPS surprise: {MIN_SURPRISE_PCT:.0f}%   Hold: {HOLDING_DAYS} days")
    print("=" * 70)

    setup_directories()

    if universe is None:
        universe = get_universe("medium")

    if data is None:
        print("\nLoading market data...")
        data = refresh_market_data(universe, show_progress=False)

    print(f"\nScanning {len(universe.tradeable)} tickers for qualifying earnings...")
    candidates = scan_earnings(universe.tradeable, lookback_days=2)

    if not candidates:
        print("No qualifying earnings events found (surprise >= {:.0f}%).".format(MIN_SURPRISE_PCT))
        return

    # Filter out tickers we already have open positions in
    existing_positions = pd.read_csv(POSITIONS_FILE)
    open_tickers = set(existing_positions["ticker"].tolist()) if not existing_positions.empty else set()

    today = date.today()
    new_positions = []

    print(f"\n{'='*70}")
    print(f"QUALIFYING EVENTS (EPS surprise >= {MIN_SURPRISE_PCT:.0f}%)")
    print(f"{'='*70}")
    print(f"\n{'Ticker':<8} {'Surprise':>9}  {'Bucket':<10}  {'Report':>12}  "
          f"{'Est EPS':>8}  {'Act EPS':>8}  {'Action'}")
    print("-" * 70)

    for c in candidates:
        ticker = c["ticker"]
        action = ""

        if ticker in open_tickers:
            action = "SKIP (open position)"
        elif ticker not in data or data[ticker].empty:
            action = "SKIP (no price data)"
        else:
            df = data[ticker]
            # Use yesterday's close as entry proxy; actual open used at settlement
            ref_close = float(df["close"].iloc[-1])
            entry_exec = ref_close * (1 + SLIPPAGE_PCT + TRANSACTION_COST_PCT)
            exit_date = _next_bday(today, HOLDING_DAYS)

            new_positions.append({
                "ticker": ticker,
                "entry_date": today.isoformat(),
                "entry_price": round(entry_exec, 4),
                "surprise_pct": c["surprise_pct"],
                "surprise_bucket": c["surprise_bucket"],
                "holding_days": HOLDING_DAYS,
                "exit_date": exit_date.isoformat(),
            })
            action = f"BUY at open  → exit {exit_date}"

        print(
            f"{ticker:<8} {c['surprise_pct']:>+8.1f}%  {c['surprise_bucket']:<10}  "
            f"{str(c['report_date']):>12}  {c['eps_estimate']:>8.2f}  "
            f"{c['eps_actual']:>8.2f}  {action}"
        )

    if new_positions:
        updated = pd.concat(
            [existing_positions, pd.DataFrame(new_positions)], ignore_index=True
        )
        updated.to_csv(POSITIONS_FILE, index=False)

        print(f"\n  → {len(new_positions)} new position(s) logged.")
        print(f"  → Execute as market BUY orders at 9:30 AM ET open.")
        print(f"  → Positions saved to: {POSITIONS_FILE}")
    else:
        print("\n  No new positions to log.")


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

    avg_hold = trades["holding_days_actual"].mean() if "holding_days_actual" in trades else HOLDING_DAYS
    sharpe = (avg_ret / std_ret) * np.sqrt(252 / max(avg_hold, 1)) if std_ret > 0 else 0.0

    cum_returns = (1 + trades["return_pct"] / 100).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    gross_wins = trades.loc[trades["profitable"], "pnl_usd"].sum()
    gross_losses = abs(trades.loc[~trades["profitable"], "pnl_usd"].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

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

    alerts = []
    if win_rate < ALERT_WIN_RATE_MIN:
        alerts.append(f"WIN RATE {win_rate:.1%} < {ALERT_WIN_RATE_MIN:.1%}")
    if sharpe < ALERT_SHARPE_MIN:
        alerts.append(f"SHARPE {sharpe:.2f} < {ALERT_SHARPE_MIN:.2f}")
    if max_dd * 100 < ALERT_DRAWDOWN_MAX * 100:
        alerts.append(f"DRAWDOWN {max_dd*100:.1f}% < {ALERT_DRAWDOWN_MAX*100:.1f}%")
    if len(recent) >= 10 and recent_win_rate < ALERT_WIN_RATE_MIN:
        alerts.append(f"RECENT WIN RATE {recent_win_rate:.1%} (last {len(recent)} trades)")

    # Breakdown by surprise bucket
    bucket_stats = []
    for bucket in ["10-15%", "15-25%", "25%+"]:
        bt = trades[trades["surprise_bucket"] == bucket]
        if len(bt) > 0:
            bucket_stats.append((bucket, len(bt), bt["profitable"].mean(), bt["pnl_usd"].sum()))

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

    if bucket_stats:
        print(f"\n{'--- By Surprise Bucket ---'}")
        print(f"  {'Bucket':<10}  {'Trades':>6}  {'WR':>6}  {'P&L':>10}")
        for bucket, n, wr, pnl in bucket_stats:
            print(f"  {bucket:<10}  {n:>6}  {wr:>5.0%}  ${pnl:>9,.0f}")

    if alerts:
        print(f"\n{'='*70}")
        print("ALERTS")
        print(f"{'='*70}")
        for a in alerts:
            print(f"  ! {a}")
    else:
        print(f"\n  All metrics within acceptable ranges.")

    return metrics


def show_summary():
    """Display current paper trading summary."""
    print("=" * 70)
    print("PEAD PAPER TRADING SUMMARY")
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
        print(f"\n{'Ticker':<8} {'Surprise':>9}  {'Bucket':<10}  {'Entry':>12}  {'Exit':>12}  {'Hold':>4}")
        print("-" * 65)
        positions["exit_date"] = pd.to_datetime(positions["exit_date"]).dt.date
        for _, row in positions.sort_values("exit_date").iterrows():
            print(
                f"{row['ticker']:<8} {row.get('surprise_pct', 0):>+8.1f}%  "
                f"{row.get('surprise_bucket', '?'):<10}  "
                f"{str(row['entry_date']):>12}  {str(row['exit_date']):>12}  "
                f"{row['holding_days']:>3}d"
            )

    if not trades.empty:
        _update_metrics(verbose=True)

        print(f"\n{'='*70}")
        print("LAST 10 CLOSED TRADES")
        print(f"{'='*70}")
        print(f"\n{'Ticker':<8} {'Bucket':<10}  {'Entry':>10}  {'Exit':>10}  {'Ret%':>7}  {'P&L':>8}  {'W/L'}")
        print("-" * 70)
        for _, row in trades.sort_values("exit_date").tail(10).iterrows():
            wl = "WIN" if row["profitable"] else "LOSS"
            print(
                f"{row['ticker']:<8} {str(row.get('surprise_bucket', '?')):<10}  "
                f"${row['entry_price']:>9.2f}  ${row['exit_price']:>9.2f}  "
                f"{row['return_pct']:>+6.2f}%  ${row['pnl_usd']:>+7.0f}  {wl}"
            )
    else:
        print("\nNo closed trades yet.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PEAD Paper Trading Tracker — run daily pre-market (~8:30 AM ET)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m futures.scripts.paper_trade_pead --full       # Full daily workflow
  python -m futures.scripts.paper_trade_pead              # Scan for new signals only
  python -m futures.scripts.paper_trade_pead --check      # Settle matured positions only
  python -m futures.scripts.paper_trade_pead --summary    # Performance dashboard
        """,
    )
    parser.add_argument("--check", action="store_true", help="Settle matured positions")
    parser.add_argument("--summary", action="store_true", help="Show performance dashboard")
    parser.add_argument(
        "--full", action="store_true",
        help="Full daily workflow: refresh data → settle exits → scan signals → summary",
    )
    parser.add_argument(
        "--universe", choices=["small", "medium", "large"], default="medium",
        help="Ticker universe to scan (default: medium)",
    )
    args = parser.parse_args()

    universe = get_universe(args.universe)

    if args.summary:
        show_summary()
        return

    if args.full:
        print("Refreshing market data...")
        data = refresh_market_data(universe, show_progress=False)
        check_outcomes(data=data)
        print()
        log_signals(data=data, universe=universe)
        print()
        show_summary()
        return

    if args.check:
        check_outcomes()
        return

    # Default: scan for new signals only
    log_signals(universe=universe)


if __name__ == "__main__":
    main()
