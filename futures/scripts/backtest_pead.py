"""
PEAD baseline backtest — no ML, raw earnings surprise signal.

Tests whether the edge exists before adding any ML complexity.

Usage:
    python -m futures.scripts.backtest_pead
    python -m futures.scripts.backtest_pead --min-surprise 5 --holding-days 15
    python -m futures.scripts.backtest_pead --start-date 2022-01-01
    python -m futures.scripts.backtest_pead --min-surprise 3 --holding-days 10 --long-short
"""

import argparse
from datetime import date

import pandas as pd

from futures.backtester.engine import Backtester
from futures.config.universes import get_universe
from futures.data.earnings_surprise import EarningsSurpriseData
from futures.data.fetcher import DataManager
from futures.strategies.pead_strategy import PEADStrategy


def main():
    parser = argparse.ArgumentParser(description="PEAD baseline backtest")
    parser.add_argument("--universe", choices=["small", "medium", "large"], default="medium")
    parser.add_argument(
        "--min-surprise", type=float, default=5.0,
        help="Minimum EPS surprise %% to trigger a trade (default: 5.0)",
    )
    parser.add_argument(
        "--holding-days", type=int, default=15,
        help="Trading days to hold each position (default: 15)",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Backtest start date YYYY-MM-DD (default: earliest available)",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="Backtest end date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--stop-loss", type=float, default=None,
        help="Stop loss as fraction of entry price (e.g. 0.03 = 3%%). Default: none.",
    )
    parser.add_argument(
        "--profit-target", type=float, default=None,
        help="Profit target as fraction of entry price (e.g. 0.10 = 10%%). Default: none.",
    )
    parser.add_argument(
        "--long-short", action="store_true",
        help="Also short tickers that miss by >= min-surprise (default: long-only)",
    )
    parser.add_argument(
        "--position-size", type=float, default=0.05,
        help="Position size as fraction of portfolio (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--variable-sizing", action="store_true",
        help="Scale position size by surprise magnitude (3%/5%/8% for 10-15/15-25/25+%%)",
    )
    parser.add_argument(
        "--regime-filter", action="store_true",
        help="Only trade when SPY is above its 200-day MA (bull regime)",
    )
    parser.add_argument(
        "--regime-ma-days", type=int, default=200,
        help="MA window for regime filter (default: 200)",
    )
    args = parser.parse_args()

    universe = get_universe(args.universe)
    print("=" * 60)
    print("PEAD Baseline Backtest")
    print("=" * 60)
    print(f"Universe:        {args.universe} ({len(universe.tradeable)} tickers)")
    print(f"Min EPS surprise: {args.min_surprise:.1f}%")
    print(f"Holding days:    {args.holding_days}")
    print(f"Mode:            {'Long/Short' if args.long_short else 'Long-only'}")
    print(f"Position size:   {args.position_size*100:.0f}% per trade")
    if args.variable_sizing:
        print(f"Position sizing: variable (3%/5%/8% for 10-15/15-25/25+%% surprise)")
    if args.regime_filter:
        print(f"Regime filter:   SPY > {args.regime_ma_days}-day MA (bull only)")

    # Load earnings surprise data
    print("\nLoading earnings surprise data...")
    surprise_data = EarningsSurpriseData()
    surprise_data.load(universe.tradeable)
    stats = surprise_data.summary()
    print(f"  {stats['total_events']:,} events, {stats['tickers_with_data']} tickers")
    print(f"  {stats['beats']:,} beats / {stats['misses']:,} misses")

    # Load price data
    print("\nLoading price data...")
    dm = DataManager()
    data = dm.get_multi(universe.all_tickers, refresh=False, show_progress=True)
    tradeable_data = {t: data[t] for t in universe.tradeable if t in data}
    print(f"  Loaded {len(tradeable_data)} tradeable tickers")

    # Determine date range
    all_dates = sorted(set(
        d for df in tradeable_data.values() for d in df.index
    ))
    if not all_dates:
        print("ERROR: No price data available")
        return

    start = pd.Timestamp(args.start_date) if args.start_date else all_dates[0]
    end = pd.Timestamp(args.end_date) if args.end_date else all_dates[-1]

    print(f"\nBacktest period: {start.date()} to {end.date()}")

    # Benchmark: SPY buy-and-hold
    spy_data = data.get("SPY")
    spy_start_price = None
    spy_end_price = None
    if spy_data is not None:
        spy_window = spy_data[(spy_data.index >= start) & (spy_data.index <= end)]
        if not spy_window.empty:
            spy_start_price = spy_window["open"].iloc[0]
            spy_end_price = spy_window["close"].iloc[-1]

    # Build strategy and backtester
    strategy = PEADStrategy(
        surprise_data=surprise_data,
        min_surprise_pct=args.min_surprise,
        holding_days=args.holding_days,
        long_only=not args.long_short,
        spy_data=spy_data if args.regime_filter else None,
        regime_ma_days=args.regime_ma_days,
        variable_sizing=args.variable_sizing,
    )

    if args.stop_loss:
        print(f"Stop loss:       {args.stop_loss*100:.1f}%")
    if args.profit_target:
        print(f"Profit target:   {args.profit_target*100:.1f}%")

    backtester = Backtester(
        strategy=strategy,
        position_size=args.position_size,
        transaction_cost_pct=0.001,
        slippage_pct=0.0005,
        stop_loss=args.stop_loss,
        profit_target=args.profit_target,
    )

    print("\nRunning backtest...")
    result = backtester.run(
        data=tradeable_data,
        start_date=start.date() if hasattr(start, "date") else start,
        end_date=end.date() if hasattr(end, "date") else end,
        show_progress=True,
    )

    metrics = result.metrics

    spy_return = (
        (spy_end_price - spy_start_price) / spy_start_price * 100
        if spy_start_price else None
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'PEAD':>12} {'SPY B&H':>12}")
    print("-" * 55)
    spy_ret_str = f"{spy_return:+.2f}%" if spy_return is not None else "N/A"
    print(f"{'Total Return':<30} {metrics.total_return:>+11.2f}% {spy_ret_str:>12}")
    print(f"{'Sharpe Ratio':<30} {metrics.sharpe_ratio:>12.2f}")
    print(f"{'Max Drawdown':<30} {-metrics.max_drawdown:>+11.2f}%")

    trades = result.trades
    if trades is not None and len(trades) > 0:
        n = len(trades)
        winners = trades[trades["pnl"] > 0]
        losers = trades[trades["pnl"] <= 0]
        wr = len(winners) / n * 100
        avg_win = winners["pnl"].mean() if len(winners) > 0 else 0
        avg_loss = losers["pnl"].mean() if len(losers) > 0 else 0
        gross_profit = winners["pnl"].sum()
        gross_loss = abs(losers["pnl"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        months = max((end - start).days / 30, 1)

        print(f"\n{'TRADE STATISTICS':-^55}")
        print(f"{'Total Trades':<30} {n:>12}")
        print(f"{'Win Rate':<30} {wr:>11.2f}%")
        print(f"{'Profit Factor':<30} {pf:>12.2f}")
        print(f"{'Avg Winner':<30} ${avg_win:>11,.2f}")
        print(f"{'Avg Loser':<30} ${avg_loss:>11,.2f}")
        print(f"{'Trades per month':<30} {n / months:>12.1f}")

        # Surprise magnitude breakdown
        print(f"\n{'BY SURPRISE MAGNITUDE':-^55}")
        buckets = [
            ("5-10%",  5,  10),
            ("10-15%", 10, 15),
            ("15-25%", 15, 25),
            ("25%+",   25, 999),
        ]
        for label, lo, hi in buckets:
            mask = []
            for _, row in trades.iterrows():
                entry_d = row["entry_date"].date() if hasattr(row["entry_date"], "date") else row["entry_date"]
                # Signal generated day before entry; try entry-1 as report_date
                report_d = (pd.Timestamp(entry_d) - pd.offsets.BDay(1)).date()
                event = strategy.get_event_for_ticker_date(row["ticker"], report_d)
                if event is None:
                    event = strategy.get_event_for_ticker_date(row["ticker"], entry_d)
                mask.append(event is not None and lo <= event.surprise_pct < hi)
            bucket = trades[mask]
            if len(bucket) > 0:
                bwr = (bucket["pnl"] > 0).mean() * 100
                bpnl = bucket["pnl"].sum()
                print(f"  {label:<10} {len(bucket):>4} trades  WR={bwr:.0f}%  P&L=${bpnl:>8,.0f}")
    else:
        print("\nNo trades executed.")
        print("Try lowering --min-surprise or broadening the date range.")

    print()


if __name__ == "__main__":
    main()
