"""
Collect earnings surprise data for the PEAD strategy.

Fetches EPS estimate vs actual (surprise %) from yfinance for all tradeable
tickers in the universe. Data is cached locally and used by the PEAD backtester.

Usage:
    python -m futures.scripts.collect_earnings_data --universe medium
    python -m futures.scripts.collect_earnings_data --universe medium --refresh
"""

import argparse
import sys

from tqdm import tqdm

from futures.config.universes import get_universe
from futures.data.earnings_surprise import EarningsSurpriseData


def main():
    parser = argparse.ArgumentParser(description="Collect earnings surprise data")
    parser.add_argument(
        "--universe",
        choices=["small", "medium", "large"],
        default="medium",
        help="Universe size (default: medium)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-fetch from yfinance (ignore cache)",
    )
    parser.add_argument(
        "--quarters",
        type=int,
        default=25,
        help="Number of quarters to fetch (default: 25 ~ 6 years)",
    )
    args = parser.parse_args()

    universe = get_universe(args.universe)
    tickers = universe.tradeable
    print(f"Collecting earnings data for {len(tickers)} tickers ({args.universe} universe)")
    print(f"Quarters: {args.quarters} | Refresh: {args.refresh}")
    print()

    surprise_data = EarningsSurpriseData()

    # Fetch with progress bar
    newly_fetched = 0
    for ticker in tqdm(tickers, desc="Fetching"):
        cached = surprise_data._load_cache(ticker)
        if cached is not None and not args.refresh:
            surprise_data._events[ticker] = cached
            continue
        events = surprise_data._fetch(ticker, args.quarters)
        surprise_data._events[ticker] = events
        surprise_data._save_cache(ticker, events)
        newly_fetched += 1

    surprise_data._by_entry_date = None  # Reset index

    print(f"\nFetched {newly_fetched} tickers from yfinance (rest from cache)")
    print()

    # Summary stats
    stats = surprise_data.summary()
    print("=" * 50)
    print("EARNINGS DATA SUMMARY")
    print("=" * 50)
    print(f"Total events:       {stats['total_events']:,}")
    print(f"Tickers with data:  {stats['tickers_with_data']} / {len(tickers)}")
    print(f"Date range:         {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"Beats:              {stats['beats']:,} ({stats['beats']/max(stats['total_events'],1)*100:.1f}%)")
    print(f"Misses:             {stats['misses']:,} ({stats['misses']/max(stats['total_events'],1)*100:.1f}%)")
    print(f"Median surprise:    {stats['median_surprise_pct']:+.2f}%")
    print(f"Mean surprise:      {stats['mean_surprise_pct']:+.2f}%")
    print()

    # Distribution of surprise magnitude
    all_events = [e for evs in surprise_data._events.values() for e in evs]
    if all_events:
        beats = [e for e in all_events if e.is_beat]
        print("Beat magnitude distribution:")
        for threshold in [2, 5, 10, 15, 20]:
            n = sum(1 for e in beats if e.surprise_pct >= threshold)
            print(f"  >= {threshold:2d}%: {n:4d} events ({n/len(beats)*100:.1f}% of beats)")

    print("\nDone. Run backtest_pead.py to validate the PEAD baseline.")


if __name__ == "__main__":
    main()
