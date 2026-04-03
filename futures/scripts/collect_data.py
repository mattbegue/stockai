"""Batch data collection script for metalabeling strategy."""

import argparse
from datetime import date, timedelta

from futures.config import get_universe, get_settings
from futures.data.fetcher import DataManager


def main():
    """Collect historical data for all tickers in the specified universe."""
    parser = argparse.ArgumentParser(description="Collect historical market data")
    parser.add_argument(
        "--universe",
        choices=["small", "medium", "large"],
        default="small",
        help="Universe size: small (~50), medium (~150), large (~300)",
    )
    args = parser.parse_args()

    universe = get_universe(args.universe)
    settings = get_settings()
    dm = DataManager()

    # Calculate date range (5 years of history)
    end_date = date.today()
    start_date = end_date - timedelta(days=settings.default_lookback_days)

    print(f"Collecting data for {len(universe.all_tickers)} tickers")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Tradeable: {len(universe.tradeable)} stocks")
    print(f"Context: {len(universe.context)} ETFs/indices")
    print()

    # Fetch all data with refresh=True to update from API
    data = dm.get_multi(
        universe.all_tickers,
        start_date=start_date,
        end_date=end_date,
        refresh=True,
        show_progress=True,
    )

    # Report results
    print(f"\nSuccessfully collected {len(data)} tickers:")
    print("-" * 40)

    tradeable_count = 0
    context_count = 0

    for ticker in universe.tradeable:
        if ticker in data:
            print(f"  {ticker}: {len(data[ticker]):,} bars")
            tradeable_count += 1
        else:
            print(f"  {ticker}: MISSING")

    print()
    for ticker in universe.context:
        if ticker in data:
            print(f"  {ticker}: {len(data[ticker]):,} bars")
            context_count += 1
        else:
            print(f"  {ticker}: MISSING")

    print("-" * 40)
    print(f"Tradeable: {tradeable_count}/{len(universe.tradeable)}")
    print(f"Context: {context_count}/{len(universe.context)}")
    print(f"Total: {len(data)}/{len(universe.all_tickers)}")

    # Report any missing tickers
    missing = set(universe.all_tickers) - set(data.keys())
    if missing:
        print(f"\nMissing tickers: {sorted(missing)}")


if __name__ == "__main__":
    main()
