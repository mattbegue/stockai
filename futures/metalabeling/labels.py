"""Label creation for metalabeling strategy."""

from typing import Optional

import pandas as pd

from .signals import CandidateSignal, Signal


def create_metalabels(
    data: dict[str, pd.DataFrame],
    candidates: list[CandidateSignal],
    holding_period: int = 5,
    min_return: float = 0.003,
) -> pd.DataFrame:
    """
    Create binary labels for candidate signals based on forward returns.

    A signal is labeled as profitable (1) if the forward return over the
    holding period exceeds the minimum return threshold (to cover costs).

    Args:
        data: Dict mapping ticker to OHLCV DataFrame
        candidates: List of CandidateSignal objects
        holding_period: Number of days to hold the position
        min_return: Minimum return threshold for "profitable" label (default 0.3%)

    Returns:
        DataFrame with columns:
        - ticker: Stock ticker
        - date: Signal date
        - direction: BUY (1) or SELL (-1)
        - source_indicators: List of indicators that fired
        - entry_price: Open price on the trading day after signal date
        - exit_price: Close price holding_period days after entry
        - forward_return: Actual return over holding period
        - label: 1 if profitable, 0 otherwise
    """
    records = []

    for candidate in candidates:
        ticker = candidate.ticker
        signal_date = candidate.date

        if ticker not in data:
            continue

        df = data[ticker]

        # Find the signal date in the dataframe
        if signal_date not in df.index:
            # Try to find closest date
            try:
                idx = df.index.get_indexer([signal_date], method="nearest")[0]
                if idx < 0 or idx >= len(df):
                    continue
                signal_date = df.index[idx]
            except Exception:
                continue

        signal_idx = df.index.get_loc(signal_date)

        # Entry: next trading day's open (signal generated at EOD, order executes next morning)
        entry_idx = signal_idx + 1
        if entry_idx >= len(df):
            continue

        # Exit: close of the holding_period-th day after entry
        exit_idx = entry_idx + holding_period
        if exit_idx >= len(df):
            continue

        entry_price = df["open"].iloc[entry_idx]
        exit_price = df["close"].iloc[exit_idx]

        # Calculate return based on direction
        if candidate.direction == Signal.BUY:
            forward_return = (exit_price - entry_price) / entry_price
        else:  # SELL
            forward_return = (entry_price - exit_price) / entry_price

        # Label: profitable if return exceeds threshold
        label = 1 if forward_return > min_return else 0

        records.append({
            "ticker": ticker,
            "date": signal_date,
            "direction": candidate.direction.value,
            "source_indicators": candidate.source_indicators,
            "num_indicators": len(candidate.source_indicators),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "forward_return": forward_return,
            "label": label,
        })

    return pd.DataFrame(records)


def get_label_statistics(labeled_df: pd.DataFrame) -> dict:
    """
    Compute statistics about the labeled dataset.

    Args:
        labeled_df: DataFrame from create_metalabels()

    Returns:
        Dict with statistics about the labels
    """
    if len(labeled_df) == 0:
        return {"total": 0}

    total = len(labeled_df)
    profitable = labeled_df["label"].sum()
    unprofitable = total - profitable

    buy_signals = labeled_df[labeled_df["direction"] == 1]
    sell_signals = labeled_df[labeled_df["direction"] == -1]

    return {
        "total": total,
        "profitable": profitable,
        "unprofitable": unprofitable,
        "profitable_pct": profitable / total * 100,
        "buy_signals": len(buy_signals),
        "buy_profitable_pct": (
            buy_signals["label"].mean() * 100 if len(buy_signals) > 0 else 0
        ),
        "sell_signals": len(sell_signals),
        "sell_profitable_pct": (
            sell_signals["label"].mean() * 100 if len(sell_signals) > 0 else 0
        ),
        "avg_return": labeled_df["forward_return"].mean() * 100,
        "avg_return_profitable": (
            labeled_df[labeled_df["label"] == 1]["forward_return"].mean() * 100
            if profitable > 0
            else 0
        ),
        "avg_return_unprofitable": (
            labeled_df[labeled_df["label"] == 0]["forward_return"].mean() * 100
            if unprofitable > 0
            else 0
        ),
        "unique_tickers": labeled_df["ticker"].nunique(),
        "date_range": (
            labeled_df["date"].min().strftime("%Y-%m-%d"),
            labeled_df["date"].max().strftime("%Y-%m-%d"),
        ),
    }


def split_by_date(
    labeled_df: pd.DataFrame,
    train_end_date: Optional[pd.Timestamp] = None,
    train_pct: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split labeled data into train/test sets by date.

    Args:
        labeled_df: DataFrame from create_metalabels()
        train_end_date: Explicit end date for training data
        train_pct: Fraction of data for training if train_end_date not specified

    Returns:
        Tuple of (train_df, test_df)
    """
    labeled_df = labeled_df.sort_values("date")

    if train_end_date is None:
        # Use percentage split
        n_train = int(len(labeled_df) * train_pct)
        train_df = labeled_df.iloc[:n_train].copy()
        test_df = labeled_df.iloc[n_train:].copy()
    else:
        train_df = labeled_df[labeled_df["date"] <= train_end_date].copy()
        test_df = labeled_df[labeled_df["date"] > train_end_date].copy()

    return train_df, test_df
