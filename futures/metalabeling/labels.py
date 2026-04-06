"""Label creation for metalabeling strategy.

Triple Barrier Method (Lopez de Prado, Advances in Financial Machine Learning):
  Three exit conditions are set at entry time. Whichever fires first determines
  the label:
    1. Upper barrier (profit target): price rises profit_target% → label = 1
    2. Lower barrier (stop loss):     price falls stop_loss%    → label = 0
    3. Vertical barrier (time limit): holding_period expires    → label based on
                                      final return vs min_return

This matches actual trade execution more faithfully than a fixed-horizon label,
and encodes stop-loss behaviour into the training signal itself.
"""

from typing import Optional

import pandas as pd

from .signals import CandidateSignal, Signal


def _apply_triple_barrier(
    df: pd.DataFrame,
    entry_idx: int,
    holding_period: int,
    direction: Signal,
    profit_target: float,
    stop_loss: float,
    min_return: float,
) -> tuple[float, float, str, int]:
    """Scan bars from entry through the vertical barrier and return exit details.

    Barrier priority on the same bar (pessimistic / conservative):
      stop loss > profit target

    Args:
        df:             OHLCV DataFrame for the ticker.
        entry_idx:      Bar index where we enter (buy at open).
        holding_period: Maximum bars to hold.
        direction:      BUY or SELL.
        profit_target:  Fractional profit that triggers the upper barrier.
        stop_loss:      Fractional loss that triggers the lower barrier.
        min_return:     Minimum return for a time-exit to be labelled 1.

    Returns:
        (forward_return, exit_price, exit_type, bars_held)
        exit_type in {"profit_target", "stop_loss", "time"}
    """
    entry_price = df["open"].iloc[entry_idx]

    if direction == Signal.BUY:
        upper_price = entry_price * (1.0 + profit_target)
        lower_price = entry_price * (1.0 - stop_loss)
    else:  # SELL / short
        upper_price = entry_price * (1.0 - profit_target)  # price falling = win
        lower_price = entry_price * (1.0 + stop_loss)      # price rising  = loss

    # Scan each bar (including the entry bar itself — intraday move after open)
    for i in range(holding_period + 1):
        bar_idx = entry_idx + i
        if bar_idx >= len(df):
            break

        bar_high = df["high"].iloc[bar_idx]
        bar_low = df["low"].iloc[bar_idx]

        if direction == Signal.BUY:
            # Stop loss checked first (pessimistic — we don't know intraday order)
            if bar_low <= lower_price:
                exit_price = lower_price
                return (exit_price - entry_price) / entry_price, exit_price, "stop_loss", i
            if bar_high >= upper_price:
                exit_price = upper_price
                return (exit_price - entry_price) / entry_price, exit_price, "profit_target", i
        else:  # SELL
            # For shorts: stop loss = price spikes up past lower_price
            if bar_high >= lower_price:
                exit_price = lower_price
                return (entry_price - exit_price) / entry_price, exit_price, "stop_loss", i
            if bar_low <= upper_price:
                exit_price = upper_price
                return (entry_price - exit_price) / entry_price, exit_price, "profit_target", i

    # Vertical barrier: exit at close of last available bar
    exit_bar_idx = min(entry_idx + holding_period, len(df) - 1)
    exit_price = df["close"].iloc[exit_bar_idx]
    bars_held = exit_bar_idx - entry_idx

    if direction == Signal.BUY:
        forward_return = (exit_price - entry_price) / entry_price
    else:
        forward_return = (entry_price - exit_price) / entry_price

    return forward_return, exit_price, "time", bars_held


def create_metalabels(
    data: dict[str, pd.DataFrame],
    candidates: list[CandidateSignal],
    holding_period: int = 5,
    min_return: float = 0.003,
    profit_target: float = 0.015,
    stop_loss: float = 0.010,
) -> pd.DataFrame:
    """Create triple-barrier labels for candidate signals.

    Labels are determined by whichever barrier is hit first:
      - Profit target hit → label = 1
      - Stop loss hit     → label = 0
      - Time expires      → label = 1 if return > min_return, else 0

    Args:
        data:           Dict mapping ticker → OHLCV DataFrame.
        candidates:     List of CandidateSignal objects.
        holding_period: Maximum bars to hold (vertical barrier).
        min_return:     Minimum return for time-exit to count as profitable.
        profit_target:  Upper barrier as fraction of entry price (default 1.5%).
        stop_loss:      Lower barrier as fraction of entry price (default 1.0%).

    Returns:
        DataFrame with columns:
          ticker, date, direction, source_indicators, num_indicators,
          entry_price, exit_price, forward_return, exit_type, bars_held, label
    """
    records = []

    for candidate in candidates:
        ticker = candidate.ticker
        signal_date = candidate.date

        if ticker not in data:
            continue

        df = data[ticker]

        # Locate the signal bar
        if signal_date not in df.index:
            try:
                idx = df.index.get_indexer([signal_date], method="nearest")[0]
                if idx < 0 or idx >= len(df):
                    continue
                signal_date = df.index[idx]
            except Exception:
                continue

        signal_idx = df.index.get_loc(signal_date)

        # Entry: next trading day's open (EOD signal → next-morning execution)
        entry_idx = signal_idx + 1
        if entry_idx >= len(df):
            continue

        # Need at least one bar beyond entry for a meaningful exit
        if entry_idx + 1 >= len(df):
            continue

        forward_return, exit_price, exit_type, bars_held = _apply_triple_barrier(
            df=df,
            entry_idx=entry_idx,
            holding_period=holding_period,
            direction=candidate.direction,
            profit_target=profit_target,
            stop_loss=stop_loss,
            min_return=min_return,
        )

        # Label: 1 = good trade (profit target hit, or time-exit above threshold)
        if exit_type == "profit_target":
            label = 1
        elif exit_type == "stop_loss":
            label = 0
        else:  # time
            label = 1 if forward_return > min_return else 0

        records.append({
            "ticker": ticker,
            "date": signal_date,
            "direction": candidate.direction.value,
            "source_indicators": candidate.source_indicators,
            "num_indicators": len(candidate.source_indicators),
            "entry_price": df["open"].iloc[entry_idx],
            "exit_price": exit_price,
            "forward_return": forward_return,
            "exit_type": exit_type,
            "bars_held": bars_held,
            "label": label,
        })

    return pd.DataFrame(records)


def get_label_statistics(labeled_df: pd.DataFrame) -> dict:
    """Compute statistics about the labeled dataset."""
    if len(labeled_df) == 0:
        return {"total": 0}

    total = len(labeled_df)
    profitable = labeled_df["label"].sum()
    unprofitable = total - profitable

    buy_signals = labeled_df[labeled_df["direction"] == 1]
    sell_signals = labeled_df[labeled_df["direction"] == -1]

    stats = {
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
            if profitable > 0 else 0
        ),
        "avg_return_unprofitable": (
            labeled_df[labeled_df["label"] == 0]["forward_return"].mean() * 100
            if unprofitable > 0 else 0
        ),
        "unique_tickers": labeled_df["ticker"].nunique(),
        "date_range": (
            labeled_df["date"].min().strftime("%Y-%m-%d"),
            labeled_df["date"].max().strftime("%Y-%m-%d"),
        ),
    }

    # Exit type breakdown (triple barrier specific)
    if "exit_type" in labeled_df.columns:
        for et in ["profit_target", "stop_loss", "time"]:
            n = (labeled_df["exit_type"] == et).sum()
            stats[f"exit_{et}_count"] = n
            stats[f"exit_{et}_pct"] = n / total * 100

    return stats


def split_by_date(
    labeled_df: pd.DataFrame,
    train_end_date: Optional[pd.Timestamp] = None,
    train_pct: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split labeled data into train/test sets by date."""
    labeled_df = labeled_df.sort_values("date")

    if train_end_date is None:
        n_train = int(len(labeled_df) * train_pct)
        train_df = labeled_df.iloc[:n_train].copy()
        test_df = labeled_df.iloc[n_train:].copy()
    else:
        train_df = labeled_df[labeled_df["date"] <= train_end_date].copy()
        test_df = labeled_df[labeled_df["date"] > train_end_date].copy()

    return train_df, test_df
