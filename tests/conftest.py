"""Shared fixtures for all tests."""

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 100, start: str = "2023-01-02", seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic OHLCV DataFrame with a business-day index."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = 100 + rng.standard_normal(n).cumsum()
    noise = rng.uniform(0.5, 2.0, n)
    df = pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, n),
            "high": close + noise,
            "low": close - noise,
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n),
        },
        index=dates,
    )
    # Ensure OHLC sanity
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def multi_ohlcv() -> dict[str, pd.DataFrame]:
    return {
        "AAPL": make_ohlcv(seed=1),
        "MSFT": make_ohlcv(seed=2),
        "SPY": make_ohlcv(seed=3),
    }
