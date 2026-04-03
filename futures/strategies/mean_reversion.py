"""Mean reversion strategies."""

import pandas as pd

from futures.indicators import SMA, RSI, BollingerBands
from .base import Strategy, Signal


class MeanReversion(Strategy):
    """
    Mean reversion strategy using Bollinger Bands.

    BUY when price touches lower band and RSI is oversold.
    SELL when price touches upper band and RSI is overbought.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ):
        """Initialize mean reversion strategy."""
        super().__init__(
            bb_period=bb_period,
            bb_std=bb_std,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
        )
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        self.bb = BollingerBands(period=bb_period, std_dev=bb_std)
        self.rsi = RSI(period=rsi_period)

    @property
    def name(self) -> str:
        return f"mean_rev_bb{self.bb_period}_rsi{self.rsi_period}"

    @property
    def required_history(self) -> int:
        return max(self.bb_period, self.rsi_period) + 1

    def precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add BB and RSI columns."""
        result = df.copy()

        bb = self.bb(df)
        result["bb_lower"] = bb["lower"]
        result["bb_upper"] = bb["upper"]
        result["bb_pct_b"] = bb["pct_b"]

        result[f"rsi_{self.rsi_period}"] = self.rsi(df)

        return result

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """Generate mean reversion signals."""
        signals = {}

        for ticker, df in data.items():
            if len(df) < self.required_history:
                signals[ticker] = Signal.HOLD
                continue

            # Get or compute indicators
            if "bb_pct_b" in df.columns:
                pct_b = df["bb_pct_b"].iloc[-1]
                rsi = df[f"rsi_{self.rsi_period}"].iloc[-1]
            else:
                bb = self.bb(df)
                pct_b = bb["pct_b"].iloc[-1]
                rsi = self.rsi(df).iloc[-1]

            if pd.isna(pct_b) or pd.isna(rsi):
                signals[ticker] = Signal.HOLD
                continue

            # Oversold: price near lower band + low RSI
            if pct_b < 0.0 and rsi < self.rsi_oversold:
                signals[ticker] = Signal.BUY
            # Overbought: price near upper band + high RSI
            elif pct_b > 1.0 and rsi > self.rsi_overbought:
                signals[ticker] = Signal.SELL
            else:
                signals[ticker] = Signal.HOLD

        return signals


class RSIMeanReversion(Strategy):
    """
    Simple RSI mean reversion.

    BUY when RSI crosses up through oversold.
    SELL when RSI crosses down through overbought.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ):
        """Initialize RSI mean reversion strategy."""
        super().__init__(
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought,
        )
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.rsi = RSI(period=rsi_period)

    @property
    def name(self) -> str:
        return f"rsi_mr_{self.rsi_period}_{self.oversold}_{self.overbought}"

    @property
    def required_history(self) -> int:
        return self.rsi_period + 2

    def precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI column."""
        result = df.copy()
        result[f"rsi_{self.rsi_period}"] = self.rsi(df)
        return result

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """Generate RSI crossover signals."""
        signals = {}

        for ticker, df in data.items():
            if len(df) < self.required_history:
                signals[ticker] = Signal.HOLD
                continue

            rsi_col = f"rsi_{self.rsi_period}"
            if rsi_col in df.columns:
                rsi = df[rsi_col]
            else:
                rsi = self.rsi(df)

            if pd.isna(rsi.iloc[-1]) or pd.isna(rsi.iloc[-2]):
                signals[ticker] = Signal.HOLD
                continue

            curr_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]

            # Cross up through oversold
            if prev_rsi < self.oversold and curr_rsi >= self.oversold:
                signals[ticker] = Signal.BUY
            # Cross down through overbought
            elif prev_rsi > self.overbought and curr_rsi <= self.overbought:
                signals[ticker] = Signal.SELL
            else:
                signals[ticker] = Signal.HOLD

        return signals
