"""Primary signal generation for metalabeling strategy."""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import numpy as np

from futures.indicators.momentum import RSI, MACD, BollingerBands, Stochastic, ROC
from futures.indicators.moving_averages import SMA, VWAP
from enum import Enum


class Signal(Enum):
    """Trading signals (duplicated to avoid circular import)."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class CandidateSignal:
    """A candidate trade signal from the primary ensemble."""

    ticker: str
    date: pd.Timestamp
    direction: Signal  # BUY or SELL
    source_indicators: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        sources = ", ".join(self.source_indicators)
        return f"CandidateSignal({self.ticker}, {self.date.date()}, {self.direction.name}, [{sources}])"


class PrimarySignalGenerator:
    """
    Generates trade candidates from an ensemble of technical indicators.

    Uses "any fires" mode - a candidate is generated if ANY indicator signals.
    This provides more training data for the meta-model to filter.
    """

    def __init__(
        self,
        sma_fast: int = 20,
        sma_slow: int = 50,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        stoch_k: int = 14,
        stoch_d: int = 3,
        stoch_oversold: float = 30.0,
        stoch_overbought: float = 70.0,
        roc_period: int = 10,
        roc_threshold: float = 3.0,
        vwap_period: int = 20,
        vwap_band: float = 0.01,
        volume_breakout_period: int = 20,
        volume_breakout_multiplier: float = 1.5,
        obv_sma_period: int = 20,
    ):
        """
        Initialize the signal generator with indicator parameters.

        Original signals (4):
            sma_fast/slow: SMA crossover parameters
            rsi_*: RSI extreme parameters
            macd_*: MACD crossover parameters
            bb_*: Bollinger Band touch parameters

        New signals (5, P2-S1):
            stoch_*: Stochastic %K/%D crossover in oversold/overbought zones
            roc_*: ROC momentum reversal (zero-cross from extreme)
            vwap_*: VWAP mean-reversion cross with minimum displacement band
            volume_breakout_*: Price breakout above N-day high with volume confirmation
            obv_sma_period: OBV crosses its own N-day SMA (volume-trend shift)
        """
        self.sma_fast_period = sma_fast
        self.sma_slow_period = sma_slow
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought
        self.roc_threshold = roc_threshold
        self.vwap_band = vwap_band
        self.volume_breakout_period = volume_breakout_period
        self.volume_breakout_multiplier = volume_breakout_multiplier
        self.obv_sma_period = obv_sma_period

        # Original indicators
        self.sma_fast = SMA(period=sma_fast)
        self.sma_slow = SMA(period=sma_slow)
        self.rsi = RSI(period=rsi_period)
        self.macd = MACD(fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
        self.bb = BollingerBands(period=bb_period, std_dev=bb_std)

        # New indicators (P2-S1)
        self.stoch = Stochastic(k_period=stoch_k, d_period=stoch_d)
        self.roc = ROC(period=roc_period)
        self.vwap = VWAP(period=vwap_period)

    @property
    def required_history(self) -> int:
        """Minimum bars needed for all indicators to be valid."""
        return self.sma_slow_period + 10  # Slow SMA + buffer for crossover detection

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators for a single ticker.

        Args:
            df: OHLCV DataFrame with DatetimeIndex

        Returns:
            DataFrame with indicator columns added
        """
        result = df.copy()

        # SMAs
        result["sma_fast"] = self.sma_fast(df)
        result["sma_slow"] = self.sma_slow(df)

        # RSI
        result["rsi"] = self.rsi(df)

        # MACD
        macd_df = self.macd(df)
        result["macd_line"] = macd_df["macd"]
        result["macd_signal"] = macd_df["signal"]
        result["macd_histogram"] = macd_df["histogram"]

        # Bollinger Bands
        bb_df = self.bb(df)
        result["bb_upper"] = bb_df["upper"]
        result["bb_lower"] = bb_df["lower"]
        result["bb_middle"] = bb_df["middle"]
        result["bb_pct_b"] = bb_df["pct_b"]

        # --- New indicators (P2-S1) ---

        # Stochastic %K and %D
        stoch_df = self.stoch(df)
        result["stoch_k"] = stoch_df["k"]
        result["stoch_d"] = stoch_df["d"]

        # Rate of Change
        result["roc"] = self.roc(df)

        # Rolling VWAP
        result["vwap"] = self.vwap(df)

        # OBV and its SMA — OBV is cumulative so computed inline
        price_diff = df["close"].diff()
        vol_signed = df["volume"] * np.sign(price_diff).fillna(0)
        result["obv"] = vol_signed.cumsum()
        result["obv_sma"] = result["obv"].rolling(window=self.obv_sma_period).mean()

        # Volume rolling stats for breakout check
        result["vol_avg"] = df["volume"].rolling(window=self.volume_breakout_period).mean()
        result["high_nd"] = df["close"].rolling(window=self.volume_breakout_period).max().shift(1)

        return result

    def _check_sma_crossover(
        self, df: pd.DataFrame, idx: int
    ) -> tuple[Optional[Signal], str]:
        """Check for SMA crossover signal at given index."""
        if idx < 1:
            return None, ""

        fast_curr = df["sma_fast"].iloc[idx]
        fast_prev = df["sma_fast"].iloc[idx - 1]
        slow_curr = df["sma_slow"].iloc[idx]
        slow_prev = df["sma_slow"].iloc[idx - 1]

        if pd.isna(fast_curr) or pd.isna(slow_curr) or pd.isna(fast_prev) or pd.isna(slow_prev):
            return None, ""

        # Bullish crossover: fast crosses above slow
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            return Signal.BUY, "sma_crossover"

        # Bearish crossover: fast crosses below slow
        if fast_prev >= slow_prev and fast_curr < slow_curr:
            return Signal.SELL, "sma_crossover"

        return None, ""

    def _check_rsi_extreme(
        self, df: pd.DataFrame, idx: int
    ) -> tuple[Optional[Signal], str]:
        """Check for RSI extreme signal at given index."""
        if idx < 1:
            return None, ""

        rsi_curr = df["rsi"].iloc[idx]
        rsi_prev = df["rsi"].iloc[idx - 1]

        if pd.isna(rsi_curr) or pd.isna(rsi_prev):
            return None, ""

        # Oversold bounce: RSI crosses above oversold threshold
        if rsi_prev <= self.rsi_oversold and rsi_curr > self.rsi_oversold:
            return Signal.BUY, "rsi_oversold"

        # Overbought reversal: RSI crosses below overbought threshold
        if rsi_prev >= self.rsi_overbought and rsi_curr < self.rsi_overbought:
            return Signal.SELL, "rsi_overbought"

        return None, ""

    def _check_macd_crossover(
        self, df: pd.DataFrame, idx: int
    ) -> tuple[Optional[Signal], str]:
        """Check for MACD crossover signal at given index."""
        if idx < 1:
            return None, ""

        macd_curr = df["macd_line"].iloc[idx]
        macd_prev = df["macd_line"].iloc[idx - 1]
        signal_curr = df["macd_signal"].iloc[idx]
        signal_prev = df["macd_signal"].iloc[idx - 1]

        if pd.isna(macd_curr) or pd.isna(signal_curr) or pd.isna(macd_prev) or pd.isna(signal_prev):
            return None, ""

        # Bullish crossover: MACD crosses above signal
        if macd_prev <= signal_prev and macd_curr > signal_curr:
            return Signal.BUY, "macd_crossover"

        # Bearish crossover: MACD crosses below signal
        if macd_prev >= signal_prev and macd_curr < signal_curr:
            return Signal.SELL, "macd_crossover"

        return None, ""

    def _check_bb_touch(
        self, df: pd.DataFrame, idx: int
    ) -> tuple[Optional[Signal], str]:
        """Check for Bollinger Band touch signal at given index."""
        if idx < 1:
            return None, ""

        close_curr = df["close"].iloc[idx]
        close_prev = df["close"].iloc[idx - 1]
        lower_curr = df["bb_lower"].iloc[idx]
        lower_prev = df["bb_lower"].iloc[idx - 1]
        upper_curr = df["bb_upper"].iloc[idx]
        upper_prev = df["bb_upper"].iloc[idx - 1]

        if pd.isna(lower_curr) or pd.isna(upper_curr):
            return None, ""

        # Touch lower band and bounce: price was at/below lower, now above
        if close_prev <= lower_prev and close_curr > lower_curr:
            return Signal.BUY, "bb_lower_touch"

        # Touch upper band and reverse: price was at/above upper, now below
        if close_prev >= upper_prev and close_curr < upper_curr:
            return Signal.SELL, "bb_upper_touch"

        return None, ""

    def _check_stochastic_crossover(
        self, df: pd.DataFrame, idx: int
    ) -> tuple[Optional[Signal], str]:
        """Stochastic %K crosses %D while inside oversold / overbought zone."""
        if idx < 1:
            return None, ""

        k_curr = df["stoch_k"].iloc[idx]
        k_prev = df["stoch_k"].iloc[idx - 1]
        d_curr = df["stoch_d"].iloc[idx]
        d_prev = df["stoch_d"].iloc[idx - 1]

        if any(pd.isna(v) for v in [k_curr, k_prev, d_curr, d_prev]):
            return None, ""

        # Bullish: %K crosses above %D while in oversold territory
        if k_prev <= d_prev and k_curr > d_curr and k_curr < self.stoch_oversold:
            return Signal.BUY, "stoch_crossover"

        # Bearish: %K crosses below %D while in overbought territory
        if k_prev >= d_prev and k_curr < d_curr and k_curr > self.stoch_overbought:
            return Signal.SELL, "stoch_crossover"

        return None, ""

    def _check_volume_breakout(
        self, df: pd.DataFrame, idx: int
    ) -> tuple[Optional[Signal], str]:
        """Price breaks N-day high on above-average volume (momentum breakout)."""
        if idx < self.volume_breakout_period:
            return None, ""

        close = df["close"].iloc[idx]
        high_nd = df["high_nd"].iloc[idx]   # prior N-day high (shift(1) applied in compute)
        vol = df["volume"].iloc[idx]
        vol_avg = df["vol_avg"].iloc[idx]

        if pd.isna(high_nd) or pd.isna(vol_avg) or vol_avg == 0:
            return None, ""

        if close > high_nd and vol > self.volume_breakout_multiplier * vol_avg:
            return Signal.BUY, "volume_breakout"

        return None, ""

    def _check_obv_cross(
        self, df: pd.DataFrame, idx: int
    ) -> tuple[Optional[Signal], str]:
        """OBV crosses its own N-day SMA — signals volume trend shift."""
        if idx < 1:
            return None, ""

        obv_curr = df["obv"].iloc[idx]
        obv_prev = df["obv"].iloc[idx - 1]
        sma_curr = df["obv_sma"].iloc[idx]
        sma_prev = df["obv_sma"].iloc[idx - 1]

        if any(pd.isna(v) for v in [obv_curr, obv_prev, sma_curr, sma_prev]):
            return None, ""

        if obv_prev <= sma_prev and obv_curr > sma_curr:
            return Signal.BUY, "obv_cross"

        if obv_prev >= sma_prev and obv_curr < sma_curr:
            return Signal.SELL, "obv_cross"

        return None, ""

    def _check_roc_reversal(
        self, df: pd.DataFrame, idx: int
    ) -> tuple[Optional[Signal], str]:
        """ROC zero-crosses from an extreme, signalling momentum exhaustion reversal."""
        if idx < 1:
            return None, ""

        roc_curr = df["roc"].iloc[idx]
        roc_prev = df["roc"].iloc[idx - 1]

        if pd.isna(roc_curr) or pd.isna(roc_prev):
            return None, ""

        # BUY: coming from meaningfully negative ROC, now crossing zero
        if roc_prev < -self.roc_threshold and roc_curr >= 0:
            return Signal.BUY, "roc_reversal"

        # SELL: coming from meaningfully positive ROC, now crossing zero
        if roc_prev > self.roc_threshold and roc_curr <= 0:
            return Signal.SELL, "roc_reversal"

        return None, ""

    def _check_vwap_cross(
        self, df: pd.DataFrame, idx: int
    ) -> tuple[Optional[Signal], str]:
        """Price crosses VWAP from a displaced position (mean reversion)."""
        if idx < 1:
            return None, ""

        close_curr = df["close"].iloc[idx]
        close_prev = df["close"].iloc[idx - 1]
        vwap_curr = df["vwap"].iloc[idx]
        vwap_prev = df["vwap"].iloc[idx - 1]

        if pd.isna(vwap_curr) or pd.isna(vwap_prev) or vwap_prev == 0:
            return None, ""

        # BUY: price was at least band% below VWAP, now crosses above
        if close_prev < vwap_prev * (1 - self.vwap_band) and close_curr > vwap_curr:
            return Signal.BUY, "vwap_cross"

        # SELL: price was at least band% above VWAP, now crosses below
        if close_prev > vwap_prev * (1 + self.vwap_band) and close_curr < vwap_curr:
            return Signal.SELL, "vwap_cross"

        return None, ""

    def generate_signals_for_ticker(
        self, ticker: str, df: pd.DataFrame
    ) -> list[CandidateSignal]:
        """
        Generate all candidate signals for a single ticker.

        Args:
            ticker: Stock ticker symbol
            df: OHLCV DataFrame with DatetimeIndex

        Returns:
            List of CandidateSignal objects
        """
        if len(df) < self.required_history:
            return []

        # Compute indicators
        df_with_indicators = self.compute_indicators(df)

        candidates = []

        # Iterate through each bar (skip warmup period)
        for idx in range(self.required_history, len(df_with_indicators)):
            date = df_with_indicators.index[idx]

            # Collect signals from all indicators
            buy_sources = []
            sell_sources = []

            # Check each indicator
            for check_fn in [
                self._check_sma_crossover,
                self._check_rsi_extreme,
                self._check_macd_crossover,
                self._check_bb_touch,
                self._check_stochastic_crossover,
                self._check_volume_breakout,
                self._check_obv_cross,
                self._check_roc_reversal,
                self._check_vwap_cross,
            ]:
                signal, source = check_fn(df_with_indicators, idx)
                if signal == Signal.BUY:
                    buy_sources.append(source)
                elif signal == Signal.SELL:
                    sell_sources.append(source)

            # Create candidate if any indicator fired (any-fires mode)
            if buy_sources:
                candidates.append(
                    CandidateSignal(
                        ticker=ticker,
                        date=pd.Timestamp(date),
                        direction=Signal.BUY,
                        source_indicators=buy_sources,
                    )
                )
            if sell_sources:
                candidates.append(
                    CandidateSignal(
                        ticker=ticker,
                        date=pd.Timestamp(date),
                        direction=Signal.SELL,
                        source_indicators=sell_sources,
                    )
                )

        return candidates

    def generate_candidates(
        self, data: dict[str, pd.DataFrame], show_progress: bool = False
    ) -> list[CandidateSignal]:
        """
        Generate all candidate signals for multiple tickers.

        Args:
            data: Dict mapping ticker to OHLCV DataFrame
            show_progress: Show progress bar

        Returns:
            List of all CandidateSignal objects, sorted by date
        """
        all_candidates = []

        tickers = list(data.keys())
        if show_progress:
            from tqdm import tqdm
            tickers = tqdm(tickers, desc="Generating signals")

        for ticker in tickers:
            df = data[ticker]
            candidates = self.generate_signals_for_ticker(ticker, df)
            all_candidates.extend(candidates)

        # Sort by date
        all_candidates.sort(key=lambda c: c.date)

        return all_candidates

    def get_current_signals(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, tuple[Optional[Signal], list[str]]]:
        """
        Get signals for the most recent bar only (for live trading).

        Args:
            data: Dict mapping ticker to OHLCV DataFrame

        Returns:
            Dict mapping ticker to (Signal or None, list of source indicators)
        """
        results = {}

        for ticker, df in data.items():
            if len(df) < self.required_history:
                results[ticker] = (None, [])
                continue

            df_with_indicators = self.compute_indicators(df)
            idx = len(df_with_indicators) - 1

            buy_sources = []
            sell_sources = []

            for check_fn in [
                self._check_sma_crossover,
                self._check_rsi_extreme,
                self._check_macd_crossover,
                self._check_bb_touch,
                self._check_stochastic_crossover,
                self._check_volume_breakout,
                self._check_obv_cross,
                self._check_roc_reversal,
                self._check_vwap_cross,
            ]:
                signal, source = check_fn(df_with_indicators, idx)
                if signal == Signal.BUY:
                    buy_sources.append(source)
                elif signal == Signal.SELL:
                    sell_sources.append(source)

            if buy_sources and not sell_sources:
                results[ticker] = (Signal.BUY, buy_sources)
            elif sell_sources and not buy_sources:
                results[ticker] = (Signal.SELL, sell_sources)
            elif buy_sources and sell_sources:
                # Conflicting signals - use majority or default to HOLD
                if len(buy_sources) > len(sell_sources):
                    results[ticker] = (Signal.BUY, buy_sources)
                elif len(sell_sources) > len(buy_sources):
                    results[ticker] = (Signal.SELL, sell_sources)
                else:
                    results[ticker] = (None, [])
            else:
                results[ticker] = (None, [])

        return results
