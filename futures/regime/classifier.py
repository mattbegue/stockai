"""
Market regime classifier.

Classifies the current market environment as BULL, NEUTRAL, or BEAR
using a composite scoring approach across trend, momentum, volatility,
and credit market signals.

Design: deliberately rules-based and interpretable. Each of 5 indicators
votes +1 (bullish) or -1 (bearish). The sum determines the regime:
  +2 to +5  → BULL
  -1 to +1  → NEUTRAL
  -5 to -2  → BEAR

This keeps the regime signal robust (no single indicator dominates),
easy to audit, and free from look-ahead bias.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"

    def __int__(self) -> int:
        return {"bull": 1, "neutral": 0, "bear": -1}[self.value]


@dataclass
class RegimeReading:
    """Full regime reading for a specific date."""

    regime: MarketRegime
    score: int          # -5 to +5
    score_pct: float    # 0.0 to 1.0 (normalised to [0,1])
    components: dict    # individual signal votes

    @property
    def is_bull(self) -> bool:
        return self.regime == MarketRegime.BULL

    @property
    def is_bear(self) -> bool:
        return self.regime == MarketRegime.BEAR

    def __repr__(self) -> str:
        return (
            f"RegimeReading(regime={self.regime.value}, score={self.score:+d}, "
            f"components={self.components})"
        )


class MarketRegimeClassifier:
    """
    Classifies market regime from context ETF data.

    Five scored signals:
      1. SPY above 50-day SMA          (trend, short-term)
      2. SPY above 200-day SMA         (trend, long-term)
      3. SPY 20-day return positive    (momentum)
      4. VXX below 63-day average      (volatility)
      5. HYG 10d return > TLT 10d return  (credit / risk appetite)

    Requires at least 200 bars of SPY data for full accuracy.
    Falls back gracefully when VXX or TLT/HYG is unavailable.
    """

    BULL_THRESHOLD = 2    # score >= 2 → BULL
    BEAR_THRESHOLD = -2   # score <= -2 → BEAR

    def __init__(
        self,
        spy_ticker: str = "SPY",
        vxx_ticker: str = "VXX",
        tlt_ticker: str = "TLT",
        hyg_ticker: str = "HYG",
    ):
        self.spy_ticker = spy_ticker
        self.vxx_ticker = vxx_ticker
        self.tlt_ticker = tlt_ticker
        self.hyg_ticker = hyg_ticker

    def classify(
        self,
        data: dict[str, pd.DataFrame],
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> RegimeReading:
        """
        Classify the market regime as of a specific date.

        Args:
            data: Dict of ticker → OHLCV DataFrame
            as_of_date: Date to classify. Defaults to the last date in SPY data.

        Returns:
            RegimeReading with regime, score, and component breakdown
        """
        spy_df = data.get(self.spy_ticker)
        if spy_df is None or len(spy_df) < 20:
            # No SPY data — return neutral, can't classify
            return RegimeReading(
                regime=MarketRegime.NEUTRAL, score=0, score_pct=0.5,
                components={"error": "insufficient_spy_data"}
            )

        # Determine reference index
        if as_of_date is not None and as_of_date in spy_df.index:
            idx = spy_df.index.get_loc(as_of_date)
        else:
            idx = len(spy_df) - 1

        score = 0
        components = {}

        spy_close = spy_df["close"]

        # --- Signal 1: SPY above 50-day SMA ---
        if idx >= 50:
            sma50 = spy_close.iloc[idx - 50 : idx + 1].mean()
            vote = 1 if spy_close.iloc[idx] > sma50 else -1
            score += vote
            components["spy_above_sma50"] = vote
        else:
            components["spy_above_sma50"] = 0

        # --- Signal 2: SPY above 200-day SMA ---
        if idx >= 200:
            sma200 = spy_close.iloc[idx - 200 : idx + 1].mean()
            vote = 1 if spy_close.iloc[idx] > sma200 else -1
            score += vote
            components["spy_above_sma200"] = vote
        else:
            # Not enough data — use 50-day as proxy
            if idx >= 50:
                sma50 = spy_close.iloc[idx - 50 : idx + 1].mean()
                vote = 1 if spy_close.iloc[idx] > sma50 else -1
                score += vote
                components["spy_above_sma200"] = vote
            else:
                components["spy_above_sma200"] = 0

        # --- Signal 3: SPY 20-day momentum ---
        if idx >= 20:
            momentum_20d = spy_close.iloc[idx] / spy_close.iloc[idx - 20] - 1
            vote = 1 if momentum_20d > 0 else -1
            score += vote
            components["spy_momentum_20d"] = vote
        else:
            components["spy_momentum_20d"] = 0

        # --- Signal 4: VXX below 63-day average (low volatility regime) ---
        vxx_df = data.get(self.vxx_ticker)
        if vxx_df is not None and len(vxx_df) >= 63:
            vxx_date = as_of_date if as_of_date in vxx_df.index else vxx_df.index[min(idx, len(vxx_df) - 1)]
            vxx_idx = vxx_df.index.get_loc(vxx_date) if vxx_date in vxx_df.index else min(idx, len(vxx_df) - 1)
            if vxx_idx >= 63:
                vxx_close = vxx_df["close"].iloc[vxx_idx]
                vxx_avg63 = vxx_df["close"].iloc[vxx_idx - 63 : vxx_idx].mean()
                vote = 1 if vxx_close < vxx_avg63 else -1
                score += vote
                components["vxx_below_avg63"] = vote
            else:
                components["vxx_below_avg63"] = 0
        else:
            # VXX unavailable — use SPY 20-day vol as proxy
            if idx >= 20:
                returns = spy_close.pct_change()
                vol_20d = returns.iloc[idx - 20 : idx].std() * np.sqrt(252)
                vol_hist = returns.iloc[max(0, idx - 126) : idx].std() * np.sqrt(252)
                vote = 1 if vol_20d < vol_hist else -1
                score += vote
                components["vxx_below_avg63"] = vote
            else:
                components["vxx_below_avg63"] = 0

        # --- Signal 5: HYG outperforms TLT over 10 days (risk-on) ---
        tlt_df = data.get(self.tlt_ticker)
        hyg_df = data.get(self.hyg_ticker)
        if tlt_df is not None and hyg_df is not None:
            tlt_date = as_of_date if (as_of_date is not None and as_of_date in tlt_df.index) else tlt_df.index[min(idx, len(tlt_df) - 1)]
            hyg_date = as_of_date if (as_of_date is not None and as_of_date in hyg_df.index) else hyg_df.index[min(idx, len(hyg_df) - 1)]

            tlt_idx = tlt_df.index.get_loc(tlt_date) if tlt_date in tlt_df.index else min(idx, len(tlt_df) - 1)
            hyg_idx = hyg_df.index.get_loc(hyg_date) if hyg_date in hyg_df.index else min(idx, len(hyg_df) - 1)

            if tlt_idx >= 10 and hyg_idx >= 10:
                tlt_10d = tlt_df["close"].iloc[tlt_idx] / tlt_df["close"].iloc[tlt_idx - 10] - 1
                hyg_10d = hyg_df["close"].iloc[hyg_idx] / hyg_df["close"].iloc[hyg_idx - 10] - 1
                vote = 1 if hyg_10d > tlt_10d else -1  # HYG > TLT = risk-on
                score += vote
                components["hyg_outperforms_tlt"] = vote
            else:
                components["hyg_outperforms_tlt"] = 0
        else:
            components["hyg_outperforms_tlt"] = 0

        # Determine regime from score
        if score >= self.BULL_THRESHOLD:
            regime = MarketRegime.BULL
        elif score <= self.BEAR_THRESHOLD:
            regime = MarketRegime.BEAR
        else:
            regime = MarketRegime.NEUTRAL

        # Normalise score to [0,1] for downstream use
        max_possible = sum(1 for v in components.values() if v != 0)
        score_pct = (score + max_possible) / (2 * max_possible) if max_possible > 0 else 0.5

        return RegimeReading(
            regime=regime,
            score=score,
            score_pct=float(score_pct),
            components=components,
        )

    def classify_series(
        self,
        data: dict[str, pd.DataFrame],
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """
        Classify regime for a series of dates.

        Useful for backtesting and visualisation.

        Returns:
            DataFrame with columns: regime, score, score_pct, and one column
            per component signal.
        """
        spy_df = data.get(self.spy_ticker)
        if spy_df is None:
            return pd.DataFrame()

        if dates is None:
            dates = spy_df.index

        rows = []
        for date in dates:
            reading = self.classify(data, as_of_date=date)
            row = {
                "date": date,
                "regime": reading.regime.value,
                "score": reading.score,
                "score_pct": reading.score_pct,
            }
            row.update(reading.components)
            rows.append(row)

        return pd.DataFrame(rows).set_index("date")
