"""
PEAD (Post-Earnings Announcement Drift) strategy.

Generates BUY signals for tickers that beat EPS estimates by more than
a configurable threshold. The engine generates signals at day T close
and executes at day T+1 open, which maps naturally to PEAD:

  - AMC report on day T → signal on T → enters T+1 open  (correct)
  - BMO report on day T → signal on T → enters T+1 open  (1 day late, acceptable)

No ML filtering — this is the raw baseline. The edge must exist here
before adding any complexity.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from futures.data.earnings_surprise import EarningsSurpriseData, EarningsEvent
from futures.strategies.base import Strategy, Signal


class PEADStrategy(Strategy):
    """
    Event-driven strategy based on earnings surprise magnitude.

    Signal logic:
        - BUY when EPS surprise >= min_surprise_pct (beat threshold)
        - Optionally SELL on misses <= -min_surprise_pct (long_only=False)
        - Signals generated on report_date, executed at next open
        - Optionally filtered by SPY 200-day MA regime (regime_filter=True)

    Parameters:
        surprise_data:     Pre-loaded EarningsSurpriseData instance
        min_surprise_pct:  Minimum EPS surprise % to trigger (default 5.0)
        holding_days:      Trading days to hold each position (default 15)
        long_only:         If True, skip miss signals (default True)
        spy_data:          SPY OHLCV DataFrame for regime filter (optional)
        regime_ma_days:    MA window for regime detection (default 200)
    """

    # Position size tiers keyed by (lo, hi) surprise % bounds → portfolio fraction
    _SIZE_TIERS: list[tuple[float, float, float]] = [
        (25.0, float("inf"), 0.08),
        (15.0, 25.0,         0.05),
        (0.0,  15.0,         0.03),
    ]

    def __init__(
        self,
        surprise_data: EarningsSurpriseData,
        min_surprise_pct: float = 5.0,
        holding_days: int = 15,
        long_only: bool = True,
        spy_data: pd.DataFrame | None = None,
        regime_ma_days: int = 200,
        variable_sizing: bool = False,
    ):
        self._surprise_data = surprise_data
        self.min_surprise_pct = min_surprise_pct
        self._holding_days = holding_days
        self.long_only = long_only
        self._regime_ma_days = regime_ma_days
        self._variable_sizing = variable_sizing
        self._last_surprise_pcts: dict[str, float] = {}  # populated each bar

        # Pre-compute regime Series: date -> True (bull) / False (bear)
        if spy_data is not None and not spy_data.empty:
            spy_ma = spy_data["close"].rolling(regime_ma_days, min_periods=regime_ma_days).mean()
            self._regime: pd.Series | None = spy_data["close"] > spy_ma
            self._regime.index = pd.to_datetime(self._regime.index)
        else:
            self._regime = None

    @property
    def name(self) -> str:
        regime_tag = f",regime={self._regime_ma_days}dMA" if self._regime is not None else ""
        return f"PEAD(min={self.min_surprise_pct:.0f}%,hold={self._holding_days}d{regime_tag})"

    @property
    def required_history(self) -> int:
        return 1  # No warmup — signal is purely event-driven

    @property
    def params(self) -> dict:
        return {
            "min_surprise_pct": self.min_surprise_pct,
            "holding_days": self._holding_days,
            "long_only": self.long_only,
            "regime_filter": self._regime is not None,
            "regime_ma_days": self._regime_ma_days,
        }

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, Signal]:
        """
        Generate BUY signals for tickers that reported qualifying earnings today.

        The engine passes `data` sliced up to the current bar, so the last
        index of any DataFrame IS the current trading date.

        Args:
            data: Dict of ticker → OHLCV DataFrame sliced to current date

        Returns:
            Dict of ticker → Signal.BUY (or SELL if long_only=False)
        """
        # Derive current date from the data (any ticker works)
        current_date: date | None = None
        for df in data.values():
            if not df.empty:
                ts = df.index[-1]
                current_date = ts.date() if hasattr(ts, "date") else ts
                break

        if current_date is None:
            return {}

        # Regime filter: skip all entries when SPY is below its MA
        if self._regime is not None:
            ts = pd.Timestamp(current_date)
            past = self._regime[self._regime.index <= ts]
            if past.empty or not bool(past.iloc[-1]):
                return {}

        signals: dict[str, Signal] = {}
        self._last_surprise_pcts = {}

        # Find all earnings events reported on this date
        # Use report_date (not entry_date) since the engine handles the T+1 execution
        for ticker, ticker_events in self._surprise_data._events.items():
            event = self._get_report_date_event(ticker_events, current_date)
            if event is None:
                continue

            # Must have price data available for this ticker
            if ticker not in data or data[ticker].empty:
                continue

            if event.surprise_pct >= self.min_surprise_pct:
                signals[ticker] = Signal.BUY
                self._last_surprise_pcts[ticker] = event.surprise_pct
            elif not self.long_only and event.surprise_pct <= -self.min_surprise_pct:
                signals[ticker] = Signal.SELL
                self._last_surprise_pcts[ticker] = event.surprise_pct

        return signals

    def get_position_sizes(self, signals: dict[str, Signal]) -> dict[str, float]:
        """Scale position size by surprise magnitude when variable_sizing=True."""
        if not self._variable_sizing:
            return {}
        sizes: dict[str, float] = {}
        for ticker in signals:
            surprise = abs(self._last_surprise_pcts.get(ticker, 0.0))
            for lo, hi, frac in self._SIZE_TIERS:
                if lo <= surprise < hi:
                    sizes[ticker] = frac
                    break
        return sizes

    def get_position_holding_days(self, signals: dict[str, Signal]) -> dict[str, int]:
        """Uniform holding period for all PEAD positions."""
        return {ticker: self._holding_days for ticker in signals}

    def get_event_for_ticker_date(
        self, ticker: str, report_date: date
    ) -> EarningsEvent | None:
        """Look up the earnings event for a ticker on a given report date."""
        events = self._surprise_data.all_events(ticker)
        return self._get_report_date_event(events, report_date)

    @staticmethod
    def _get_report_date_event(
        events: list[EarningsEvent], target_date: date
    ) -> EarningsEvent | None:
        """Return the event matching target_date, or None."""
        for event in events:
            if event.report_date == target_date:
                return event
        return None
