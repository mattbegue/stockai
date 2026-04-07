"""Feature engineering for the meta-model."""

from typing import Optional

import numpy as np
import pandas as pd

from futures.indicators.momentum import RSI, MACD, BollingerBands
from futures.indicators.moving_averages import SMA
from futures.indicators.volatility import ATR


# Mapping of tickers to their sector ETF (expanded for ~300 stock universe)
SECTOR_MAP = {
    # =========================================================================
    # TECHNOLOGY (XLK)
    # =========================================================================
    # Mega-cap tech
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "GOOG": "XLK", "META": "XLK",
    "NVDA": "XLK", "TSLA": "XLK",
    # Semiconductors
    "AMD": "XLK", "INTC": "XLK", "AVGO": "XLK", "TXN": "XLK", "QCOM": "XLK",
    "MU": "XLK", "ADI": "XLK", "LRCX": "XLK", "AMAT": "XLK", "KLAC": "XLK",
    "MCHP": "XLK", "NXPI": "XLK", "ON": "XLK", "SWKS": "XLK", "MRVL": "XLK",
    "MPWR": "XLK",
    # Software
    "CRM": "XLK", "ORCL": "XLK", "ADBE": "XLK", "NOW": "XLK", "INTU": "XLK",
    "SNPS": "XLK", "CDNS": "XLK", "ANSS": "XLK", "WDAY": "XLK", "TEAM": "XLK",
    "DDOG": "XLK", "ZS": "XLK", "CRWD": "XLK", "PANW": "XLK", "FTNT": "XLK",
    "SPLK": "XLK",
    # IT Services & Hardware
    "CSCO": "XLK", "ACN": "XLK", "IBM": "XLK", "HPQ": "XLK", "HPE": "XLK",
    "DELL": "XLK", "NTAP": "XLK", "WDC": "XLK", "STX": "XLK",

    # =========================================================================
    # COMMUNICATION SERVICES (XLC) - map to XLK as fallback
    # =========================================================================
    "AMZN": "XLK",  # Consumer discretionary but often grouped with tech
    "NFLX": "XLK", "ABNB": "XLK", "UBER": "XLK", "LYFT": "XLK",
    "SNAP": "XLK", "PINS": "XLK", "MTCH": "XLK",
    "T": "XLK", "VZ": "XLK", "TMUS": "XLK",  # Telecom
    "WBD": "XLK", "PARA": "XLK", "FOX": "XLK",  # Media
    "EA": "XLK", "TTWO": "XLK", "ATVI": "XLK",  # Gaming

    # =========================================================================
    # FINANCIALS (XLF)
    # =========================================================================
    # Banks
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "C": "XLF", "GS": "XLF",
    "MS": "XLF", "USB": "XLF", "PNC": "XLF", "TFC": "XLF", "COF": "XLF",
    "FITB": "XLF", "KEY": "XLF", "RF": "XLF", "CFG": "XLF", "HBAN": "XLF",
    "MTB": "XLF", "ZION": "XLF",
    # Payment processors
    "V": "XLF", "MA": "XLF", "AXP": "XLF", "PYPL": "XLF", "SQ": "XLF",
    "FIS": "XLF", "FISV": "XLF", "GPN": "XLF",
    # Insurance
    "BRK.B": "XLF", "PGR": "XLF", "CB": "XLF", "MMC": "XLF", "AON": "XLF",
    "AJG": "XLF", "TRV": "XLF", "ALL": "XLF", "MET": "XLF", "AFL": "XLF",
    # Asset Management & Exchanges
    "BLK": "XLF", "SCHW": "XLF", "ICE": "XLF", "CME": "XLF", "SPGI": "XLF",
    "MCO": "XLF", "MSCI": "XLF", "NDAQ": "XLF",

    # =========================================================================
    # HEALTHCARE (XLV)
    # =========================================================================
    # Pharma
    "JNJ": "XLV", "PFE": "XLV", "MRK": "XLV", "ABBV": "XLV", "LLY": "XLV",
    "BMY": "XLV", "GILD": "XLV", "VRTX": "XLV", "REGN": "XLV", "BIIB": "XLV",
    "MRNA": "XLV", "ZTS": "XLV", "TAK": "XLV",
    # Healthcare Equipment
    "ABT": "XLV", "MDT": "XLV", "DHR": "XLV", "SYK": "XLV", "ISRG": "XLV",
    "BDX": "XLV", "BSX": "XLV", "EW": "XLV", "BAX": "XLV", "HOLX": "XLV",
    "IDXX": "XLV", "ALGN": "XLV", "DXCM": "XLV", "TFX": "XLV",
    # Healthcare Services
    "UNH": "XLV", "ELV": "XLV", "CI": "XLV", "HUM": "XLV", "CNC": "XLV",
    "CVS": "XLV", "MCK": "XLV", "CAH": "XLV", "ABC": "XLV",
    # Life Sciences & Biotech
    "TMO": "XLV", "AMGN": "XLV", "IQV": "XLV", "A": "XLV", "MTD": "XLV",
    "WAT": "XLV", "PKI": "XLV", "TECH": "XLV", "BIO": "XLV",
    # Healthcare Facilities
    "HCA": "XLV", "THC": "XLV", "UHS": "XLV", "DVA": "XLV",

    # =========================================================================
    # CONSUMER DISCRETIONARY (XLY)
    # =========================================================================
    # Retail
    "HD": "XLY", "LOW": "XLY", "TJX": "XLY", "ROST": "XLY", "ORLY": "XLY",
    "AZO": "XLY", "BBY": "XLY", "ULTA": "XLY", "DG": "XLY", "DLTR": "XLY",
    "KMX": "XLY", "AN": "XLY", "GPC": "XLY", "AAP": "XLY",
    # Restaurants & Leisure
    "MCD": "XLY", "SBUX": "XLY", "CMG": "XLY", "YUM": "XLY", "DRI": "XLY",
    "WYNN": "XLY", "LVS": "XLY", "MGM": "XLY",
    # Hotels & Travel
    "MAR": "XLY", "HLT": "XLY", "BKNG": "XLY", "EXPE": "XLY", "CCL": "XLY",
    "RCL": "XLY", "NCLH": "XLY",
    # Apparel & Luxury
    "NKE": "XLY", "LULU": "XLY", "TPR": "XLY", "VFC": "XLY", "PVH": "XLY", "RL": "XLY",
    # Entertainment / Media
    "DIS": "XLY", "CMCSA": "XLY", "CHTR": "XLY", "NWSA": "XLY",
    # Homebuilders
    "DHI": "XLY",

    # =========================================================================
    # CONSUMER STAPLES (XLP)
    # =========================================================================
    # Food & Beverage
    "KO": "XLP", "PEP": "XLP", "MDLZ": "XLP", "GIS": "XLP", "K": "XLP",
    "KHC": "XLP", "SJM": "XLP", "HSY": "XLP", "MKC": "XLP", "CPB": "XLP",
    "CAG": "XLP", "HRL": "XLP", "TSN": "XLP", "BG": "XLP",
    # Household Products
    "PG": "XLP", "CL": "XLP", "KMB": "XLP", "CHD": "XLP", "CLX": "XLP",
    # Retail
    "WMT": "XLP", "COST": "XLP", "TGT": "XLP", "KR": "XLP", "SYY": "XLP",
    # Tobacco & Alcohol
    "PM": "XLP", "MO": "XLP", "STZ": "XLP", "BF.B": "XLP",

    # =========================================================================
    # INDUSTRIALS (XLI)
    # =========================================================================
    # Aerospace & Defense
    "BA": "XLI", "LMT": "XLI", "RTX": "XLI", "GD": "XLI", "NOC": "XLI",
    "HII": "XLI", "TDG": "XLI", "HWM": "XLI", "LHX": "XLI",
    # Machinery & Equipment
    "CAT": "XLI", "DE": "XLI", "EMR": "XLI", "ETN": "XLI", "ROK": "XLI",
    "PH": "XLI", "ITW": "XLI", "IR": "XLI", "DOV": "XLI", "SWK": "XLI",
    "CMI": "XLI", "PCAR": "XLI", "GNRC": "XLI",
    # Transportation
    "UPS": "XLI", "FDX": "XLI", "UNP": "XLI", "NSC": "XLI", "CSX": "XLI",
    "DAL": "XLI", "UAL": "XLI", "LUV": "XLI", "AAL": "XLI",
    "JBHT": "XLI", "CHRW": "XLI", "EXPD": "XLI",
    # Industrial Conglomerates
    "HON": "XLI", "GE": "XLI", "MMM": "XLI",
    # Building Products & Services
    "JCI": "XLI", "CARR": "XLI", "TT": "XLI", "LII": "XLI", "AOS": "XLI", "MAS": "XLI",
    # Waste Management
    "WM": "XLI", "RSG": "XLI", "WCN": "XLI",

    # =========================================================================
    # ENERGY (XLE)
    # =========================================================================
    # Integrated Oil & Gas
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "OXY": "XLE", "HES": "XLE",
    # Exploration & Production
    "EOG": "XLE", "PXD": "XLE", "DVN": "XLE", "FANG": "XLE", "MRO": "XLE",
    "APA": "XLE",
    # Refining & Marketing
    "MPC": "XLE", "PSX": "XLE", "VLO": "XLE",
    # Oil Services
    "SLB": "XLE", "HAL": "XLE", "BKR": "XLE",
    # Midstream
    "WMB": "XLE", "KMI": "XLE", "OKE": "XLE",

    # =========================================================================
    # MATERIALS (XLB) - map to XLI as fallback since XLB not in context
    # =========================================================================
    # Chemicals
    "LIN": "XLI", "APD": "XLI", "SHW": "XLI", "ECL": "XLI", "PPG": "XLI",
    "DD": "XLI", "DOW": "XLI", "LYB": "XLI", "ALB": "XLI", "CE": "XLI",
    # Metals & Mining
    "NEM": "XLI", "FCX": "XLI", "NUE": "XLI", "STLD": "XLI", "CLF": "XLI",

    # =========================================================================
    # UTILITIES (XLU)
    # =========================================================================
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "D": "XLU", "AEP": "XLU",
    "EXC": "XLU", "SRE": "XLU", "XEL": "XLU", "WEC": "XLU", "ES": "XLU",
    "AWK": "XLU", "ED": "XLU",

    # =========================================================================
    # REAL ESTATE (XLRE) - map to XLF as fallback since XLRE not in context
    # =========================================================================
    # Data Centers & Towers
    "AMT": "XLF", "CCI": "XLF", "EQIX": "XLF", "DLR": "XLF",
    # Industrial & Logistics
    "PLD": "XLF", "WELL": "XLF", "AVB": "XLF", "EQR": "XLF",
    # Retail & Office
    "SPG": "XLF", "O": "XLF", "VICI": "XLF", "PSA": "XLF",
    # Residential
    "INVH": "XLF", "MAA": "XLF", "UDR": "XLF",
}


class MetaFeatureEngineering:
    """
    Create features for the meta-model that predicts signal profitability.

    Features include:
    1. Signal quality features (indicator values at signal time)
    2. Stock momentum/volatility features
    3. Market context features (from ETFs)
    4. Regime features
    """

    def __init__(
        self,
        context_tickers: Optional[list[str]] = None,
        lookback_periods: tuple[int, ...] = (5, 10, 20),
        earnings_calendar=None,
        holding_period: int = 5,
    ):
        """
        Initialize the feature engineering pipeline.

        Args:
            context_tickers: List of ETF tickers for market context
            lookback_periods: Periods for momentum calculations
            earnings_calendar: Optional EarningsCalendar instance. When provided,
                               adds `days_to_earnings` and `earnings_within_hold`
                               features to each signal.
            holding_period: Expected holding period in trading days (matches the
                            triple barrier time limit). Used for earnings risk window.
        """
        self.context_tickers = context_tickers or [
            "SPY", "QQQ", "VXX", "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY",
            "XLU", "TLT", "HYG", "GLD",
        ]
        self.lookback_periods = lookback_periods
        self.earnings_calendar = earnings_calendar
        self.holding_period = holding_period

        # Initialize indicators
        self.rsi = RSI(period=14)
        self.macd = MACD()
        self.bb = BollingerBands(period=20)
        self.sma_20 = SMA(period=20)
        self.sma_50 = SMA(period=50)
        self.atr = ATR(period=14)

    def _compute_returns(self, df: pd.DataFrame, periods: tuple[int, ...]) -> dict:
        """Compute returns over multiple periods."""
        features = {}
        for period in periods:
            ret = df["close"].pct_change(period)
            features[f"return_{period}d"] = ret
        return features

    def _get_signal_quality_features(
        self,
        df: pd.DataFrame,
        signal_idx: int,
        direction: int,
        source_indicators: list[str],
    ) -> dict:
        """
        Extract features about the signal quality.

        Args:
            df: OHLCV DataFrame with indicators computed
            signal_idx: Index of the signal date
            direction: 1 for BUY, -1 for SELL
            source_indicators: List of indicators that fired
        """
        features = {}

        # RSI value
        rsi_val = df["rsi"].iloc[signal_idx] if "rsi" in df.columns else np.nan
        features["rsi"] = rsi_val
        features["rsi_oversold"] = 1 if rsi_val < 30 else 0
        features["rsi_overbought"] = 1 if rsi_val > 70 else 0

        # MACD histogram
        if "macd_histogram" in df.columns:
            features["macd_histogram"] = df["macd_histogram"].iloc[signal_idx]
            features["macd_positive"] = 1 if features["macd_histogram"] > 0 else 0

        # Distance from SMAs
        close = df["close"].iloc[signal_idx]
        if "sma_20" in df.columns:
            sma20 = df["sma_20"].iloc[signal_idx]
            features["dist_sma_20_pct"] = (close - sma20) / sma20 * 100 if sma20 else np.nan
            features["above_sma_20"] = 1 if close > sma20 else 0
        if "sma_50" in df.columns:
            sma50 = df["sma_50"].iloc[signal_idx]
            features["dist_sma_50_pct"] = (close - sma50) / sma50 * 100 if sma50 else np.nan
            features["above_sma_50"] = 1 if close > sma50 else 0

        # Bollinger Band position
        if "bb_pct_b" in df.columns:
            features["bb_pct_b"] = df["bb_pct_b"].iloc[signal_idx]

        # Number of indicators agreeing
        features["num_indicators"] = len(source_indicators)
        # Original signal flags
        features["has_sma_crossover"] = 1 if "sma_crossover" in source_indicators else 0
        features["has_rsi_signal"] = 1 if any("rsi" in s for s in source_indicators) else 0
        features["has_macd_crossover"] = 1 if "macd_crossover" in source_indicators else 0
        features["has_bb_touch"] = 1 if any("bb" in s for s in source_indicators) else 0
        # New signal flags (P2-S1) — zero-padded for current model; used in next retrain
        features["has_stoch_crossover"] = 1 if "stoch_crossover" in source_indicators else 0
        features["has_volume_breakout"] = 1 if "volume_breakout" in source_indicators else 0
        features["has_obv_cross"] = 1 if "obv_cross" in source_indicators else 0
        features["has_roc_reversal"] = 1 if "roc_reversal" in source_indicators else 0
        features["has_vwap_cross"] = 1 if "vwap_cross" in source_indicators else 0

        # Signal direction
        features["direction"] = direction

        return features

    def _get_momentum_volatility_features(
        self, df: pd.DataFrame, signal_idx: int
    ) -> dict:
        """Extract momentum and volatility features for the stock."""
        features = {}

        # Returns at different horizons
        for period in self.lookback_periods:
            if signal_idx >= period:
                ret = (
                    df["close"].iloc[signal_idx] / df["close"].iloc[signal_idx - period] - 1
                )
                features[f"return_{period}d"] = ret * 100
            else:
                features[f"return_{period}d"] = np.nan

        # Volatility (20-day rolling std of returns)
        # Use prior 20 bars (exclusive of current bar to avoid lookahead)
        returns = df["close"].pct_change()
        if signal_idx >= 20:
            vol = returns.iloc[signal_idx - 20 : signal_idx].std() * np.sqrt(252)
            features["volatility_20d"] = vol * 100
        else:
            features["volatility_20d"] = np.nan

        # ATR as percentage of price
        if "atr" in df.columns:
            atr_val = df["atr"].iloc[signal_idx]
            close = df["close"].iloc[signal_idx]
            features["atr_pct"] = atr_val / close * 100 if close else np.nan

        # Volume ratio
        # Use prior 20 bars (exclusive of current bar to avoid lookahead)
        if "volume" in df.columns and signal_idx >= 20:
            avg_vol = df["volume"].iloc[signal_idx - 20 : signal_idx].mean()
            curr_vol = df["volume"].iloc[signal_idx]
            features["volume_ratio"] = curr_vol / avg_vol if avg_vol else np.nan
        else:
            features["volume_ratio"] = np.nan

        # Drawdown from recent high
        # Use prior 20 bars (exclusive of current bar to avoid lookahead)
        if signal_idx >= 20:
            recent_high = df["high"].iloc[signal_idx - 20 : signal_idx].max()
            close = df["close"].iloc[signal_idx]
            features["drawdown_20d"] = (close - recent_high) / recent_high * 100
        else:
            features["drawdown_20d"] = np.nan

        return features

    @staticmethod
    def _etf_return(df: pd.DataFrame, idx: int, period: int) -> float:
        """Return period-day return for an ETF at index idx; NaN if not enough history."""
        if idx >= period:
            return df["close"].iloc[idx] / df["close"].iloc[idx - period] - 1
        return np.nan

    @staticmethod
    def _percentile_rank(series: pd.Series, value: float, window: int = 252) -> float:
        """Percentile rank of value in the prior `window` bars of series."""
        if len(series) < 2:
            return np.nan
        return (series < value).sum() / len(series) * 100

    def _get_context_features(
        self,
        context_data: dict[str, pd.DataFrame],
        signal_date: pd.Timestamp,
        ticker: str,
    ) -> dict:
        """
        Extract market context features from ETFs.

        Args:
            context_data: Dict mapping ETF ticker to OHLCV DataFrame
            signal_date: Date of the signal
            ticker: Ticker of the stock (to find sector ETF)
        """
        features = {}

        # ------------------------------------------------------------------
        # SPY features (market momentum)
        # ------------------------------------------------------------------
        spy_idx: int | None = None
        if "SPY" in context_data:
            spy_df = context_data["SPY"]
            if signal_date in spy_df.index:
                spy_idx = spy_df.index.get_loc(signal_date)
                idx = spy_idx

                # SPY returns — signals are generated at EOD, so today's close is available.
                for period in (5, 10, 20):
                    ret = self._etf_return(spy_df, idx, period)
                    features[f"spy_return_{period}d"] = ret * 100 if not np.isnan(ret) else np.nan

                # SPY above 50-SMA (trend regime)
                sma50 = spy_df["close"].rolling(50).mean()
                if idx >= 50:
                    features["spy_above_sma50"] = 1 if spy_df["close"].iloc[idx] > sma50.iloc[idx] else 0
                else:
                    features["spy_above_sma50"] = np.nan

        # ------------------------------------------------------------------
        # VXX features (volatility regime)
        # ------------------------------------------------------------------
        if "VXX" in context_data:
            vxx_df = context_data["VXX"]
            if signal_date in vxx_df.index:
                idx = vxx_df.index.get_loc(signal_date)
                vxx_close = vxx_df["close"].iloc[idx]

                # VXX relative to 20-day average (use prior bars to avoid lookahead)
                if idx >= 20:
                    vxx_avg = vxx_df["close"].iloc[idx - 20 : idx].mean()
                    features["vxx_ratio"] = vxx_close / vxx_avg if vxx_avg else np.nan

                    # VXX percentile (vol regime) - use prior bars to avoid lookahead
                    vxx_values = vxx_df["close"].iloc[max(0, idx - 252) : idx]
                    features["vxx_percentile"] = self._percentile_rank(vxx_values, vxx_close)
                else:
                    features["vxx_ratio"] = np.nan
                    features["vxx_percentile"] = np.nan

        # ------------------------------------------------------------------
        # Sector ETF momentum + relative strength vs SPY (P2-8)
        # ------------------------------------------------------------------
        sector_etf = SECTOR_MAP.get(ticker)
        if sector_etf and sector_etf in context_data:
            sector_df = context_data[sector_etf]
            if signal_date in sector_df.index:
                idx = sector_df.index.get_loc(signal_date)

                # Absolute momentum
                for period in (5, 20):
                    ret = self._etf_return(sector_df, idx, period)
                    if period == 5:
                        features["sector_return_5d"] = ret * 100 if not np.isnan(ret) else np.nan
                    else:
                        features["sector_return_20d"] = ret * 100 if not np.isnan(ret) else np.nan

                # Percentile rank within own 252-day history
                window_prices = sector_df["close"].iloc[max(0, idx - 252) : idx]
                features["sector_percentile"] = self._percentile_rank(
                    window_prices, sector_df["close"].iloc[idx]
                )

                # Relative strength vs SPY (sector alpha)
                if spy_idx is not None and "SPY" in context_data:
                    spy_df2 = context_data["SPY"]
                    for period in (5, 20):
                        sect_ret = self._etf_return(sector_df, idx, period)
                        spy_ret = self._etf_return(spy_df2, spy_idx, period)
                        if not np.isnan(sect_ret) and not np.isnan(spy_ret):
                            features[f"sector_vs_spy_{period}d"] = (sect_ret - spy_ret) * 100
                        else:
                            features[f"sector_vs_spy_{period}d"] = np.nan
        else:
            features["sector_return_5d"] = np.nan
            features["sector_return_20d"] = np.nan
            features["sector_percentile"] = np.nan
            features["sector_vs_spy_5d"] = np.nan
            features["sector_vs_spy_20d"] = np.nan

        # ------------------------------------------------------------------
        # TLT-HYG spread (risk appetite — existing)
        # ------------------------------------------------------------------
        if "TLT" in context_data and "HYG" in context_data:
            tlt_df = context_data["TLT"]
            hyg_df = context_data["HYG"]
            if signal_date in tlt_df.index and signal_date in hyg_df.index:
                tlt_idx = tlt_df.index.get_loc(signal_date)
                hyg_idx = hyg_df.index.get_loc(signal_date)

                if tlt_idx >= 5 and hyg_idx >= 5:
                    tlt_ret = self._etf_return(tlt_df, tlt_idx, 5)
                    hyg_ret = self._etf_return(hyg_df, hyg_idx, 5)
                    if not np.isnan(tlt_ret) and not np.isnan(hyg_ret):
                        # Positive = risk-off (TLT outperforming HYG)
                        features["tlt_hyg_spread_5d"] = (tlt_ret - hyg_ret) * 100
                    else:
                        features["tlt_hyg_spread_5d"] = np.nan
                else:
                    features["tlt_hyg_spread_5d"] = np.nan

        # ------------------------------------------------------------------
        # Credit spread: HYG vs LQD (high-yield vs investment grade) — P2-8
        # Positive spread = risk appetite, negative = credit stress
        # ------------------------------------------------------------------
        if "HYG" in context_data and "LQD" in context_data:
            hyg_df = context_data["HYG"]
            lqd_df = context_data["LQD"]
            if signal_date in hyg_df.index and signal_date in lqd_df.index:
                hyg_idx = hyg_df.index.get_loc(signal_date)
                lqd_idx = lqd_df.index.get_loc(signal_date)
                for period in (10, 20):
                    h = self._etf_return(hyg_df, hyg_idx, period)
                    l = self._etf_return(lqd_df, lqd_idx, period)
                    if not np.isnan(h) and not np.isnan(l):
                        features[f"hyg_lqd_spread_{period}d"] = (h - l) * 100
                    else:
                        features[f"hyg_lqd_spread_{period}d"] = np.nan
                # HYG percentile (absolute credit conditions)
                if hyg_idx >= 20:
                    window = hyg_df["close"].iloc[max(0, hyg_idx - 252) : hyg_idx]
                    features["hyg_percentile"] = self._percentile_rank(
                        window, hyg_df["close"].iloc[hyg_idx]
                    )
                else:
                    features["hyg_percentile"] = np.nan
        else:
            features["hyg_lqd_spread_10d"] = np.nan
            features["hyg_lqd_spread_20d"] = np.nan
            features["hyg_percentile"] = np.nan

        # ------------------------------------------------------------------
        # Dollar strength: UUP — P2-8
        # Strong dollar = headwind for risk assets / EM / commodities
        # ------------------------------------------------------------------
        if "UUP" in context_data:
            uup_df = context_data["UUP"]
            if signal_date in uup_df.index:
                idx = uup_df.index.get_loc(signal_date)
                ret_10d = self._etf_return(uup_df, idx, 10)
                features["uup_return_10d"] = ret_10d * 100 if not np.isnan(ret_10d) else np.nan
                if idx >= 20:
                    window = uup_df["close"].iloc[max(0, idx - 252) : idx]
                    features["uup_percentile"] = self._percentile_rank(
                        window, uup_df["close"].iloc[idx]
                    )
                else:
                    features["uup_percentile"] = np.nan
        else:
            features["uup_return_10d"] = np.nan
            features["uup_percentile"] = np.nan

        # ------------------------------------------------------------------
        # Growth vs safety: USO / GLD ratio — P2-8
        # Rising USO relative to GLD = cyclical demand growing
        # ------------------------------------------------------------------
        if "USO" in context_data and "GLD" in context_data:
            uso_df = context_data["USO"]
            gld_df = context_data["GLD"]
            if signal_date in uso_df.index and signal_date in gld_df.index:
                uso_idx = uso_df.index.get_loc(signal_date)
                gld_idx = gld_df.index.get_loc(signal_date)
                uso_ret = self._etf_return(uso_df, uso_idx, 20)
                gld_ret = self._etf_return(gld_df, gld_idx, 20)
                if not np.isnan(uso_ret) and not np.isnan(gld_ret):
                    features["uso_gld_ratio_20d"] = (uso_ret - gld_ret) * 100
                else:
                    features["uso_gld_ratio_20d"] = np.nan
                # GLD absolute momentum (flight-to-safety signal)
                gld_ret_10 = self._etf_return(gld_df, gld_idx, 10)
                features["gld_return_10d"] = gld_ret_10 * 100 if not np.isnan(gld_ret_10) else np.nan
        else:
            features["uso_gld_ratio_20d"] = np.nan
            features["gld_return_10d"] = np.nan

        # ------------------------------------------------------------------
        # Risk asset breadth (P2-8)
        # Count of risky assets above their 20-day SMA
        # ------------------------------------------------------------------
        risk_etfs = ["QQQ", "EEM", "HYG", "EFA"]
        def_etfs = ["TLT", "GLD", "XLP", "XLU"]

        def _above_sma20(etf: str) -> int | float:
            if etf not in context_data:
                return np.nan
            df = context_data[etf]
            if signal_date not in df.index:
                return np.nan
            idx = df.index.get_loc(signal_date)
            if idx < 20:
                return np.nan
            sma = df["close"].iloc[idx - 20 : idx].mean()
            return 1 if df["close"].iloc[idx] > sma else 0

        risk_signals = [_above_sma20(e) for e in risk_etfs]
        def_signals = [_above_sma20(e) for e in def_etfs]

        risk_valid = [v for v in risk_signals if not (isinstance(v, float) and np.isnan(v))]
        def_valid = [v for v in def_signals if not (isinstance(v, float) and np.isnan(v))]

        features["risk_breadth"] = sum(risk_valid) / len(risk_valid) if risk_valid else np.nan
        features["defensive_breadth"] = sum(def_valid) / len(def_valid) if def_valid else np.nan
        # Net breadth: +1 → full risk-on, -1 → full risk-off
        if risk_valid and def_valid:
            features["breadth_net"] = features["risk_breadth"] - features["defensive_breadth"]
        else:
            features["breadth_net"] = np.nan

        # ------------------------------------------------------------------
        # Composite regime score (existing — kept for backward compat)
        # ------------------------------------------------------------------
        regime_score = 0
        if "SPY" in context_data:
            spy_df = context_data["SPY"]
            if signal_date in spy_df.index:
                idx = spy_df.index.get_loc(signal_date)
                spy_close = spy_df["close"]
                if idx >= 50:
                    sma50 = spy_close.iloc[idx - 50 : idx + 1].mean()
                    regime_score += 1 if spy_close.iloc[idx] > sma50 else -1
                if idx >= 200:
                    sma200 = spy_close.iloc[idx - 200 : idx + 1].mean()
                    regime_score += 1 if spy_close.iloc[idx] > sma200 else -1
                elif idx >= 50:
                    regime_score += 1 if spy_close.iloc[idx] > sma50 else -1
                if idx >= 20:
                    mom = spy_close.iloc[idx] / spy_close.iloc[idx - 20] - 1
                    regime_score += 1 if mom > 0 else -1
        if "VXX" in context_data:
            vxx_df = context_data["VXX"]
            if signal_date in vxx_df.index:
                vidx = vxx_df.index.get_loc(signal_date)
                if vidx >= 63:
                    vxx_close = vxx_df["close"].iloc[vidx]
                    vxx_avg = vxx_df["close"].iloc[vidx - 63 : vidx].mean()
                    regime_score += 1 if vxx_close < vxx_avg else -1
        if "TLT" in context_data and "HYG" in context_data:
            tlt_df = context_data["TLT"]
            hyg_df = context_data["HYG"]
            if signal_date in tlt_df.index and signal_date in hyg_df.index:
                tidx = tlt_df.index.get_loc(signal_date)
                hidx = hyg_df.index.get_loc(signal_date)
                if tidx >= 10 and hidx >= 10:
                    tlt_10d = self._etf_return(tlt_df, tidx, 10)
                    hyg_10d = self._etf_return(hyg_df, hidx, 10)
                    if not np.isnan(tlt_10d) and not np.isnan(hyg_10d):
                        regime_score += 1 if hyg_10d > tlt_10d else -1

        features["regime_score"] = regime_score
        features["regime_bull"] = 1 if regime_score >= 2 else 0
        features["regime_bear"] = 1 if regime_score <= -2 else 0

        return features

    def _get_seasonality_features(self, signal_date: pd.Timestamp) -> dict:
        """
        Calendar-based seasonality features (P2-8).

        These capture well-documented seasonal patterns:
        - Month-of-year effects
        - Quarter-end window dressing (fund managers buy winners)
        - January effect (small caps)
        - Tax loss harvesting season (Nov-Dec selling; Jan buying)
        """
        features: dict = {}
        month = signal_date.month
        day = signal_date.day

        features["month"] = month

        # Is this within 10 calendar days of quarter end?
        quarter_ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
        is_qtr_end = 0
        for qm, qd in quarter_ends:
            if month == qm and (qd - day) <= 10:
                is_qtr_end = 1
                break
            # Also catch first 5 days of new quarter (post-window-dressing reversal)
            if month in (1, 4, 7, 10) and day <= 5:
                is_qtr_end = 1
                break
        features["is_quarter_end"] = is_qtr_end

        # Tax loss harvesting: Nov-Dec (selling pressure) vs Jan (reversal buying)
        features["is_tax_selling_season"] = 1 if month in (11, 12) else 0
        features["is_january_effect"] = 1 if month == 1 else 0

        # Summer doldrums (Aug-Sep: lower volume, weaker momentum)
        features["is_summer"] = 1 if month in (7, 8, 9) else 0

        return features

    def compute_indicators_for_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to a DataFrame."""
        result = df.copy()

        result["rsi"] = self.rsi(df)

        macd_df = self.macd(df)
        result["macd_line"] = macd_df["macd"]
        result["macd_signal"] = macd_df["signal"]
        result["macd_histogram"] = macd_df["histogram"]

        bb_df = self.bb(df)
        result["bb_upper"] = bb_df["upper"]
        result["bb_lower"] = bb_df["lower"]
        result["bb_pct_b"] = bb_df["pct_b"]

        result["sma_20"] = self.sma_20(df)
        result["sma_50"] = self.sma_50(df)
        result["atr"] = self.atr(df)

        return result

    def create_features_for_signal(
        self,
        ticker_data: pd.DataFrame,
        context_data: dict[str, pd.DataFrame],
        ticker: str,
        signal_date: pd.Timestamp,
        direction: int,
        source_indicators: list[str],
    ) -> pd.Series:
        """
        Create feature vector for a single candidate signal.

        Args:
            ticker_data: OHLCV DataFrame for the stock (with indicators)
            context_data: Dict mapping ETF ticker to OHLCV DataFrame
            ticker: Stock ticker
            signal_date: Date of the signal
            direction: 1 for BUY, -1 for SELL
            source_indicators: List of indicators that fired

        Returns:
            Series with all features
        """
        if signal_date not in ticker_data.index:
            return pd.Series(dtype=float)

        signal_idx = ticker_data.index.get_loc(signal_date)

        features = {}

        # Signal quality features
        features.update(
            self._get_signal_quality_features(
                ticker_data, signal_idx, direction, source_indicators
            )
        )

        # Momentum and volatility features
        features.update(self._get_momentum_volatility_features(ticker_data, signal_idx))

        # Market context features (existing + P2-8 macro/sector)
        features.update(self._get_context_features(context_data, signal_date, ticker))

        # Seasonality features (P2-8)
        features.update(self._get_seasonality_features(signal_date))

        # Earnings risk features (P2-4) — only when calendar is available
        if self.earnings_calendar is not None and self.earnings_calendar.has_data(ticker):
            features["days_to_earnings"] = self.earnings_calendar.days_to_next(
                ticker, signal_date
            )
            features["earnings_within_hold"] = float(
                self.earnings_calendar.within_hold(
                    ticker, signal_date, holding_days=self.holding_period
                )
            )
        else:
            # Neutral sentinel values when no earnings data is available
            features["days_to_earnings"] = 45.0   # assume mid-cycle
            features["earnings_within_hold"] = 0.0

        return pd.Series(features)

    def create_feature_matrix(
        self,
        labeled_df: pd.DataFrame,
        data: dict[str, pd.DataFrame],
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Create feature matrix for all labeled signals.

        Args:
            labeled_df: DataFrame from create_metalabels() with ticker, date, direction, etc.
            data: Dict mapping all tickers to OHLCV DataFrames
            show_progress: Show progress bar

        Returns:
            DataFrame with features, indexed same as labeled_df
        """
        # Precompute indicators for all tickers
        ticker_data_with_indicators = {}
        tickers_to_process = labeled_df["ticker"].unique()

        if show_progress:
            from tqdm import tqdm
            tickers_to_process = tqdm(tickers_to_process, desc="Computing indicators")

        for ticker in tickers_to_process:
            if ticker in data:
                ticker_data_with_indicators[ticker] = self.compute_indicators_for_df(
                    data[ticker]
                )

        # Context data (ETFs)
        context_data = {
            t: data[t] for t in self.context_tickers if t in data
        }

        # Create features for each signal
        feature_rows = []
        iterator = labeled_df.iterrows()
        if show_progress:
            iterator = tqdm(list(iterator), desc="Creating features")

        for idx, row in iterator:
            ticker = row["ticker"]
            signal_date = row["date"]
            direction = row["direction"]
            source_indicators = row["source_indicators"]

            if ticker not in ticker_data_with_indicators:
                feature_rows.append(pd.Series(dtype=float, name=idx))
                continue

            features = self.create_features_for_signal(
                ticker_data_with_indicators[ticker],
                context_data,
                ticker,
                signal_date,
                direction,
                source_indicators,
            )
            features.name = idx
            feature_rows.append(features)

        feature_matrix = pd.DataFrame(feature_rows)

        return feature_matrix
