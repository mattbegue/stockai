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
    ):
        """
        Initialize the feature engineering pipeline.

        Args:
            context_tickers: List of ETF tickers for market context
            lookback_periods: Periods for momentum calculations
        """
        self.context_tickers = context_tickers or [
            "SPY", "QQQ", "VXX", "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY",
            "XLU", "TLT", "HYG", "GLD",
        ]
        self.lookback_periods = lookback_periods

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
        features["has_sma_crossover"] = 1 if "sma_crossover" in source_indicators else 0
        features["has_rsi_signal"] = 1 if any("rsi" in s for s in source_indicators) else 0
        features["has_macd_crossover"] = 1 if "macd_crossover" in source_indicators else 0
        features["has_bb_touch"] = 1 if any("bb" in s for s in source_indicators) else 0

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

        # SPY features (market momentum)
        if "SPY" in context_data:
            spy_df = context_data["SPY"]
            if signal_date in spy_df.index:
                idx = spy_df.index.get_loc(signal_date)

                # SPY returns (use prior bar to avoid lookahead)
                for period in (5, 10, 20):
                    if idx >= period + 1:
                        ret = spy_df["close"].iloc[idx - 1] / spy_df["close"].iloc[idx - 1 - period] - 1
                        features[f"spy_return_{period}d"] = ret * 100
                    else:
                        features[f"spy_return_{period}d"] = np.nan

                # SPY above 50-SMA (trend regime)
                sma50 = spy_df["close"].rolling(50).mean()
                if idx >= 50:
                    features["spy_above_sma50"] = 1 if spy_df["close"].iloc[idx] > sma50.iloc[idx] else 0
                else:
                    features["spy_above_sma50"] = np.nan

        # VXX features (volatility regime)
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
                    features["vxx_percentile"] = (
                        (vxx_values < vxx_close).sum() / len(vxx_values) * 100
                    )
                else:
                    features["vxx_ratio"] = np.nan
                    features["vxx_percentile"] = np.nan

        # Sector ETF momentum (use prior bar to avoid lookahead)
        sector_etf = SECTOR_MAP.get(ticker)
        if sector_etf and sector_etf in context_data:
            sector_df = context_data[sector_etf]
            if signal_date in sector_df.index:
                idx = sector_df.index.get_loc(signal_date)
                if idx >= 6:
                    ret = sector_df["close"].iloc[idx - 1] / sector_df["close"].iloc[idx - 1 - 5] - 1
                    features["sector_return_5d"] = ret * 100
                else:
                    features["sector_return_5d"] = np.nan

        # TLT-HYG spread (risk appetite)
        if "TLT" in context_data and "HYG" in context_data:
            tlt_df = context_data["TLT"]
            hyg_df = context_data["HYG"]
            if signal_date in tlt_df.index and signal_date in hyg_df.index:
                tlt_idx = tlt_df.index.get_loc(signal_date)
                hyg_idx = hyg_df.index.get_loc(signal_date)

                if tlt_idx >= 5 and hyg_idx >= 5:
                    tlt_ret = tlt_df["close"].iloc[tlt_idx] / tlt_df["close"].iloc[tlt_idx - 5] - 1
                    hyg_ret = hyg_df["close"].iloc[hyg_idx] / hyg_df["close"].iloc[hyg_idx - 5] - 1
                    # Positive = risk-off (TLT outperforming HYG)
                    features["tlt_hyg_spread_5d"] = (tlt_ret - hyg_ret) * 100
                else:
                    features["tlt_hyg_spread_5d"] = np.nan

        # Market breadth: % of tradeable universe above 50-SMA
        # (This would require all tradeable data, so we'll skip for now)

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

        # Market context features
        features.update(self._get_context_features(context_data, signal_date, ticker))

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
