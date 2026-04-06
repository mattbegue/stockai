"""Ticker universe definitions for trading strategies."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TickerUniverse:
    """A universe of tickers for trading and context."""

    tradeable: list[str]  # Stocks to generate signals on
    context: list[str]  # ETFs/indices for features only (no trading)
    name: str = "default"

    @property
    def all_tickers(self) -> list[str]:
        """Get all tickers (tradeable + context)."""
        return self.tradeable + self.context

    def __repr__(self) -> str:
        return f"TickerUniverse({self.name}: {len(self.tradeable)} tradeable, {len(self.context)} context)"


def get_default_universe() -> TickerUniverse:
    """
    Get the default ticker universe (small, 50 stocks).
    For backward compatibility.
    """
    return get_universe("small")


def get_universe(size: Literal["small", "medium", "large"] = "medium") -> TickerUniverse:
    """
    Get a ticker universe by size.

    Args:
        size: "small" (~50 stocks), "medium" (~150 stocks), "large" (~300 stocks)

    Returns:
        TickerUniverse with tradeable stocks and context ETFs
    """
    if size == "small":
        return _get_small_universe()
    elif size == "medium":
        return _get_medium_universe()
    elif size == "large":
        return _get_large_universe()
    else:
        raise ValueError(f"Unknown universe size: {size}. Use 'small', 'medium', or 'large'")


def _get_small_universe() -> TickerUniverse:
    """Original 50-stock universe."""
    tradeable = [
        # Technology (10)
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "AMD", "INTC", "CRM",
        # Financials (10)
        "JPM", "BAC", "GS", "MS", "V",
        "MA", "BRK.B", "C", "WFC", "AXP",
        # Healthcare (10)
        "JNJ", "UNH", "PFE", "MRK", "ABBV",
        "LLY", "TMO", "ABT", "BMY", "AMGN",
        # Consumer (10)
        "WMT", "PG", "KO", "PEP", "COST",
        "HD", "MCD", "NKE", "SBUX", "DIS",
        # Industrials/Energy (10)
        "XOM", "CVX", "CAT", "BA", "UPS",
        "HON", "GE", "LMT", "RTX", "DE",
    ]

    context = _get_base_context()

    return TickerUniverse(tradeable=tradeable, context=context, name="small")


def _get_medium_universe() -> TickerUniverse:
    """~150 stock universe - top tier S&P 500."""
    tradeable = [
        # Technology (25)
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC",
        "CRM", "ORCL", "ADBE", "CSCO", "AVGO", "ACN", "TXN", "QCOM", "IBM", "NOW",
        "INTU", "AMAT", "MU", "ADI", "LRCX",
        # Financials (20)
        "JPM", "BAC", "GS", "MS", "V", "MA", "BRK.B", "C", "WFC", "AXP",
        "BLK", "SCHW", "MMC", "CB", "PGR", "ICE", "CME", "AON", "USB", "PNC",
        # Healthcare (25)
        "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "BMY", "AMGN",
        "MDT", "DHR", "ISRG", "SYK", "GILD", "VRTX", "REGN", "ZTS", "BDX", "CI",
        "ELV", "HUM", "CVS", "MCK", "HCA",
        # Consumer Discretionary (15)
        "HD", "MCD", "NKE", "SBUX", "DIS", "LOW", "TJX", "BKNG", "CMG", "MAR",
        "ORLY", "AZO", "ROST", "YUM", "DHI",
        # Consumer Staples (15)
        "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ", "CL", "KMB",
        "GIS", "K", "SYY", "STZ", "KHC",
        # Industrials (20)
        "CAT", "BA", "UPS", "HON", "GE", "LMT", "RTX", "DE", "UNP", "FDX",
        "WM", "ETN", "ITW", "EMR", "NSC", "CSX", "GD", "NOC", "MMM", "JCI",
        # Energy (10)
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD",
        # Materials (8)
        "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "DOW",
        # Utilities (6)
        "NEE", "DUK", "SO", "D", "AEP", "EXC",
        # REITs (6)
        "PLD", "AMT", "EQIX", "CCI", "PSA", "SPG",
    ]

    context = _get_expanded_context()

    return TickerUniverse(tradeable=tradeable, context=context, name="medium")


def _get_large_universe() -> TickerUniverse:
    """~300 stock universe - full large-cap coverage."""
    tradeable = [
        # =========================================================================
        # TECHNOLOGY (50)
        # =========================================================================
        # Mega-cap tech
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
        # Semiconductors
        "AMD", "INTC", "AVGO", "TXN", "QCOM", "MU", "ADI", "LRCX", "AMAT", "KLAC",
        "MCHP", "NXPI", "ON", "SWKS", "MRVL", "MPWR",
        # Software
        "CRM", "ORCL", "ADBE", "NOW", "INTU", "SNPS", "CDNS", "ANSS", "WDAY", "TEAM",
        "DDOG", "ZS", "CRWD", "PANW", "FTNT", "SPLK",
        # IT Services & Hardware
        "CSCO", "ACN", "IBM", "HPQ", "HPE", "DELL", "NTAP", "WDC", "STX",
        # Internet & Interactive Media
        "NFLX", "ABNB", "UBER", "LYFT", "SNAP", "PINS", "MTCH",

        # =========================================================================
        # FINANCIALS (40)
        # =========================================================================
        # Banks
        "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF",
        "FITB", "KEY", "RF", "CFG", "HBAN", "MTB", "ZION",
        # Payment processors
        "V", "MA", "AXP", "PYPL", "SQ", "FIS", "FISV", "GPN",
        # Insurance
        "BRK.B", "PGR", "CB", "MMC", "AON", "AJG", "TRV", "ALL", "MET", "AFL",
        # Asset Management & Exchanges
        "BLK", "SCHW", "ICE", "CME", "SPGI", "MCO", "MSCI", "NDAQ",

        # =========================================================================
        # HEALTHCARE (45)
        # =========================================================================
        # Pharma
        "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "GILD", "VRTX", "REGN", "BIIB",
        "MRNA", "ZTS", "TAK",
        # Healthcare Equipment
        "ABT", "MDT", "DHR", "SYK", "ISRG", "BDX", "BSX", "EW", "BAX", "HOLX",
        "IDXX", "ALGN", "DXCM", "TFX",
        # Healthcare Services
        "UNH", "ELV", "CI", "HUM", "CNC", "CVS", "MCK", "CAH", "ABC",
        # Life Sciences & Biotech
        "TMO", "AMGN", "IQV", "A", "MTD", "WAT", "PKI", "TECH", "BIO",
        # Healthcare Facilities
        "HCA", "THC", "UHS", "DVA",

        # =========================================================================
        # CONSUMER DISCRETIONARY (35)
        # =========================================================================
        # Retail
        "HD", "LOW", "TJX", "ROST", "ORLY", "AZO", "BBY", "ULTA", "DG", "DLTR",
        "KMX", "AN", "GPC", "AAP",
        # Restaurants & Leisure
        "MCD", "SBUX", "CMG", "YUM", "DRI", "WYNN", "LVS", "MGM",
        # Hotels & Travel
        "MAR", "HLT", "BKNG", "EXPE", "CCL", "RCL", "NCLH",
        # Apparel & Luxury
        "NKE", "LULU", "TPR", "VFC", "PVH", "RL",
        # Entertainment
        "DIS", "CMCSA", "CHTR", "NWSA",

        # =========================================================================
        # CONSUMER STAPLES (25)
        # =========================================================================
        # Food & Beverage
        "KO", "PEP", "MDLZ", "GIS", "K", "KHC", "SJM", "HSY", "MKC", "CPB",
        "CAG", "HRL", "TSN", "BG",
        # Household Products
        "PG", "CL", "KMB", "CHD", "CLX",
        # Retail
        "WMT", "COST", "TGT", "KR", "SYY",
        # Tobacco & Alcohol
        "PM", "MO", "STZ", "BF.B",

        # =========================================================================
        # INDUSTRIALS (40)
        # =========================================================================
        # Aerospace & Defense
        "BA", "LMT", "RTX", "GD", "NOC", "HII", "TDG", "HWM", "LHX",
        # Machinery & Equipment
        "CAT", "DE", "EMR", "ETN", "ROK", "PH", "ITW", "IR", "DOV", "SWK",
        "CMI", "PCAR", "GNRC",
        # Transportation
        "UPS", "FDX", "UNP", "NSC", "CSX", "DAL", "UAL", "LUV", "AAL",
        "JBHT", "CHRW", "EXPD",
        # Industrial Conglomerates
        "HON", "GE", "MMM", "JCI",
        # Building Products & Services
        "JCI", "CARR", "TT", "LII", "AOS", "MAS",
        # Waste Management
        "WM", "RSG", "WCN",

        # =========================================================================
        # ENERGY (20)
        # =========================================================================
        # Integrated Oil & Gas
        "XOM", "CVX", "COP", "OXY", "HES",
        # Exploration & Production
        "EOG", "PXD", "DVN", "FANG", "MRO", "APA",
        # Refining & Marketing
        "MPC", "PSX", "VLO",
        # Oil Services
        "SLB", "HAL", "BKR",
        # Midstream
        "WMB", "KMI", "OKE",

        # =========================================================================
        # MATERIALS (15)
        # =========================================================================
        # Chemicals
        "LIN", "APD", "SHW", "ECL", "PPG", "DD", "DOW", "LYB", "ALB", "CE",
        # Metals & Mining
        "NEM", "FCX", "NUE", "STLD", "CLF",

        # =========================================================================
        # UTILITIES (12)
        # =========================================================================
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ES",
        "AWK", "ED",

        # =========================================================================
        # REAL ESTATE (15)
        # =========================================================================
        # Data Centers & Towers
        "AMT", "CCI", "EQIX", "DLR",
        # Industrial & Logistics
        "PLD", "WELL", "AVB", "EQR",
        # Retail & Office
        "SPG", "O", "VICI", "PSA",
        # Residential
        "INVH", "MAA", "UDR",

        # =========================================================================
        # COMMUNICATION SERVICES (15)
        # =========================================================================
        # Telecom
        "T", "VZ", "TMUS",
        # Media & Entertainment
        "DIS", "CMCSA", "NFLX", "WBD", "PARA", "FOX",
        # Interactive Media
        "GOOGL", "META",  # Already in tech, but primary business is here
        # Gaming
        "EA", "TTWO", "ATVI",
    ]

    # Remove duplicates while preserving order
    seen = set()
    tradeable = [x for x in tradeable if not (x in seen or seen.add(x))]

    context = _get_expanded_context()

    return TickerUniverse(tradeable=tradeable, context=context, name="large")


def _get_base_context() -> list[str]:
    """Base context ETFs (15 tickers)."""
    return [
        # Broad market ETFs
        "SPY",  # S&P 500
        "QQQ",  # Nasdaq 100
        "DIA",  # Dow Jones
        "IWM",  # Russell 2000
        # Volatility
        "VXX",  # VIX short-term futures
        # Sector ETFs
        "XLF",  # Financials
        "XLK",  # Technology
        "XLE",  # Energy
        "XLV",  # Healthcare
        "XLI",  # Industrials
        "XLP",  # Consumer Staples
        "XLY",  # Consumer Discretionary
        # Fixed income and alternatives
        "TLT",  # Long-term Treasuries
        "HYG",  # High Yield Corporate Bonds
        "GLD",  # Gold
    ]


def _get_expanded_context() -> list[str]:
    """Expanded context ETFs with international & alternatives (25 tickers)."""
    return [
        # -------------------------------------------------------------------------
        # US Broad Market (4)
        # -------------------------------------------------------------------------
        "SPY",  # S&P 500
        "QQQ",  # Nasdaq 100
        "DIA",  # Dow Jones
        "IWM",  # Russell 2000 (small cap)

        # -------------------------------------------------------------------------
        # Volatility (1)
        # -------------------------------------------------------------------------
        "VXX",  # VIX short-term futures

        # -------------------------------------------------------------------------
        # US Sector ETFs (8)
        # -------------------------------------------------------------------------
        "XLF",  # Financials
        "XLK",  # Technology
        "XLE",  # Energy
        "XLV",  # Healthcare
        "XLI",  # Industrials
        "XLP",  # Consumer Staples
        "XLY",  # Consumer Discretionary
        "XLU",  # Utilities

        # -------------------------------------------------------------------------
        # Fixed Income (3)
        # -------------------------------------------------------------------------
        "TLT",  # Long-term Treasuries (20+ yr)
        "IEF",  # Intermediate Treasuries (7-10 yr)
        "HYG",  # High Yield Corporate Bonds

        # -------------------------------------------------------------------------
        # International (4)
        # -------------------------------------------------------------------------
        "EFA",  # Developed Markets (Europe, Japan, Australia)
        "EEM",  # Emerging Markets
        "FXI",  # China Large-Cap
        "EWJ",  # Japan

        # -------------------------------------------------------------------------
        # Commodities & Alternatives (4)
        # -------------------------------------------------------------------------
        "GLD",  # Gold
        "SLV",  # Silver
        "USO",  # Oil
        "UUP",  # US Dollar Index

        # -------------------------------------------------------------------------
        # Credit & Risk Appetite (1)
        # -------------------------------------------------------------------------
        "LQD",  # Investment Grade Corporate Bonds
    ]


# Convenience function to list available universes
def list_universes() -> dict:
    """List available universe configurations."""
    return {
        "small": "~50 stocks - Original universe, fast training",
        "medium": "~150 stocks - Top tier S&P 500, balanced",
        "large": "~300 stocks - Full large-cap coverage",
    }


# Sector classification for position limit enforcement (P2-6).
# Covers all tickers in the small + medium + large universes.
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "AMZN": "Technology", "META": "Technology", "NVDA": "Technology", "TSLA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "CRM": "Technology", "ORCL": "Technology",
    "ADBE": "Technology", "CSCO": "Technology", "AVGO": "Technology", "ACN": "Technology",
    "TXN": "Technology", "QCOM": "Technology", "IBM": "Technology", "NOW": "Technology",
    "INTU": "Technology", "AMAT": "Technology", "MU": "Technology", "ADI": "Technology",
    "LRCX": "Technology", "KLAC": "Technology", "MCHP": "Technology", "NXPI": "Technology",
    "ON": "Technology", "SWKS": "Technology", "MRVL": "Technology", "MPWR": "Technology",
    "SNPS": "Technology", "CDNS": "Technology", "ANSS": "Technology", "WDAY": "Technology",
    "TEAM": "Technology", "DDOG": "Technology", "ZS": "Technology", "CRWD": "Technology",
    "PANW": "Technology", "FTNT": "Technology", "SPLK": "Technology", "HPQ": "Technology",
    "HPE": "Technology", "DELL": "Technology", "NTAP": "Technology", "WDC": "Technology",
    "STX": "Technology", "NFLX": "Technology", "ABNB": "Technology", "UBER": "Technology",
    "LYFT": "Technology", "SNAP": "Technology", "PINS": "Technology", "MTCH": "Technology",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "MS": "Financials",
    "V": "Financials", "MA": "Financials", "BRK.B": "Financials", "C": "Financials",
    "WFC": "Financials", "AXP": "Financials", "BLK": "Financials", "SCHW": "Financials",
    "MMC": "Financials", "CB": "Financials", "PGR": "Financials", "ICE": "Financials",
    "CME": "Financials", "AON": "Financials", "USB": "Financials", "PNC": "Financials",
    "TFC": "Financials", "COF": "Financials", "FITB": "Financials", "KEY": "Financials",
    "RF": "Financials", "CFG": "Financials", "HBAN": "Financials", "MTB": "Financials",
    "ZION": "Financials", "PYPL": "Financials", "SQ": "Financials", "FIS": "Financials",
    "FISV": "Financials", "GPN": "Financials", "AJG": "Financials", "TRV": "Financials",
    "ALL": "Financials", "MET": "Financials", "AFL": "Financials", "SPGI": "Financials",
    "MCO": "Financials", "MSCI": "Financials", "NDAQ": "Financials",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare",
    "ABBV": "Healthcare", "LLY": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare", "MDT": "Healthcare", "DHR": "Healthcare",
    "ISRG": "Healthcare", "SYK": "Healthcare", "GILD": "Healthcare", "VRTX": "Healthcare",
    "REGN": "Healthcare", "ZTS": "Healthcare", "BDX": "Healthcare", "CI": "Healthcare",
    "ELV": "Healthcare", "HUM": "Healthcare", "CVS": "Healthcare", "MCK": "Healthcare",
    "HCA": "Healthcare", "BIIB": "Healthcare", "MRNA": "Healthcare", "TAK": "Healthcare",
    "BSX": "Healthcare", "EW": "Healthcare", "BAX": "Healthcare", "HOLX": "Healthcare",
    "IDXX": "Healthcare", "ALGN": "Healthcare", "DXCM": "Healthcare", "TFX": "Healthcare",
    "CNC": "Healthcare", "CAH": "Healthcare", "ABC": "Healthcare", "IQV": "Healthcare",
    "A": "Healthcare", "MTD": "Healthcare", "WAT": "Healthcare", "PKI": "Healthcare",
    "TECH": "Healthcare", "BIO": "Healthcare", "THC": "Healthcare", "UHS": "Healthcare",
    "DVA": "Healthcare",
    # Consumer Discretionary
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "DIS": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "TJX": "Consumer Discretionary", "BKNG": "Consumer Discretionary", "CMG": "Consumer Discretionary",
    "MAR": "Consumer Discretionary", "ORLY": "Consumer Discretionary", "AZO": "Consumer Discretionary",
    "ROST": "Consumer Discretionary", "YUM": "Consumer Discretionary", "DHI": "Consumer Discretionary",
    "BBY": "Consumer Discretionary", "ULTA": "Consumer Discretionary", "DG": "Consumer Discretionary",
    "DLTR": "Consumer Discretionary", "KMX": "Consumer Discretionary", "AN": "Consumer Discretionary",
    "GPC": "Consumer Discretionary", "AAP": "Consumer Discretionary", "DRI": "Consumer Discretionary",
    "WYNN": "Consumer Discretionary", "LVS": "Consumer Discretionary", "MGM": "Consumer Discretionary",
    "HLT": "Consumer Discretionary", "EXPE": "Consumer Discretionary", "CCL": "Consumer Discretionary",
    "RCL": "Consumer Discretionary", "NCLH": "Consumer Discretionary", "LULU": "Consumer Discretionary",
    "TPR": "Consumer Discretionary", "VFC": "Consumer Discretionary", "PVH": "Consumer Discretionary",
    "RL": "Consumer Discretionary", "CMCSA": "Consumer Discretionary", "CHTR": "Consumer Discretionary",
    "NWSA": "Consumer Discretionary",
    # Consumer Staples
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "MDLZ": "Consumer Staples", "CL": "Consumer Staples",
    "KMB": "Consumer Staples", "GIS": "Consumer Staples", "K": "Consumer Staples",
    "SYY": "Consumer Staples", "STZ": "Consumer Staples", "KHC": "Consumer Staples",
    "SJM": "Consumer Staples", "HSY": "Consumer Staples", "MKC": "Consumer Staples",
    "CPB": "Consumer Staples", "CAG": "Consumer Staples", "HRL": "Consumer Staples",
    "TSN": "Consumer Staples", "BG": "Consumer Staples", "CHD": "Consumer Staples",
    "CLX": "Consumer Staples", "TGT": "Consumer Staples", "KR": "Consumer Staples",
    "BF.B": "Consumer Staples",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "UPS": "Industrials", "HON": "Industrials",
    "GE": "Industrials", "LMT": "Industrials", "RTX": "Industrials", "DE": "Industrials",
    "UNP": "Industrials", "FDX": "Industrials", "WM": "Industrials", "ETN": "Industrials",
    "ITW": "Industrials", "EMR": "Industrials", "NSC": "Industrials", "CSX": "Industrials",
    "GD": "Industrials", "NOC": "Industrials", "MMM": "Industrials", "JCI": "Industrials",
    "HII": "Industrials", "TDG": "Industrials", "HWM": "Industrials", "LHX": "Industrials",
    "ROK": "Industrials", "PH": "Industrials", "IR": "Industrials", "DOV": "Industrials",
    "SWK": "Industrials", "CMI": "Industrials", "PCAR": "Industrials", "GNRC": "Industrials",
    "DAL": "Industrials", "UAL": "Industrials", "LUV": "Industrials", "AAL": "Industrials",
    "JBHT": "Industrials", "CHRW": "Industrials", "EXPD": "Industrials", "CARR": "Industrials",
    "TT": "Industrials", "LII": "Industrials", "AOS": "Industrials", "MAS": "Industrials",
    "RSG": "Industrials", "WCN": "Industrials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy",
    "MPC": "Energy", "PSX": "Energy", "VLO": "Energy", "OXY": "Energy", "PXD": "Energy",
    "HES": "Energy", "DVN": "Energy", "FANG": "Energy", "MRO": "Energy", "APA": "Energy",
    "HAL": "Energy", "BKR": "Energy", "WMB": "Energy", "KMI": "Energy", "OKE": "Energy",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials", "ECL": "Materials",
    "NEM": "Materials", "FCX": "Materials", "NUE": "Materials", "DOW": "Materials",
    "PPG": "Materials", "DD": "Materials", "LYB": "Materials", "ALB": "Materials",
    "CE": "Materials", "STLD": "Materials", "CLF": "Materials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    "AEP": "Utilities", "EXC": "Utilities", "SRE": "Utilities", "XEL": "Utilities",
    "WEC": "Utilities", "ES": "Utilities", "AWK": "Utilities", "ED": "Utilities",
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "EQIX": "Real Estate", "CCI": "Real Estate",
    "PSA": "Real Estate", "SPG": "Real Estate", "DLR": "Real Estate", "WELL": "Real Estate",
    "AVB": "Real Estate", "EQR": "Real Estate", "O": "Real Estate", "VICI": "Real Estate",
    "INVH": "Real Estate", "MAA": "Real Estate", "UDR": "Real Estate",
    # Communication Services
    "T": "Communication Services", "VZ": "Communication Services", "TMUS": "Communication Services",
    "WBD": "Communication Services", "PARA": "Communication Services", "FOX": "Communication Services",
    "EA": "Communication Services", "TTWO": "Communication Services", "ATVI": "Communication Services",
}


def get_sector_map(tickers: list[str] | None = None) -> dict[str, str]:
    """
    Return the sector mapping for the given tickers.

    Args:
        tickers: List of tickers to include. None returns the full map.

    Returns:
        Dict mapping ticker → GICS sector name. Tickers not in the map
        are assigned "Unknown".
    """
    if tickers is None:
        return dict(SECTOR_MAP)
    return {t: SECTOR_MAP.get(t, "Unknown") for t in tickers}
