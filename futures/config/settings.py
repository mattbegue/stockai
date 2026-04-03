"""Application settings and configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Alpaca API credentials
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    alpaca_paper: bool = True  # Use paper trading endpoint

    # Database settings
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("DB_PATH", "data/futures.db"))
    )

    # Trading settings
    default_cash: float = 100_000.0
    transaction_cost_pct: float = 0.001  # 0.1% per trade (10 bps)
    slippage_pct: float = 0.0005  # 0.05% slippage

    # Data settings
    default_lookback_days: int = 365 * 5  # 5 years of history by default

    def __post_init__(self):
        """Validate settings after initialization."""
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            import warnings
            warnings.warn(
                "Alpaca API credentials not set. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "environment variables or create a .env file."
            )

    @property
    def alpaca_base_url(self) -> str:
        """Get the appropriate Alpaca base URL."""
        if self.alpaca_paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"


# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
