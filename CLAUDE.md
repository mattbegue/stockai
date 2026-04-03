# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Futures is a stock trading backtesting framework with ML capabilities. It fetches market data via Alpaca API, computes technical indicators, and runs trading strategy backtests.

## Common Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Install with extra ML dependencies (XGBoost, LightGBM)
pip install -e ".[dev,ml-extra]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=futures

# Format code
black futures/

# Lint code
ruff check futures/

# Lint and fix
ruff check --fix futures/

# Collect market data
python -m futures.scripts.collect_data --universe small|medium|large

# Train metalabeling model
python -m futures.scripts.train_metalabeling --universe small|medium|large

# Run backtest
python -m futures.scripts.backtest_metalabeling

# Generate daily trading signals
python -m futures.scripts.daily_signals
```

## Code Style

- Line length: 100 characters
- Python 3.10+ with type hints
- Use `black` for formatting, `ruff` for linting
- Ruff rules: E (errors), F (pyflakes), I (isort), W (warnings)

## Architecture

### Core Modules

- **`backtester/`** - Backtesting engine that runs simulations
  - `engine.py`: `Backtester` class and `BacktestResult` dataclass
  - `portfolio.py`: Portfolio and trade management
  - `metrics.py`: Performance metrics (Sharpe, drawdown, etc.)
  - Supports `max_holding_days` for time-based position exits

- **`strategies/`** - Trading strategy implementations
  - `base.py`: Abstract `Strategy` class, `Signal` enum (BUY/SELL/HOLD), `Position` dataclass
  - Concrete strategies: `sma_crossover.py`, `mean_reversion.py`, `momentum.py`, `ml_strategy.py`
  - `metalabeling_strategy.py`: Two-stage ML-filtered strategy
  - `CompositeStrategy` for combining multiple strategies
  - `FilterStrategy` for screening tickers

- **`metalabeling/`** - Metalabeling strategy components
  - `signals.py`: `PrimarySignalGenerator` with 4 technical indicators (SMA crossover, RSI extremes, MACD crossover, Bollinger Band touches)
  - `labels.py`: `create_metalabels()` for training label generation
  - `features.py`: `MetaFeatureEngineering` with ~30 features (signal quality, momentum, market context, regime). All rolling calculations use prior bars only to prevent lookahead bias.

- **`indicators/`** - Technical indicator computations
  - `base.py`: `Indicator` base class and `IndicatorPipeline`
  - `moving_averages.py`: SMA, EMA, VWAP
  - `momentum.py`: RSI, MACD, Bollinger Bands, ROC, Stochastic, MFI
  - `volatility.py`: ATR, Standard Deviation

- **`ml/`** - Machine learning module
  - `features.py`: `FeatureEngineering` pipeline, `FeatureSet` container
  - `models.py`: Model wrappers (RandomForest, GradientBoosting)
  - `training.py`: `MLTrainer` and `WalkForwardValidator`

- **`data/`** - Data management
  - `fetcher.py`: `AlpacaFetcher` for market data, `DataManager` for multi-ticker loading
  - `storage.py`: SQLite storage for OHLCV data
  - Includes rate limiting (200ms between requests) and retry logic

- **`config/`** - Configuration
  - `settings.py`: `Settings` dataclass, loads from `.env` file via `get_settings()`
  - `universes.py`: `TickerUniverse` dataclass, `get_universe(size)` for small/medium/large universes

- **`scripts/`** - CLI scripts
  - `collect_data.py`: Batch data collection with `--universe` argument
  - `train_metalabeling.py`: Model training with walk-forward validation and 180-day embargo enforcement
  - `backtest_metalabeling.py`: Backtesting with date validation, visualizations, and run archiving
  - `daily_signals.py`: Daily signal generation for live trading
  - `paper_trade.py`: Paper trading tracker with forward metrics and alerts
  - `diagnose_backtest.py`: Survivorship bias and backtest flaw detection
  - `manage_models.py`: List, compare, and switch between trained models (shows embargo dates)

- **`models/`** - Model management
  - `registry.py`: `ModelRegistry` class for versioned model storage
  - `active_model.json`: Tracks which model is currently active

### Key Patterns

**Strategy interface**: All strategies extend `Strategy` ABC and implement:
- `name` property (unique identifier)
- `generate_signals(data: dict[str, pd.DataFrame]) -> dict[str, Signal]`
- Optional: `required_history` (warmup bars), `precompute_indicators(df)`

**Strategies should be stateless** - all state is managed by the backtester.

**DataFrames**: OHLCV data uses lowercase columns (`open`, `high`, `low`, `close`, `volume`) with `DatetimeIndex`.

**Ticker Universes**: Three sizes available via `get_universe("small"|"medium"|"large")`:
- small: ~50 tradeable stocks, 15 context ETFs
- medium: ~150 tradeable stocks, 25 context ETFs
- large: ~332 tradeable stocks, 25 context ETFs (full large-cap coverage)

**Backtest Runs**: Results saved to `runs/<timestamp>/` with config.json, results.json, trades.csv, equity_curve.csv, and visualizations.

**Model Registry**: Models are saved with versioned names like `metalabeling_large_20260129_v1.pkl`. Use `manage_models.py` to list, compare, and switch between models. The active model is tracked in `models/active_model.json`. Models include embargo metadata (`training_end_date`, `embargo_end_date`) for validation.

**Train/Test Separation (Embargo System)**: The training script enforces a 180-day (6-month) embargo period between training data and backtest data to prevent lookahead bias. The backtest script validates that the test period starts after the embargo ends.

## Environment Setup

Copy `.env.example` to `.env` and set Alpaca API credentials:
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

## Example Usage

See `futures/notebooks/quickstart.ipynb` for complete examples of:
- Fetching data with Alpaca API
- Computing technical indicators
- Running backtests
- Training ML strategies with walk-forward validation

## Lookahead Bias Prevention

The framework includes safeguards against lookahead bias:

### Embargo System
- **EMBARGO_DAYS = 180**: Training excludes the most recent 6 months of data
- Training script saves `training_end_date` and `embargo_end_date` in model metadata
- Backtest script validates that test period starts after embargo ends
- Legacy models (without embargo metadata) trigger warnings

### Feature Engineering
- All rolling window calculations use prior bars only (exclusive of current bar)
- Context features (SPY returns, sector momentum, VXX) use prior bar's close
- Prevents subtle lookahead where current bar data leaks into features

### Model Metadata
Models store date boundaries for validation:
```python
{
    "training_end_date": "2025-09-01",   # Last date used for training
    "embargo_end_date": "2026-03-01",    # First valid backtest date
    "embargo_days": 180,                  # Gap between train and test
}
```

### Validation
```bash
# List models with embargo dates
python -m futures.scripts.manage_models

# Output shows:
# Model Name                     Universe  Train End    Embargo      Samples
# metalabeling_large_20260301_v1 large     2025-09-01   2026-03-01    12345 *
```

## Metalabeling Workflow

```bash
# 1. Collect data for large universe
python -m futures.scripts.collect_data --universe large

# 2. Train model (enforces 180-day embargo, saves to models/)
python -m futures.scripts.train_metalabeling --universe large

# 3. Run backtest (validates dates against embargo period)
python -m futures.scripts.backtest_metalabeling

# 4. Diagnose backtest for potential biases
python -m futures.scripts.diagnose_backtest

# 5. Paper trade to validate forward performance (run daily)
python -m futures.scripts.paper_trade --full

# 6. After 3+ months of paper trading, consider live trading
python -m futures.scripts.daily_signals
```

## Paper Trading Tracker

The paper trading system tracks forward performance without risking capital:

```bash
python -m futures.scripts.paper_trade              # Log today's predictions
python -m futures.scripts.paper_trade --check      # Check outcomes (5 days later)
python -m futures.scripts.paper_trade --summary    # View rolling metrics
python -m futures.scripts.paper_trade --full       # Full daily workflow
```

Data stored in `paper_trading/`:
- `predictions.csv`: All logged predictions with timestamps
- `outcomes.csv`: Actual results after holding period
- `forward_metrics.json`: Rolling performance metrics

Alerts trigger when:
- Win rate drops below 50%
- Sharpe ratio goes negative
- Drawdown exceeds 10%
