# Futures

A stock trading backtesting framework with ML capabilities. Fetches market data via Alpaca API, computes technical indicators, and runs trading strategy backtests. 

## Features

- **Data Management**: Fetch and cache OHLCV data from Alpaca API with SQLite storage
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and more
- **Backtesting Engine**: Simulate trading strategies with realistic transaction costs and slippage
- **ML Strategies**: Train machine learning models with walk-forward validation
- **Metalabeling Strategy**: Two-stage strategy using technical indicators filtered by ML model
- **Lookahead Bias Prevention**: Enforced train/test separation with 180-day embargo period

## Installation

```bash
# Install dependencies
pip install -e ".[dev]"

# Install with extra ML dependencies (XGBoost, LightGBM)
pip install -e ".[dev,ml-extra]"
```

## Environment Setup

Copy `.env.example` to `.env` and set your Alpaca API credentials:

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

## Quick Start

### Basic Backtesting

```python
from futures.data.fetcher import DataManager
from futures.strategies import SMACrossover
from futures.backtester import Backtester

# Fetch data
dm = DataManager()
data = dm.get_multi(["AAPL", "MSFT", "GOOGL"])

# Create strategy and run backtest
strategy = SMACrossover(fast_period=20, slow_period=50)
bt = Backtester(strategy=strategy, initial_cash=100_000)
result = bt.run(data)

print(result.summary())
```

## Metalabeling Strategy

The metalabeling strategy uses a two-stage approach:
1. **Primary Signals**: Ensemble of technical indicators (SMA crossover, RSI extremes, MACD crossover, Bollinger Band touches) generates trade candidates
2. **Meta-Model**: ML classifier filters candidates based on predicted profitability

### Training and Backtesting

**Step 1: Collect data for your chosen universe**

```bash
# Small universe (~50 stocks) - fast, good for testing
python -m futures.scripts.collect_data --universe small

# Medium universe (~150 stocks) - balanced
python -m futures.scripts.collect_data --universe medium

# Large universe (~300 stocks) - comprehensive coverage
python -m futures.scripts.collect_data --universe large
```

**Step 2: Train the metalabeling model**

```bash
python -m futures.scripts.train_metalabeling --universe large
```

This trains with walk-forward validation and enforces a **180-day embargo period** to prevent lookahead bias. The model is saved to `models/` with versioned naming (e.g., `metalabeling_large_20260301_v1.pkl`).

**Step 3: Run backtest**

```bash
python -m futures.scripts.backtest_metalabeling
```

The backtest automatically validates that the test period starts **after the embargo period** to ensure proper train/test separation. It uses the same universe the model was trained on. Results are saved to `runs/<timestamp>/` including:
- `config.json` - Configuration for reproducibility
- `results.json` - Performance metrics
- `trades.csv` / `trades.xlsx` - All trades with P&L
- `equity_curve.csv` - Daily portfolio values
- Visualizations (equity curves, drawdown, monthly returns, etc.)

### Daily Trading Signals

Generate daily trading signals for live trading:

```bash
python -m futures.scripts.daily_signals
```

This outputs:
- **EXIT** signals for positions hitting time-based (5 days) or signal-based exits
- **ENTRY** signals for new high-confidence trades
- **HOLD** signals for current positions

Position tracking is stored in `positions.json`.

### Paper Trading (Forward Testing)

Before risking real capital, validate the strategy with paper trading:

```bash
# Daily workflow (run once per day)
python -m futures.scripts.paper_trade --full

# Or run steps separately:
python -m futures.scripts.paper_trade              # Log today's predictions
python -m futures.scripts.paper_trade --check      # Check 5-day-old outcomes
python -m futures.scripts.paper_trade --summary    # View performance metrics
```

This tracks:
- Every prediction with timestamp and confidence
- Actual outcomes after the 5-day holding period
- Rolling forward metrics (win rate, Sharpe, drawdown)
- Alerts when performance deviates from expectations

Data is stored in `paper_trading/` directory.

### Backtest Diagnostics

After backtesting, check for survivorship bias and other issues:

```bash
python -m futures.scripts.diagnose_backtest
```

### Model Management

Multiple models can coexist with versioned names:

```bash
# List all models (shows embargo dates)
python -m futures.scripts.manage_models list

# Set active model
python -m futures.scripts.manage_models set metalabeling_large_20260129_v1

# View model details
python -m futures.scripts.manage_models info metalabeling_large_20260129_v1

# Compare two models
python -m futures.scripts.manage_models compare model1 model2
```

Models are automatically named: `metalabeling_{universe}_{date}_v{version}.pkl`

Each model stores metadata including:
- `training_end_date`: Last date used for training
- `embargo_end_date`: First valid date for backtesting
- `embargo_days`: Gap between training and test (default: 180 days)

### Lookahead Bias Prevention

The framework includes safeguards to prevent lookahead bias, which can cause inflated backtest results that don't translate to live trading:

**Embargo System**: Training automatically excludes the most recent 180 days of data. This ensures the model never "sees" data from the backtest period during training.

**Date Validation**: The backtest script validates that the test period starts after the embargo ends. If dates overlap, it automatically adjusts and warns you.

**Feature Engineering**: All rolling window calculations (volatility, volume ratios, drawdowns, market context) use prior bars only, excluding the current bar to prevent subtle data leakage.

**Legacy Model Warnings**: Models trained before the embargo system was implemented will show warnings during backtesting, prompting you to retrain.

## Ticker Universes

Three pre-configured universes are available:

| Size   | Tradeable | Context ETFs | Description |
|--------|-----------|--------------|-------------|
| small  | 50        | 15           | Original universe, fast training |
| medium | 150       | 25           | Top-tier S&P 500, balanced |
| large  | 332       | 25           | Full large-cap coverage |

Context ETFs include broad market (SPY, QQQ), sectors (XLF, XLK, etc.), fixed income (TLT, HYG), international (EEM, EFA), and commodities (GLD, USO).

```python
from futures.config import get_universe, list_universes

# See available universes
print(list_universes())

# Get a specific universe
universe = get_universe("large")
print(f"Tradeable: {len(universe.tradeable)}")
print(f"Context: {len(universe.context)}")
```

## Project Structure

```
futures/
├── backtester/       # Backtesting engine
├── config/           # Settings and universe definitions
├── data/             # Data fetching and storage
├── indicators/       # Technical indicators
├── metalabeling/     # Metalabeling strategy (features, signals, labels)
├── ml/               # Machine learning utilities
├── models/           # Trained models with embargo metadata
├── scripts/          # CLI scripts
└── strategies/       # Trading strategy implementations
```

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=futures

# Format code
black futures/

# Lint code
ruff check futures/
```

## License

MIT
