# Quant Research

Systematic trading research platform for developing, backtesting, and deploying quantitative strategies via Interactive Brokers.

## Overview

This project provides infrastructure for:
- **Data pipelines**: Fetching, cleaning, and feature engineering for market data
- **ML models**: Predictive models for asset returns, volatility, and regime detection
- **Systematic strategies**: Momentum, managed futures, and global macro implementations
- **Live execution**: Order management and position tracking via Interactive Brokers API

## Project Structure

```
quant-research/
├── config/              # Strategy parameters, model configs, API settings
├── data/
│   ├── raw/             # Raw market data (prices, fundamentals, alt data)
│   ├── processed/       # Cleaned and normalized datasets
│   └── features/        # Engineered features for modeling
├── notebooks/           # Research notebooks and analysis
├── src/
│   ├── data_pipelines/  # ETL, data fetching, feature engineering
│   ├── models/          # ML models (return prediction, risk, regime)
│   ├── strategies/      # Strategy logic (signals, portfolio construction)
│   └── execution/       # Order execution, IB integration, risk checks
└── tests/               # Unit and integration tests
```

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

## Strategy Focus

- **Momentum**: Cross-sectional and time-series momentum across asset classes
- **Managed Futures**: Trend-following with dynamic position sizing
- **Global Macro**: Factor-based allocation using economic indicators

## Usage

```python
# Example: Fetch data and generate signals
from src.data_pipelines import fetch_prices
from src.strategies import momentum_signal

prices = fetch_prices(['SPY', 'QQQ', 'TLT'])
signals = momentum_signal(prices, lookback=252)
```

## License

Private research project.
