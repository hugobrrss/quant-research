# Quant Research

Production-grade quantitative finance infrastructure for systematic trading via Interactive Brokers.

## Overview

This project provides infrastructure for:
- **Data pipelines**: Fetching, cleaning, and feature engineering for market data
- **ML models**: Predictive models for asset returns, volatility, and regime detection
- **Systematic strategies**: Momentum, managed futures, and global macro implementations
- **Live execution**: Order management and position tracking via Interactive Brokers API

## Project Structure

```
quant-research/
├── config/                  # confi files, API settings, YAML universe
├── data/
│   ├── research/            # full daily data sit here
│   └── production/          # the dynamic production file containing the last 3 years
├── notebooks/exploration/   # Research notebooks and analysis
├── src/
│   ├── data_pipelines/      # data fetching scripts, validators, features construction
│   ├── models/              # ML and other models
│   ├── strategies/          # Strategy logic (signals, portfolio construction)
│   └── execution/           # Order execution, IB integration
├── tests/
├── requirements.txt
└── .env              
```

## Strategy Focus

- **Momentum**: Cross-sectional and time-series equity momentum 
- **Managed Futures**: Trend-following with dynamic position sizing
- **Systematic Global Macro**: Factor-based allocation using economic indicators


## Disclosure

Private research project.
