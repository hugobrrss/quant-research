# Quant Research Project - Context & Progress

## Project workflow
- Your grill me - you are not a substitution tool, just a super powerful assistant and brainstormer. I'm not looking for comfort.  
- I aim for top-quality, high-end design and being as close as possible to best practices in quant shops. 
- We use Claude code for tedious and *uninteresting* tasks such as setup or config steps
- I work on two machines: principally my personal Macbook Air, but also on my work laptop
- I use GitHub to go from one machine to the next seemlessly (project url is given below)
- I use PyCharm. All the code test/stress-testing is done though Jupyter notebooks in PyCharm
- We write most scripts iteratively: you guide me and question me on the architecture, but I write the code and you correct it. It is an iterative and back and forth process

## Project Overview

Production-grade quantitative finance infrastructure for systematic trading via Interactive Brokers. Built by a PhD economist transitioning into alternative asset management/quant research. The goal is to add a line on my CV that we help me get a job in the quant buy-side industry. 

## Project Architecture
quant-research/
├── config/
│   ├── equity_universes.yaml      # Ticker lists (mag7, russell1000)
│   ├── macro_universes.yaml       # FRED series by category
│   └── russell1000_sectors.csv    # GICS sector/sub-industry mapping
├── data/
│   ├── research/                  # Full history (backfill, backtesting)
│   └── production/                # Rolling 2-year window (daily pipeline)
├── notebooks/exploration/         # Research notebooks
├── src/
│   ├── data_pipelines/
│   │   ├── yahoo_fetcher.py       # Yahoo Finance fetcher (kept for ad-hoc research)
│   │   ├── ib_fetcher.py          # IB historical data fetcher (primary source)
│   │   ├── fred_fetcher.py        # FRED macro data fetcher
│   │   ├── validator.py           # Pipeline-level validation (daily + backfill modes)
│   │   ├── processor.py           # Data cleaning (dedup, ffill, outlier flagging)
│   │   ├── features_equities.py   # Equity features (momentum, vol, volume, mean-rev)
│   │   ├── run_pipeline.py        # Daily pipeline orchestration
│   │   └── run_backfill.py        # One-off historical data fetch
│   ├── models/                    # ML models from Phase 2 (baseline, trees, neural net)
│   ├── strategies/                # Strategy logic (to be built)
│   └── execution/
│       └── ib_connection.py       # IB TWS connection management
├── tests/
├── requirements.txt
└── .env                           # API keys (FRED_API_KEY)

## Completed Work

### Phase 1: Data Pipelines ✓
- Yahoo Finance and IB fetchers with consistent interface (all accept list of tickers/series)
- FRED macro data fetcher with YAML-based universe config
- Pipeline-level validator with daily/backfill modes and ok/warning/critical severity
- Processor: dedup, forward-fill, outlier flagging, return computation
- Features: momentum (multiple horizons + bespoke), volatility, volume, mean reversion
- All operations use groupby('ticker') for multi-ticker DataFrames

### Phase 2: Prediction Models ✓
- Walk-forward validation framework
- Logistic regression, Random Forest, Gradient Boosting, PyTorch MLP
- Key finding: daily single-stock return prediction ~50-53% accuracy — alpha is not in simple predictions

### Phase 3: Live APIs & Infrastructure ✓
- IB connection via ib_insync (paper account DUP678137)
- Generic contract factory supporting STK, FUT, OPT, CASH, BOND, CRYPTO
- IB fetcher with chunked backfill (365-day max per request)
- FRED fetcher with rate limiting and YAML-driven universe selection
- Daily pipeline: auto-detects gap since last fetch, validates, processes, builds features, saves
- Backfill script: chunked fetching, trims to requested date range, saves research + production files
- Pipeline reads universe from config YAML, no hardcoded tickers
- Duplicate-run protection (checks if today's date already in data)
- DigitalOcean droplet created (IP: 104.248.166.125) — server setup pending
- Russell 1000 universe scraped from Wikipedia with GICS sector data (967/1005 tickers fetch successfully from IB)

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary data source | Interactive Brokers | Full account, professional grade, handles fees in paper trading |
| Data format | Parquet (long format, date index, ticker column) | Fast I/O, type preservation, clean cross-sectional operations |
| Pipeline architecture | Separate backfill (manual) + daily pipeline (automated) | Different requirements: one-off vs incremental |
| Validator philosophy | Flags issues; processor fixes unconditionally | Validator = early warning; processor = safety net |
| Feature vs processor | Separate modules | Processor is universal cleaning; features are strategy-specific |
| Config management | YAML for universes, .env for secrets, CSV for sector mapping | Separation of concerns, easy to edit without touching code |
| Fetcher design | One function per module accepting list input | Consistent interface across Yahoo, IB, FRED |
| IB contract creation | Generic Contract() object | Single code path for all instrument types, qualifyContracts validates |
| Server | DigitalOcean $4/month droplet (London) | Always-on for automated daily pipeline + IB Gateway |

## Data Flow
Backfill (one-off):     IB API → validate(backfill) → process → save research + production parquet
Daily (automated):      IB API → validate(daily) → load existing → append → trim 2yr → process → features → save

## Environment

- Python 3.12, virtual environment
- Key packages: pandas, numpy, ib_insync, nest_asyncio, fredapi, pyyaml, python-dotenv, scikit-learn, torch
- Two machines: personal Mac (/Users/hugo/) + work Windows PC (C:\Users\hbourrou\), synced via git
- IB Paper trading: port 7497, account DUP678137
- FRED API key in .env

## IB Notes

- ib_insync requires nest_asyncio in Jupyter notebooks
- Historical data: duration parameter uses trading days for 'D' unit
- Max duration per request: 365 days for daily bars
- Live/delayed quotes require market data subscription (not needed for historical)
- Tickers with dots (BRK.B) use spaces in IB (BRK B)
- YAML parses 'ON' as boolean True — must be quoted

## Current Status & Next Steps

- **Now**: DigitalOcean server setup (IB Gateway + cron), Russell 1000 backfill running
- **Soon**: CRSP data acquisition (25yr history for research), strategy research begins
- **Phase 4**: Momentum strategy (signal generation, portfolio construction, backtesting with transaction costs, paper trading)
- **Later**: Managed futures, systematic global macro, additional macro data APIs (World Bank, IMF, ECB, OECD)