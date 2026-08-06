# Quant Research

An independent, production-grade pipeline and infrastructure for systematic equity
research, built around the Interactive Brokers API and deployed on a cloud server.

This is a personal research project. The data pipeline and supporting
infrastructure are the mature part of the codebase.  
The modelling, strategy, and execution layers are research-stage and currently under development.

## Status
**Complete**
- Data pipeline: automated fetching, validation, cleaning, and feature engineering
  for about 1000-ticker equity universe via the IB API, with a weekly incremental update.
- Infrastructure: headless IB Gateway on a Linux cloud server (systemd-supervised,
  auto-login), a scheduled pipeline timer, a gateway healthcheck with Telegram
  alerting and auto-restart. 
- Prediction models: baseline, tree-based, and neural-net return-prediction models
  (exploratory).

**In progress**
- First strategy: cross-sectional momentum strategy currently at notebook/backtest stage.
- Execution: IB Gateway connection and monitoring in place; order management,
  position tracking, and a live paper-trading layer under development.

**Planned**
- Additional strategies (managed futures, systematic global macro).
- Point-in-time, survivorship-free backtest universe (CRSP-based).

## Project Structure

```
quant-research/
├── config/ # config files, API settings, YAML universes
├── data/
│ ├── research/ # full-history research datasets (gitignored)
│ └── production/ # rolling 2-year production dataset (gitignored)
├── notebooks/ # research and exploration
├── src/
│ ├── data_pipelines/ # fetchers, validators, feature engineering, runners
│ ├── models/ # exploratory return-prediction models
│ ├── strategies/ # strategy logic (in development)
│ └── execution/ # IB connection, gateway healthcheck, notifications
├── tests/
├── requirements.txt
└── .env # secrets, gitignored         
```

## Setup

```bash
pip install -r requirements.txt
```
Running the pipeline requires an Interactive Brokers account and a local `.env`
(credentials and configuration are not tracked).

