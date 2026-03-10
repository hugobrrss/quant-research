""""
Backfill script: fetches an initial research dataset
This script is designed as a one-off task, to be run manually when starting a new universe or refreshing a research dataset
"""
import yaml
import logging
import pandas as pd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from src.data_pipelines.validator import validate_pipeline_data
from src.data_pipelines.processor import process_pipeline_data
from src.execution.ib_connection import IBConnection
from src.data_pipelines.ib_fetcher import fetch_historical_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_backfill(universe: str, start_date: str, end_date: str) -> None:
    """
    Fetch a research dataset.
    Steps are:
        1. fetch raw price data
        2. validate data
        3. process data
        4. save to disk

    Arguments:
        universe: name of the equity universe in the config YAML file
        start_date: start date for backfill, format YYYY-MM-DD
        end_date: end date for backfill, format YYYY-MM-DD

    Notes: IB's API has a max duration of one year for daily bars. For data longer than one year, request must be chunked in
           multiple calls of at most one year each.
    """

    project_root = Path(__file__).parent.parent.parent  # this assumes src/data_pipelines
    data_research = project_root / "data" / "research" / f"{universe}_research.parquet"
    data_production = project_root / "data" / "production" / f"{universe}_production.parquet"

    config_path = project_root / "config" / "equity_universes.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if universe not in config:
        raise ValueError(f"Universe '{universe}' not found in {config_path}")

    tickers = config[universe]

    conn = IBConnection()
    try:
        conn.connect()
        contract_specs = []
        for ticker in tickers:
            ticker_dic = {'sec_type': 'STK', 'symbol': ticker}
            contract_specs.append(ticker_dic)

        # deal with the start and end dates so it works with how the fetcher is defined
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date)

        all_data = []
        chunk_end = e
        while chunk_end > s:
            chunk_start = max(s, chunk_end - relativedelta(days=365))
            duration = (chunk_end - chunk_start).days
            chunk = fetch_historical_data(conn, contract_specs, end_date=chunk_end, duration=f'{duration} D')
            all_data.append(chunk)
            logger.info(f"Fetched chunk ending {chunk_end}, got {len(chunk)} rows")
            chunk_end = chunk_start

        data_raw = pd.concat(all_data, ignore_index=False)
        data_raw = data_raw.drop_duplicates()
        data_raw = data_raw[data_raw.index >= pd.Timestamp(start_date)]
        data_raw = data_raw[data_raw.index <= pd.Timestamp(end_date)]

        # The procedure fills "backwards", so sort on tickers and index (date)
        data_raw = data_raw.sort_index()
        data_raw = data_raw.sort_values(['ticker'])

    finally:
        conn.disconnect()


    # Validate raw data
    validation_dict = validate_pipeline_data(data_raw, tickers, "backfill")

    if validation_dict['critical']:
        # instructions to stop the pipeline and obtain an alert
        logger.critical(f"Critical issue(s) with the fetched data")
        for msg in validation_dict['critical']:
            print(msg)

        # here something to obtain email and SMS alerts
        return

    # Process data
    df_processed = process_pipeline_data(data_raw)

    # Save research file (full sample)
    df_processed.to_parquet(data_research)

    # Save production file (only one year needed)
    cutoff = pd.Timestamp(end_date) - pd.DateOffset(years=2)
    df_production = df_processed[df_processed.index >= cutoff]
    df_production.to_parquet(data_production)

    logger.info(f"Research and production datasets {universe} successfully saved")
    logger.info(f"Final data has {len(df_processed)} rows for {df_processed['ticker'].nunique()} tickers")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python -m src.data_pipelines.run_backfill <universe> <start_date> <end_date>")
        sys.exit(1)

    universe = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]

    run_backfill(universe, start_date, end_date)