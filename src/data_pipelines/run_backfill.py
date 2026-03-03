""""
Backfill script: fetches an initial research dataset
This script is designed as a one-off task, to be run manually when starting a new universe or refreshing a research dataset
"""

import logging
import pandas as pd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from src.data_pipelines.validator import validate_pipeline_data
from src.data_pipelines.processor import process_pipeline_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_backfill(tickers: list[str],
                        source: str,
                        name: str,
                        start_date: str,
                        end_date: str) -> None:
    """
    Fetch a research dataset.
    Steps are:
        1. fetch raw price data
        2. validate data
        3. process data
        4. save to disk

    Arguments:
        tickers: list of strings (tickers) for the universe of instruments
        source: 'yahoo' or 'IB' depending on which source is used to fetch data
        name: name of the file to save
        start_date: start date for backfill, format YYYY-MM-DD
        end_date: end date for backfill, format YYYY-MM-DD

    Notes: IB's API has a max duration of one year for daily bars. For data longer than one year, request must be chunked in
           multiple calls of at most one year each.
    """

    project_root = Path(__file__).parent.parent.parent  # this assumes src/data_pipelines
    data_research = project_root / "data" / "features" / f"{name}_research.parquet"
    data_production = project_root / "data" / "features" / f"{name}_production.parquet"

    if source == 'yahoo':
        from src.data_pipelines.yahoo_fetcher import fetch_tickers
        data_raw = fetch_tickers(tickers, start_date, end_date)

    elif source == 'IB':
        from src.execution.ib_connection import IBConnection
        from src.data_pipelines.ib_fetcher import fetch_historical_data

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
                chunk_start = max(s, chunk_end - relativedelta(years=1))
                duration = (chunk_end - chunk_start).days
                chunk = fetch_historical_data(conn, contract_specs, end_date=chunk_end, duration=f'{duration} D')
                all_data.append(chunk)
                chunk_end = chunk_start

            data_raw = pd.concat(all_data, ignore_index=False)
            data_raw = data_raw.drop_duplicates()
            # The procedure fills "backwards", so sort on tickers and index (date)
            data_raw = data_raw.sort_values(['ticker'])
            data_raw = data_raw.sort_index()
        finally:
            conn.disconnect()

    else:
        raise ValueError(f"Unknown source: {source}")

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

    logger.info(f"Research dataset {name} successfully saved")
    logger.info(f"Final data has {len(df_processed)} rows for {df_processed['ticker'].nunique()} tickers")