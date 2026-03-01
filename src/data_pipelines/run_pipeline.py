"""
Daily data pipeline orchestration script
Fetches, validates, processes, and stores market data. In particular, the pipeline flow is as follows:
    fetch → validate → if critical, stop and notify
                     → if ok/warning, proceed to processor → processor handles the warnings (drop duplicates, fill NaN, etc.)

                     after processing -> features building and saving of data
"""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

def run_equity_pipeline(tickers: list[str],
                        source: str,
                        name: str) -> None:
    """
    Run the full equity data pipeline.
    Steps are:
        1. fetch raw price data
        2. validate data
        3. process data
        4. build features
        5. save to disk

    Arguments:
        tickers: list of strings (tickers)
        source: 'yahoo' or 'IB' depending on which source is used for fetching data
        name: name of the file to which data is saved, should correspond to a universe of stocks, eg sp500
    """

    date_today = datetime.now()

    if source == 'yahoo':
        from src.data_pipelines.yahoo_fetcher import fetch_tickers
        data_raw = fetch_tickers(tickers, date_today.strftime("%Y-%m-%d"), date_today.strftime("%Y-%m-%d"))

    elif source == 'IB':
        from src.execution.ib_connection import IBConnection
        from src.data_pipelines.ib_fetcher import fetch_historical_data

        conn = IBConnection()
        conn.connect()
        contract_specs = []
        for ticker in tickers:
            ticker_dic = {'sec_type': 'STK', 'symbol': ticker}
            contract_specs.append(ticker_dic)

        data_raw = fetch_historical_data(conn, contract_specs, duration='1 D', bar_size='1 day')

    # Validate raw data
    from src.data_pipelines.validator import validate_pipeline_data
    validation_dict = validate_pipeline_data(data_raw,tickers,"daily",date_today.strftime("%Y-%m-%d"))



    # Process validated data


    """
    below up to the print line is not correct:
    I want to:
        1. find the latest version of the file
        2. append with today's data
        3. save it and update the file name so it reads as the latest date
    """
    project_root = Path(__file__).parent.parent.parent # this assumes src/data_pipelines
    raw_path = project_root / "data" / "raw" / (name + '_' + datetime.now().strftime("%Y%m%d"))

    data_raw.to_parquet(raw_path)
    print(f"Saved raw data to {raw_path}")



    # process

    # create features from processed data


