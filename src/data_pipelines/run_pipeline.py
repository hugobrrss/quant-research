"""
Daily data pipeline orchestration script
Fetches, validates, processes, and stores market data. In particular, the pipeline flow is as follows:
    fetch → validate → if critical, stop and notify
                     → if ok/warning, proceed to processor → processor handles the warnings (drop duplicates, fill NaN, etc.)

                     after processing -> features building and saving of data
"""

import pandas as pd
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

    project_root = Path(__file__).parent.parent.parent  # this assumes src/data_pipelines
    raw_path = project_root / "data" / "raw" / name

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

        conn.disconnect()

    # Validate raw data
    from src.data_pipelines.validator import validate_pipeline_data
    validation_dict = validate_pipeline_data(data_raw, tickers, "daily", date_today.strftime("%Y-%m-%d"))

    if validation_dict['critical']:
        # instructions to stop the pipeline and obtain an alert
        logger.info(f"Critical issue(s) with the fetched data")
        for msg in validation_dict['critical']:
            print(msg)

        # here something to obtain email and SMS alerts
        return

    # Load current file and append the new data
    df = pd.read_parquet(raw_path)
    df = pd.concat([df, data_raw], ignore_index=False)
    df = df.sort_values(['ticker'])
    df = df.sort_index()

    # Save new raw_data
    df.to_parquet(raw_path)

    # Process data and save processed file
    from src.data_pipelines.processor import process_pipeline_data
    df_processed = process_pipeline_data(df)
    processed_path = project_root / "data" / "processed" / name
    df_processed.to_parquet(processed_path)

    # Build features and saved data containing the features
    import src.data_pipelines.features_equities as feat
    df_features = feat.add_momentum_features(df_processed)

    # bespoke momentum: standard momentum (last year except last month), and short-term reversal
    bespoke_momentum_params = [(21, 252), (1, 21)]
    for k, p in bespoke_momentum_params:
        df_features = feat.add_bespoke_momentum(df_features, k, p)

    df_features = feat.add_volume_features(df_features)
    df_features = feat.add_volatility_features(df_features)
    df_features = feat.add_mean_rev_features(df_features)

    features_path = project_root / "data" / "features" / name
    df_features.to_parquet(features_path)
