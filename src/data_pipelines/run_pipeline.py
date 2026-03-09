"""
Daily data pipeline orchestration script
Fetches, validates, processes, and stores market data. In particular, the pipeline flow is as follows:
    fetch → validate → if critical, stop and notify
                     → if ok/warning, proceed to processor → processor handles the warnings (drop duplicates, fill NaN, etc.)

                     after processing -> features building and saving of data
"""
import yaml
import pandas as pd
import logging
from datetime import datetime, date
from pathlib import Path
from src.data_pipelines.validator import validate_pipeline_data
from src.data_pipelines.processor import process_pipeline_data
import src.data_pipelines.features_equities as feat
from src.execution.ib_connection import IBConnection
from src.data_pipelines.ib_fetcher import fetch_historical_data

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_equity_pipeline(universe: str) -> None:
    """
    Run the full equity data pipeline.
    Steps are:
        1. fetch raw price data
        2. validate data
        3. process data
        4. build features
        5. save to disk

    Arguments:
        universe: name of the equity universe in the config YAML file
    """

    # Obtain the list of tickers from the config file
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config" / "equity_universes.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if universe not in config:
        raise ValueError(f"Universe '{universe}' not found in {config_path}")

    tickers = config[universe]

    date_today = datetime.now().date()

    project_root = Path(__file__).parent.parent.parent  # this assumes src/data_pipelines
    data_path = project_root / "data" / "production" / f"{universe}_production.parquet"

    # Load current data
    if not data_path.exists():
        raise FileNotFoundError(f"No existing dataset found at {data_path}. Run backfill first.")

    df_existing = pd.read_parquet(data_path)

    # Keep only base columns (for proper concatenation later)
    base_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'ticker']
    df_existing = df_existing[base_columns]

    # Check that today's date is not already in data to prevent the pipeline running twice on a single day
    if pd.Timestamp(date_today) in df_existing.index.normalize():
        logger.info(f"Today's ({date_today}) data already fetched")
        return

    conn = IBConnection()
    try:
        conn.connect()
        contract_specs = []
        for ticker in tickers:
            ticker_dic = {'sec_type': 'STK', 'symbol': ticker}
            contract_specs.append(ticker_dic)

        last_date = df_existing.index.max().date()
        days_gap = (date_today - last_date).days
        data_raw = fetch_historical_data(conn, contract_specs, duration=f'{days_gap} D', bar_size='1 day')

        # trim the data: IB fetcher likely works on trading and not calendar days, so duration will overshoot
        last_date = df_existing.index.max()
        data_raw = data_raw[data_raw.index > last_date]

    finally:
        conn.disconnect()


    # Validate raw data
    validation_dict = validate_pipeline_data(data_raw, tickers, "daily", date_today.strftime("%Y-%m-%d"))

    if validation_dict['critical']:
        # instructions to stop the pipeline and obtain an alert
        logger.critical(f"Critical issue(s) with the fetched data")
        for msg in validation_dict['critical']:
            print(msg)

        # here something to obtain email and SMS alerts
        return

    # Append the new data
    df_new = pd.concat([df_existing, data_raw], ignore_index=False)
    df_new = df_new.sort_index()
    df_new = df_new.sort_values(['ticker'])

    # Trim the data to keep only one year of data in the production dataset
    cutoff = pd.Timestamp(date_today) - pd.DateOffset(years=2)
    df_new = df_new[df_new.index >= cutoff]

    # Process data
    df_processed = process_pipeline_data(df_new)

    # Build features and save the updated data
    df_features = feat.add_momentum_features(df_processed)

    # bespoke momentum: standard momentum (last year except last month), and short-term reversal
    bespoke_momentum_params = [(21, 252), (1, 21)]
    for k, p in bespoke_momentum_params:
        df_features = feat.add_bespoke_momentum(df_features, k, p)

    df_features = feat.add_volume_features(df_features)
    df_features = feat.add_volatility_features(df_features)
    df_features = feat.add_mean_rev_features(df_features)

    # Save new file
    df_features.to_parquet(data_path)

    logger.info(f"Today's {date_today} pipeline successfully ran")
    logger.info(f"Final data has {len(df_features)} rows for {df_features['ticker'].nunique()} tickers")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python -m src.data_pipelines.run_pipeline <universe> <start_date> <end_date>")
        sys.exit(1)

    universe = sys.argv[1]

    run_equity_pipeline(universe)