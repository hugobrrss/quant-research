"""
Fetches daily price data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

def fetch_tickers(
        tickers: list[str],
        start_date: str,
        end_date: str
) -> pd.DataFrame:
    """
    Fetch daily OHLCV for a list of ticker(s).

    Arguments:
        tickers: list of tickers (eg ['AAPL'], or ['MSFT', 'AMZN', 'GOOG'] for multiple tickers)
        start_date: start date in 'YYYY-MM-DD' format
        end_date: end date in 'YYYY-MM-DD' format

    Returns:
        DataFrame with columns: Open, High, Low, Close, Adj Close, Volume, Ticker
    """
    all_df = []
    for ticker in tickers:
        df = yf.download(ticker, start_date, end_date)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if not df.empty:
            df['ticker'] = ticker
            all_df.append(df)

        time.sleep(0.5)

    if all_df:
        combined_df = pd.concat(all_df)
        logger.info(f"Fetched data for {len(tickers)} tickers ({len(combined_df)} rows")
        return combined_df
    else:
        return pd.DataFrame()


def save_raw_data(df: pd.DataFrame, filename: str) -> Path:
    """
    Save raw data to data/raw as Parquet
    Parquet is better than csv:
    - preservers data types
    - much smaller file size
    - faster to read/write
    """
    project_root = Path(__file__).parent.parent.parent # this assumes src/data_pipelines
    raw_path = project_root / "data" / "raw" / filename

    df.to_parquet(raw_path)
    print(f"Saved raw data to {raw_path}")
    return raw_path

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "AMZN", "GOOG"]
    start_date = '2020-01-01'
    end_date = '2020-12-31'

    df = fetch_tickers(tickers, start_date, end_date)
    print(f"\nFetched {len(df)} rows")
    print(df.head())

    # save
    timestamp = datetime.now().strftime("%Y%m%d")
    save_raw_data(df,f"prices_{timestamp}.parquet")