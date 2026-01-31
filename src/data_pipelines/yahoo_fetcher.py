"""
Fetches daily price data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

def fetch_single_ticker(
        ticker: str,
        start_date: str,
        end_date: str
) -> pd.DataFrame:
    """
    Fetch daily OHLCV for a single ticker.

    Arguments:
        ticker: stock ticker (eg 'AAPL')
        start_date: start date in 'YYYY-MM-DD' format
        end_date: end date in 'YYYY-MM-DD' format

    Returns:
        DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
    """
    data = yf.download(ticker, start_date, end_date, progress=False)
    # Flatten MultiIndex column if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data['ticker'] = ticker # add ticker to the DataFrame to anticipate multiple tickers fetching
    return data

def fetch_multiple_tickers(
        tickers: list[str],
        start_date: str,
        end_date: str
) -> pd.DataFrame:
    """
    Same but for multiple tickers, stacked into a single DataFrame
    Arguments:
        tickers: list of tickers
    Rest is unchanged
    """
    all_data = []
    for ticker in tickers:
        df = fetch_single_ticker(ticker, start_date, end_date)
        all_data.append(df)

    combined = pd.concat(all_data)
    return combined

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

    df = fetch_multiple_tickers(tickers, start_date, end_date)
    print(f"\nFetched {len(df)} rows")
    print(df.head())

    # save
    timestamp = datetime.now().strftime("%Y%m%d")
    save_raw_data(df,f"prices_{timestamp}.parquet")