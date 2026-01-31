"""
Cleans and processes validated price data
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def fill_missing_prices(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Fill missing price values

    Args:
        df: DataFrame of prices
        method: 'ffill' for forward fill, or 'drop'

    Returns:
        DataFrame with missing values handled
    """

    price_cols = ['High', 'Low', 'Open', 'Close']
    missing_before = df[price_cols].isna().sum().sum()

    if method == 'ffill':
        df[price_cols] = df[price_cols].ffill()

    elif method == 'drop':
        df = df.dropna(subset=price_cols)

    missing_after = df[price_cols].isna().sum().sum()

    if missing_before > 0:
        logger.info(f"Filled {missing_before-missing_after} missing prices")

    return df

def detect_outliers(df: pd.DataFrame, column: str = "Close", threshold: float = 0.5) -> pd.Series:
    """
    Detect outliers based on daily returns
    A return above the threshold (50% by default) is suspicious
    Likely a data error or stock split not adjusted

    Args:
        df: DataFrame of prices
        column: column name to look for
        threshold: threshold for outliers

    Returns:
        Boolean series with outliers detected
        """

    returns = df[column].pct_change().abs()
    outliers = returns > threshold

    if outliers.any():
        outliers_dates = df.index[outliers].tolist()
        logger.info(f"Outliers detected on {outliers_dates}")

    return outliers

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
        - daily_return: a column with daily returns in the DataFrame
        - log_return: a column with log returns in the DataFrame
    """

    df = df.copy()
    df['daily_return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    return df

def process_ticker_data(
        df: pd.DataFrame,
        ticker: str,
        fill_method: str = 'ffill',
        check_outliers: bool = True
) -> pd.DataFrame:
    """
    Full processing timeline for a single ticker
    Returns:
        Processed DataFrame with added returns
    """
    df = df.copy()
    df = fill_missing_prices(df, method=fill_method)
    if check_outliers:
        df['is_outlier'] = detect_outliers(df)
    df = compute_returns(df)

    logger.info(f"{ticker} processing complete {len(df)} rows")
    return df

def save_processed_data(df: pd.DataFrame, filename: str) -> Path:
    """"Save processed data to data/processed """
    project_root = Path(__file__).parent.parent.parent
    processed_path = project_root / "data" / "processed" / filename
    df.to_parquet(processed_path)
    logger.info(f"Saved processed data to {processed_path}")
    return processed_path
