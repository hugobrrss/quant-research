"""
Cleans and processes validated price data
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def process_pipeline_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes previously validated price data
    The processor always reads the full dataframe (either appended with last day's data or entirely fetched)
    Steps:
        1. remove duplicates
        2. fill missing values
        3. drop all NaN tickers
        5. compute daily returns
        4. check and flag outliers (based on daily returns)
    """

    df = df.copy()
    df = df.sort_values('ticker').sort_index()

    outlier_thresh = 0.5

    # remove duplicates
    n_beforedrop = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped {n_beforedrop-len(df)} rows of duplicate observations")

    # fill missing values (forward fill) - some NaN could remain if they are at the start of a ticker's history
    price_cols = ['High', 'Low', 'Open', 'Close']
    nan_before = df[price_cols].isna().sum().sum()
    df[price_cols] = df.groupby('ticker')[price_cols].transform('ffill')
    nan_after = df[price_cols].isna().sum().sum()
    logger.info(f"Filled {nan_before - nan_after} missing prices ({nan_after} remaining)")

    # drop all NaN tickers
    tickers_allnan = df.groupby('ticker')[price_cols].apply(lambda x: x.isna().all().all())
    tickers_to_drop = tickers_allnan[tickers_allnan].index
    df = df[~df['ticker'].isin(tickers_to_drop)]
    logger.info(f"Dropped {len(tickers_to_drop)} tickers (all NaN prices)")

    # compute daily returns
    df['return'] = df.groupby('ticker')['Close'].pct_change()

    # check and flag outliers
    df['outliers'] = np.abs(df['return']) > outlier_thresh
    logger.info(f"Found {df['outliers'].sum()} potential outliers (shown in column 'outliers')")

    return df
