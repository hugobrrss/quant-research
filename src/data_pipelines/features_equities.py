"""
Features engineering for equity momentum strategy
The data read by the script has been validated and processed
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum features (past returns over various horizons)
    Features added:
        - mom_5d: 5-day return
        - mom_21d: 21-day (1 month) return
        - mom_63d: 63-day (1 quarter) return
        - mom_252d: 252-day (1 year) return
    """
    df = df.copy()
    windows = {
        'mom_5d': 5,
        'mom_21d': 21,
        'mom_63d': 63,
        'mom_252d': 252
    }
    for name, window in windows.items():
        df[name] = df.groupby('ticker')['Close'].pct_change(window)

    logger.info(f"Added momentum features {list(windows.keys())}")
    return df

def add_bespoke_momentum(df: pd.DataFrame, k: int, p: int) -> pd.DataFrame:
    """
    This adds "bespoke" momentum measure based on two parameters k and p, with k<p
    In particular:
        mom_k_p = (P_{t-p} - P_{t-k}) / P_{t-k}
    """
    df = df.copy()
    if k >= p:
        logger.warning(f"Start date must be before end date")
    else:
        df['mom_'+str(k)+'_'+str(p)] = ((df.groupby('ticker')['Close'].shift(k) - df.groupby('ticker')['Close'].shift(p))
                                        / df.groupby('ticker')['Close'].shift(p))

    logger.info(f"Added bespoke momentum features {'mom_'+str(k)+'_'+str(p)}")
    return df

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility features.

    Features added:
        - vol_21d: 21-day rolling volatility (annualized)
        - vol_63d: 63-day rolling volatility (annualized)
        - range_hl: High-Low range as % of close
    """
    df = df.copy()
    df['vol_21d'] = df.groupby('ticker')['return'].transform(lambda x: x.rolling(21).std() * np.sqrt(252))
    df['vol_63d'] = df.groupby('ticker')['return'].transform(lambda x: x.rolling(63).std() * np.sqrt(252))
    df['range_hl'] = (df['High'] - df['Low']) / df['Close']

    logger.info(f"Added volatility features: vol_21d, vol_63d, range_hl")
    return df

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based features.

    Features added:
        - volume_ma_21d: 21-day average volume
        - volume_ratio: Today's volume vs 21-day average
    """
    df = df.copy()
    df['volume_ma_21d'] = df.groupby('ticker')['Volume'].transform(lambda x: x.rolling(21).mean())
    df['volume_ratio'] = df['Volume'] / df['volume_ma_21d']

    logger.info(f"Added volume-based features: volume_ma_21d, volume_ratio")
    return df

def add_mean_rev_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mean reversion features.

    Features added:
        - ma_21d: 21-day moving average of close
        - ma_50d: 50-day moving average of close
        - dist_from_ma_21d: % distance from 21-day MA
        - dist_from_ma_50d: % distance from 50-day MA
    """
    df = df.copy()
    df['ma_21d'] = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(21).mean())
    df['ma_50d'] = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(50).mean())
    df['dist_from_ma_21d'] = (df.groupby('ticker')['Close'] - df.groupby('ticker')['ma_21d']) / df.groupby('ticker')['ma_21d']
    df['dist_from_ma_50d'] = (df.groupby('ticker')['Close'] - df.groupby('ticker')['ma_50d']) / df.groupby('ticker')['ma_50d']

    logger.info(f"Added mean-reversion features: ma_21d, ma_50d, dist_from_ma_21d, dist_from_ma_50d")
    return df



