"""
This is a pipeline-level validation script
It checks:
    1. ticker coverage
    2. all NaN values
    3. price is positive
    4. outlier values
"""

import pandas as pd
import numpy as np
from collections import Counter
import logging

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_pipeline_data(df: pd.DataFrame,
                           expected_tickers: list[str],
                           mode: str,
                           expected_date: str=None) -> dict:
    """
    Add description
    Arguments:
        df: raw DataFrame of prices
        expected_tickers: list of tickers expected to be fetched
        expected_date: expected date of data to be fetched, format 'YYYY-MM-DD'
        mode: 'backfill' or 'daily': the set of checks for backfill vs. daily changes
    Returns:
        a summary dictionary with severity levels 'Ok', 'warning' or 'critical', and the associated warnings as values
        the dictionary is then used for data processing:
            Ok: everything is fine, proceed
            Warning: something is off but tolerable and/or the processor can fix it (duplicates, some NaN values, a few missing tickers)
            Critical: the data is fundamentally broken and not even worth processing
    """
    validator_summary = {"ok": [], "warning": [], "critical": []}

    threshold_ok = 0.95
    threshold_warning = 0.8


    if mode == 'backfill':

        logger.info("Starting backfill validation")

        # empty data
        if df.empty:
            msg = f"Didn't fetch anything, data is empty"
            logger.critical(msg)
            validator_summary["critical"].append(msg)
            return validator_summary

        # ticker coverage (should handle delisting and index rebalancing in some way)
        actual_tickers = df['ticker'].unique().tolist()
        missing = set(expected_tickers) - set(actual_tickers)
        if missing:
            msg = f"Found {len(missing)} tickers with missing tickers: {missing}"
            logger.warning(msg)
            validator_summary["warning"].append(msg)

        # negative prices
        price_columns = ['Open','High','Low','Close']
        for col in price_columns:
            msg = f"Found negative price(s) in {col}"
            if (df[col]<0).any():
                logger.warning(msg)
                validator_summary["warning"].append(msg)

        # all NaN tickers, for all dates
        tickers_allnan = df[price_columns].isna().groupby(df['ticker']).all().all(axis=1)
        tickers_allnan = tickers_allnan[tickers_allnan].index.tolist()
        if tickers_allnan:
            msg = f"Found {len(tickers_allnan)} tickers with all NaN values for all dates: {tickers_allnan}"
            logger.warning(msg)
            validator_summary["warning"].append(msg)

        # tickers that have less than half the rows of the median tickers
        obs_per_ticker = df.groupby('ticker')['Close'].count()
        median_obs = obs_per_ticker.median()
        low_coverage_tickers = obs_per_ticker[obs_per_ticker < 0.5 * median_obs].index.tolist()
        if low_coverage_tickers:
            msg = f"Fetched tickers with coverage below 50% of the median ticker coverage: {low_coverage_tickers}"
            logger.warning(msg)
            validator_summary["warning"].append(msg)

        # duplicates
        if df.duplicated().any():
            duplicated_tickers = df.loc[df.duplicated(),'ticker'].tolist()
            msg = f"Found duplicate observations for the following tickers: {duplicated_tickers}"
            logger.warning(msg)
            validator_summary["warning"].append(msg)

        # outliers
        df['daily_return'] = df.groupby('ticker')['Close'].pct_change()
        threshold_outlier = 0.5
        outlier_tickers = df.loc[np.abs(df['daily_return']) > threshold_outlier,'ticker'].tolist()
        if outlier_tickers:
            msg = f"Found outlier(s) in tickers: {outlier_tickers}"
            logger.warning(msg)
            validator_summary["warning"].append(msg)

        logger.info(f"Validation completed: {len(validator_summary['warning'])} warnings, {len(validator_summary['critical'])} critical")
        return validator_summary

    elif mode == 'daily':

        logger.info("Starting daily validation")

        # empty data
        if df.empty:
            msg = f"Didn't fetch anything, data is empty"
            logger.critical(msg)
            validator_summary["critical"].append(msg)
            return validator_summary

        # missing tickers
        n_expected_tickers = len(expected_tickers)
        n_actual_tickers = df['ticker'].nunique()
        msg = f"{(n_actual_tickers / n_expected_tickers)*100}% of expected tickers were fetched"
        if n_actual_tickers > n_expected_tickers * threshold_ok:
            validator_summary["ok"].append(msg)
        elif n_expected_tickers * threshold_ok > n_actual_tickers > n_expected_tickers * threshold_warning:
            logger.warning(msg)
            validator_summary["warning"].append(msg)
        else:
            logger.critical(msg)
            validator_summary["critical"].append(msg)

        # negative prices
        price_columns = ['Open','High','Low','Close']
        for col in price_columns:
            msg = f"Found negative price(s) in {col}"
            if (df[col]<0).any():
                logger.warning(msg)
                validator_summary["warning"].append(msg)

        # all prices NaN for a ticker
        tickers_allnan = df.loc[df[price_columns].isna().all(axis=1),'ticker'].tolist()
        if tickers_allnan:
            msg = f"Found {len(tickers_allnan)} tickers with all NaN values: {tickers_allnan}"
            logger.warning(msg)
            validator_summary["warning"].append(msg)

        # duplicate rows
        if df.duplicated().any():
            duplicated_tickers = df.loc[df.duplicated(),'ticker'].tolist()
            msg = f"Found duplicate tickers: {duplicated_tickers}"
            logger.warning(msg)
            validator_summary["warning"].append(msg)

        # check that date is as expected
        if expected_date is not None:
            dates_in_data = df.index.normalize().unique()
            expected = pd.Timestamp(expected_date)

            if len(dates_in_data) == 1:
                if dates_in_data[0] != expected:
                    msg = f"Expected date: {expected_date}, but got {dates_in_data[0]}"
                    logger.critical(msg)
                    validator_summary["critical"].append(msg)
            else:
                msg = f"Expected 1 date, found {len(dates_in_data)}: {dates_in_data.tolist()}"
                logger.critical(msg)
                validator_summary["critical"].append(msg)

        logger.info(f"Validation completed: {len(validator_summary['warning'])} warnings, {len(validator_summary['critical'])} critical")
        return validator_summary


    return validator_summary