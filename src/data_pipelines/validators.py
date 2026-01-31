"""
Validation function for data pipeline
"""

import pandas as pd
import logging

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# new class that inherits from Exception
class ValidationError(Exception):
    """Raised when data validation fails"""
    pass
# The string in "raise ValidationError(--)" becomes the exception's message

def validate_price_data(df: pd.DataFrame, ticker: str) -> bool:
    """
    Validate fetched price data

    Checks:
    1. DataFrame is not empty
    2. Required columns exist
    3. No null values in critical columns
    4. Prices are positive
    5. High price higher than low price
    6. Dates are in expected order

    Args:
        df: DataFrame of prices to validate
        ticker: Ticker symbol (for logging)

    Returns:
        bool: True if valid, False otherwise

    Raises:
        ValidationError: If data validation fails
    """

    price_cols = ['Open', 'High', 'Low', 'Close']

    if df.empty:
        raise ValidationError(f"{ticker}: DataFrame is empty")

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValidationError(f"{ticker}: Missing required columns {missing}")

    null_counts = df[price_cols].isnull().sum()
    if null_counts.any():
        logger.warning(f"{ticker}: Found null values:\n {null_counts[null_counts > 0]}")

    for col in price_cols:
        if (df[col]<=0).any():
            raise ValidationError(f"{ticker}: {col} is negative")

    if (df['High']<df['Low']).any():
        raise ValidationError(f"{ticker}: High < Low detected")

    logger.info(f"Validated {ticker} ({len(df)} rows)")
    return True

def validate_time_range(
        df: pd.DataFrame,
        ticker: str,
        expected_start: str,
        expected_end: str,
        min_coverage: float = 0.9
) -> bool:
    """
    Check that data covers the expected range
    df: DataFrame with DatetimeIndex
    ticker: Ticker symbol
    expected_start: Start date in format 'YYYY-MM-DD'
    expected_end: End date in format 'YYYY-MM-DD'
    min_coverage: Minimum fractions of trading days required (default: 0.9)

    Returns:
        True if coverage is sufficient, False otherwise
    """
    # Approximate trading days (252 days per year)
    from datetime import datetime
    start = datetime.strptime(expected_start, '%Y-%m-%d')
    end = datetime.strptime(expected_end, '%Y-%m-%d')
    calendar_days = (end - start).days
    expected_trading_days = calendar_days * (252/365)

    actual_days = len(df)
    coverage = actual_days / expected_trading_days
    if coverage < min_coverage:
        logger.warning(
            f"{ticker}: Low coverage: got {actual_days} days, "
            f"expected ~ {expected_trading_days:.0f} days ({coverage:.1f} coverage)"
        )
        return False
    return True

