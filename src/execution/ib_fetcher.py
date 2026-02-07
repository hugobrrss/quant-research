"""
Fetches historical data from Interactive Brokers.
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Optional

from src.execution.ib_connection import *

logger = logging.getLogger(__name__)


def fetch_historical_data(
        symbol: str,
        duration: str = '1 Y',
        bar_size: str = '1 day',
        what_to_show: str = 'TRADES',
        use_rth: bool = True,
        instrument_type: str = 'stock',
        exchange: str = 'SMART',
        currency: str = 'USD',
        expiry: str = None,
        port: int = 7497
) -> pd.DataFrame:
    """
    Fetch historical data from Interactive Brokers.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL', 'EURUSD', 'ES')
        duration: How far back ('1 D', '1 W', '1 M', '1 Y')
        bar_size: Bar size ('1 min', '5 mins', '1 hour', '1 day')
        what_to_show: Data type ('TRADES', 'MIDPOINT', 'BID', 'ASK')
        use_rth: Regular trading hours only
        instrument_type: 'stock', 'forex', or 'future'
        exchange: Exchange (for stocks, 'SMART' = IB smart routing)
        currency: Currency of instrument
        expiry: Expiry for futures (e.g., '202403')
        port: TWS port (7497=paper, 7496=live)

    Returns:
        DataFrame with OHLCV data
    """
    conn = IBConnection(port=port)

    try:
        conn.connect()

        # Create contract based on instrument type
        if instrument_type == 'stock':
            contract = get_stock_contract(symbol, exchange, currency)
        elif instrument_type == 'forex':
            contract = get_forex_contract(symbol)
        elif instrument_type == 'future':
            if expiry is None:
                raise ValueError("Expiry required for futures")
            contract = get_future_contract(symbol, exchange, expiry)
        else:
            raise ValueError(f"Unknown instrument type: {instrument_type}")

        # Qualify contract
        conn.ib.qualifyContracts(contract)

        # Fetch historical data
        bars = conn.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth
        )

        if not bars:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([{
            'date': bar.date,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume
        } for bar in bars])

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df['ticker'] = symbol

        logger.info(f"Fetched {len(df)} bars for {symbol}")
        return df

    finally:
        conn.disconnect()


def fetch_multiple_symbols(
        symbols: list[str],
        duration: str = '1 Y',
        bar_size: str = '1 day',
        instrument_type: str = 'stock',
        port: int = 7497
) -> pd.DataFrame:
    """
    Fetch historical data for multiple symbols.

    Args:
        symbols: List of ticker symbols
        duration: How far back
        bar_size: Bar size
        instrument_type: Type of instrument
        port: TWS port

    Returns:
        Combined DataFrame with all symbols
    """
    all_data = []

    conn = IBConnection(port=port)
    conn.connect()

    try:
        for symbol in symbols:
            logger.info(f"Fetching {symbol}...")

            if instrument_type == 'stock':
                contract = get_stock_contract(symbol)
            elif instrument_type == 'forex':
                contract = get_forex_contract(symbol)
            else:
                raise ValueError(f"Use fetch_historical_data for futures")

            conn.ib.qualifyContracts(contract)

            bars = conn.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True
            )

            if bars:
                df = pd.DataFrame([{
                    'date': bar.date,
                    'Open': bar.open,
                    'High': bar.high,
                    'Low': bar.low,
                    'Close': bar.close,
                    'Volume': bar.volume
                } for bar in bars])

                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df['ticker'] = symbol
                all_data.append(df)

                logger.info(f"  {symbol}: {len(df)} bars")
            else:
                logger.warning(f"  {symbol}: No data")

            # Small delay to avoid rate limits
            import time
            time.sleep(0.5)

    finally:
        conn.disconnect()

    if all_data:
        return pd.concat(all_data)
    return pd.DataFrame()