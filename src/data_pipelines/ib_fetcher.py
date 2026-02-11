"""
Fetches historical data from Interactive Brokers.
"""

import pandas as pd
import logging
import time
from ib_insync import Contract
from src.execution.ib_connection import IBConnection

logger = logging.getLogger(__name__)

def create_contract(
        sec_type: str,
        symbol: str,
        exchange: str = 'SMART',
        currency: str = 'USD',
        expiry: str = None,
        strike: float = None,
        right: str = None,
        secIdType: str = None,
        secId: str = None) -> Contract:
    """
    Factory function to create IB contract objects.
    """
    contract = Contract()
    contract.secType = sec_type
    contract.symbol = symbol
    contract.exchange = exchange
    contract.currency = currency

    if expiry is not None:
        contract.lastTradeDateOrContractMonth = expiry

    if strike is not None:
        contract.strike = strike

    if right is not None:
        contract.right = right

    if secIdType is not None and secId is not None:
        contract.secIdType = secIdType
        contract.secId = secId

    return contract


def fetch_historical_data(
        conn: IBConnection,
        contract_specs: list[dict],
        duration: str = '1 Y',
        bar_size: str = '1 day',
        what_to_show: str = 'TRADES',
        use_rth: bool = True
) -> pd.DataFrame:
    """
    Fetch historical data from Interactive Brokers.

    Args:
        conn: IBConnection object
        contract_specs: a list of contract spec dictionaries containing the arguments of create_contract()
            !! the key of the dictionaries must correspond to the arguments of create_contract() !!
        duration: How far back ('1 D', '1 W', '1 M', '1 Y')
        bar_size: Bar size ('1 min', '5 mins', '1 hour', '1 day')
        what_to_show: Data type ('TRADES', 'MIDPOINT', 'BID', 'ASK', ‘BID_ASK’, ‘ADJUSTED_LAST’, ‘HISTORICAL_VOLATILITY’, ‘OPTION_IMPLIED_VOLATILITY’)
        use_rth: Regular trading hours only

    Returns:
        DataFrame with OHLCV data
    """

    all_data = []

    for contract_spec in contract_specs:
        contract = create_contract(**contract_spec)

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
            logger.warning(f"No data returned for {contract_spec['sec_type']} contract {contract_spec['symbol']}")
            continue

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
        if contract_spec['sec_type'] == 'BOND':
            df['ticker'] = contract_spec['sec_type'] + '_' + contract_spec['secId']
        elif contract_spec['sec_type'] == 'CASH':
            df['ticker'] = contract_spec['symbol'] + contract_spec['currency']
        elif contract_spec['sec_type'] == 'OPT':
            df['ticker'] = contract.conId
        else:
            df['ticker'] = contract_spec['symbol']

        all_data.append(df)

        time.sleep(0.5)

    if all_data:
        return pd.concat(all_data)
    return pd.DataFrame()
