"""
Daily data pipeline orchestration script
Fetches, validates, processes, and stores market data
"""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

def run_equity_pipeline(tickers: str | list[str],
                        source: str,
                        start_date: str = None,
                        end_date: str = None):
    """
    Run the full equity data pipeline.
    Steps are:
        1. fetch raw price data
        2. validate data
        3. process data
        4. build features
        5. store to disk

    Arguments:
        tickers: string of list of strings (tickers)
        start_date/end_date: date in string format (YYYY-MM-DD)
        source: 'yahoo' or 'IB' depending on which source is used for fetching data
    """
    project_root = Path(__file__).parent.parent.parent

    if source == 'yahoo':
        from src.data_pipelines.yahoo_fetcher import *


    elif source == 'IB':
