""" Fetch data from FRED """
import pandas as pd
import time
import logging
from fredapi import Fred
from dotenv import load_dotenv
import os
import yaml
from pathlib import Path

from networkx.classes import is_empty

logger = logging.getLogger(__name__)

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

def fetch_fred_series(series_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Args:
        - series_id: FRED series id
        - start_date: format 'YYYY-MM-DD'
        - end_date: format 'YYYY-MM-DD'
    Returns:
        - a DataFrame with columns date, value, series_id
          the index is NOT the date on purpose, to anticipate multiple series fetching with different dates and frequency
    """
    try:
        raw = fred.get_series(series_id=series_id, observation_start=start_date, observation_end=end_date)
        df = raw.reset_index()
        df.columns = ['date', 'value']
        df['series_id'] = series_id

        logger.info(f"Fetched {len(df)} observation for FRED series {series_id}")
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch FRED series {series_id}: {e}")
        return pd.DataFrame()

def fetch_multiple_fred_series(series_ids: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    all_df = []
    for series_id in series_ids:

        df = fetch_fred_series(series_id, start_date, end_date)
        if not df.empty:
            all_df.append(df)

        # small delay to avoid rate limits
        time.sleep(0.5)

    if all_df:
        return pd.concat(all_df, ignore_index=True)
    else:
        return pd.DataFrame()


def fetch_macro_universe(group_names: str | list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    This function fetches all the variables from a macro universe, defined in a YAML file.
    Args:
        - group_name: name of the macro universe, must be in the file /config/macro_universe.yaml
        - start_date: format 'YYYY-MM-DD'
        - end_date: format 'YYYY-MM-DD'
    """
    project_root = Path(__file__).parent.parent.parent # this assumes src/data_pipelines
    raw_path = project_root / "config" / "macro_universes.yaml"
    with open(raw_path, "r") as f:
        config = yaml.safe_load(f)

    if isinstance(group_names, str):
        series_ids = list(config[group_names].keys())
    else:
        series_ids = []
        for group_name in group_names:
            series_id_group = list(config[group_name].keys())
            series_ids.extend(series_id_group)

    return fetch_multiple_fred_series(series_ids, start_date, end_date)
