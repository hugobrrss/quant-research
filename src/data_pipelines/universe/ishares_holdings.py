"""
Download, parse, filter and validate the iShares IWB (Russell 1000) holdings file.
This is to obtain, monitor and maintain the universe of tickers for the live trading strategy
"""
import io
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.paths import UNIVERSE_RAW_DIR

logger = logging.getLogger(__name__)

# --- Source ---
IWB_HOLDINGS_URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/"
    "get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares"
    "&locale=en_US&portfolioId=239707&userType=individual&component=holdings"
)
REQUEST_TIMEOUT = 30

# --- File format ---
AS_OF_LABEL = "Fund Holdings as of"
HEADER_RE = re.compile(r"^Ticker,", re.MULTILINE)
BLANK_LINE_RE = re.compile(r"\r?\n[ \t]*\r?\n")

# --- Filters ---
TICKER_PATTERN = r"[A-Z]+( [A-Z])?"          # common share, optional class letter (e.g. BRK B)
LISTED_EXCHANGES = {"NYSE", "NASDAQ", "Cboe BZX", "Nyse Mkt Llc", "TXE"}
COLUMNS = {"Ticker": "ticker", "Name": "name", "Sector": "sector",
           "Market Value": "market_value", "Price": "price",
           "Exchange": "exchange", "Location": "location"}

# --- Validation ---
MIN_ROWS, MAX_ROWS = 990, 1040
MAX_AGE_DAYS = 7
REQUIRED_COLUMNS = ["ticker", "name", "sector", "price", "market_value"]
GICS_SECTORS = {
    "Communication", "Consumer Discretionary", "Consumer Staples", "Energy",
    "Financials", "Health Care", "Industrials", "Information Technology",
    "Materials", "Real Estate", "Utilities",
}

# --- I/O: the only functions touching network or disk ---

def download_holdings(url: str = IWB_HOLDINGS_URL) -> bytes:
    """Fetch the holdings file. Raises on HTTP errors and timeouts."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content

def save_raw(content: bytes, raw_dir: Path = UNIVERSE_RAW_DIR) -> Path:
    """Write the raw download to raw_dir, named by UTC download time. Return the path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_path = raw_dir / f"IWB_holdings_{current_time}.csv"
    file_path.write_bytes(content)
    return file_path

# --- Pure logic: text / DataFrames in, data out ---

def parse_as_of_date(text: str) -> date:
    """
    Returns the holdings 'as of' date from the text metadata block.
    Raises ValueError if the label is missing or the date can't be parsed.
    """
    for line in text.splitlines():
        if line.startswith(AS_OF_LABEL):
            raw = line.split(",", maxsplit=1)[1].strip().strip('"')
            return datetime.strptime(raw, "%b %d, %Y").date()
    raise ValueError(f"'{AS_OF_LABEL}' not found in text metadata")

def parse_holdings_table(text: str) -> pd.DataFrame:
    """Locate the table (header line to first blank line) and parse it as CSV."""
    header_match = HEADER_RE.search(text)
    if header_match is None:
        raise ValueError("holdings header row ('Ticker,...') not found")
    parse_start = header_match.start()

    end_match = BLANK_LINE_RE.search(text, parse_start)  # search from the header onwards
    parse_end = end_match.start() if end_match else len(text)

    table_text = text[parse_start:parse_end]
    return pd.read_csv(io.StringIO(table_text), thousands=",", na_values="-")

def filter_equities(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the raw parsed datea: only equities and tickers with valid pattern, traded on known stock exchanges
    """
    equities = raw[raw["Asset Class"] == "Equity"]
    valid_ticker = equities["Ticker"].str.fullmatch(TICKER_PATTERN)
    listed = equities["Exchange"].isin(LISTED_EXCHANGES)
    keep = valid_ticker & listed
    dropped = equities.loc[~keep, ["Ticker", "Name", "Exchange"]]
    logger.info("Dropped %s equity rows:\n%s", len(dropped), dropped.to_string(index=False))
    return equities.loc[keep, list(COLUMNS)].rename(columns=COLUMNS)

def validate_holdings(holdings: pd.DataFrame, as_of: date, today: date) -> None:
    """
    Check the cleaned IWB holdings are plausible before they become a snapshot.
    Raises ValueError listing every failed check.
    """
    failures = []

    n = len(holdings)
    if not MIN_ROWS <= n <= MAX_ROWS:
        failures.append(f"row count {n} outside [{MIN_ROWS}, {MAX_ROWS}]")

    dupes = holdings.loc[holdings["ticker"].duplicated(), "ticker"].unique().tolist()
    if dupes:
        failures.append(f"duplicate tickers: {dupes}")

    missing = holdings[REQUIRED_COLUMNS].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        failures.append(f"missing values: {missing.to_dict()}")

    bad = holdings.loc[(holdings["price"] <= 0) | (holdings["market_value"] <= 0), "ticker"].tolist()
    if bad:
        failures.append(f"non-positive price or market value: {bad}")

    unknown = set(holdings["sector"].dropna()) - GICS_SECTORS
    if unknown:
        failures.append(f"unknown sectors: {sorted(unknown)}")

    age = (today - as_of).days
    if not 0 <= age <= MAX_AGE_DAYS:
        failures.append(f"as_of {as_of} is {age} days before today {today} (allowed 0-{MAX_AGE_DAYS})")

    if failures:
        raise ValueError("Holdings validation failed:\n- " + "\n- ".join(failures))


# ---------- Entry points ----------

def holdings_from_text(text: str, today: date) -> tuple[date, pd.DataFrame]:
    """Parse, filter and validate a holdings file's text."""
    as_of = parse_as_of_date(text)
    holdings = filter_equities(parse_holdings_table(text))
    validate_holdings(holdings, as_of, today)
    return as_of, holdings


def load_latest_holdings(today: date | None = None) -> tuple[date, pd.DataFrame]:
    """Download, archive and process the latest IWB holdings."""
    content = download_holdings()
    path = save_raw(content)
    logger.info("Saved raw holdings to %s", path)
    return holdings_from_text(content.decode("utf-8"), today or date.today())