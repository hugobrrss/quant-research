from datetime import date

import pandas as pd
import pytest

from src.data_pipelines.universe.ishares_holdings import (
    GICS_SECTORS,
    filter_equities,
    parse_as_of_date,
    parse_holdings_table,
    validate_holdings,
)

FOOTER = '"The content contained herein is owned or licensed by BlackRock, and/or its third-party providers."\n'

FIXTURE = (
    'iShares Russell 1000 ETF\n'
    'Fund Holdings as of,"Nov 20, 2026"\n'
    'Inception Date,"May 15, 2000"\n'
    'Shares Outstanding,"116,250,000.00"\n'
    'Stock,"-"\n'
    'Bond,"-"\n'
    'Cash,"-"\n'
    'Other,"-"\n'
    '\n'
    'Ticker,Name,Type,Sector,Asset Class,Market Value,Price,Location,Exchange\n'
    '"AAPL","APPLE","EQUITY","Information Technology","Equity","3,548,169,884.32","356.27","United States","NASDAQ"\n'
    '"BRK B","BERKSHIRE HATHAWAY CLASS B","EQUITY","Financials","Equity","1,200,000.00","480.10","United States","NYSE"\n'
    '"DDS","DILLARDS CLASS A","EQUITY","Consumer Discretionary","Equity","900,000.00","410.00","United States","TXE"\n'
    '"XYZ-W","XYZ WARRANT","EQUITY","Industrials","Equity","1,000.00","0.50","United States","NYSE"\n'
    '"HOLX","HOLOGIC INC","EQUITY","Health Care","Equity","1,574.05","0.01","United States","NO MARKET (E.G. UNLISTED)"\n'
    '"USD","USD CASH","CASH","Cash and/or Derivatives","Cash","20,031,512.22","100.00","United States","-"\n'
    '"ESZ6","S&P500 EMINI DEC 26","FUTURES","Cash and/or Derivatives","Futures","0.00","7,704.25","-","Index And Options Market"\n'
    '\n'
    + FOOTER
)

# --- Test parsing ---
def test_parse_as_of_date_reads_label():
    assert parse_as_of_date(FIXTURE) == date(2026, 11, 20)


def test_parse_as_of_date_missing_label_raises():
    text = FIXTURE.replace("Fund Holdings as of", "Something else")
    with pytest.raises(ValueError, match="Fund Holdings as of"):
        parse_as_of_date(text)

def test_parse_as_of_date_bad_format_raises():
    text = FIXTURE.replace("Nov 20, 2026","20/11/2026")
    with pytest.raises(ValueError):
        parse_as_of_date(text)

def test_parse_table_stops_before_footer():
    assert parse_holdings_table(FIXTURE).shape[0] == 7
    assert parse_holdings_table(FIXTURE)["Ticker"].iloc[-1] == "ESZ6"

def test_parse_table_without_footer():
    text = FIXTURE.replace("\n"+FOOTER, "")
    assert parse_holdings_table(text).shape[0] == 7

def test_parse_table_windows_line_endings():
    text = FIXTURE.replace("\n", "\r\n")
    assert parse_holdings_table(text).shape[0] == 7

def test_parse_table_missing_header_raises():
    text = FIXTURE.replace("Ticker", "Symbol")
    with pytest.raises(ValueError, match="header"):
        parse_holdings_table(text)

# --- Test filter ---

EXPECTED_COLUMNS = ["ticker", "name", "sector", "market_value", "price", "exchange", "location"]

def test_filter_keeps_tradable_equities():
    eq = filter_equities(parse_holdings_table(FIXTURE))
    assert set(eq["ticker"])=={"AAPL", "BRK B", "DDS"}
    assert eq.columns.to_list()==EXPECTED_COLUMNS

# --- Test validation ---

# synthetic data generation
def make_holdings(n: int = 1000) -> pd.DataFrame:
    """Clean synthetic holdings that pass every validation check."""
    sectors = sorted(GICS_SECTORS)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "name": [f"COMPANY {i}" for i in range(n)],
        "sector": [sectors[i % len(sectors)] for i in range(n)],
        "market_value": 1e6,
        "price": 10.0,
        "exchange": "NYSE",
        "location": "United States",
    })

AS_OF = date(2026, 11, 20)
TODAY = date(2026, 11, 22)

def test_validate_passes_on_clean_data():
    validate_holdings(make_holdings(),AS_OF,TODAY)

def test_validate_row_count():
    with pytest.raises(ValueError, match="500"):
        validate_holdings(make_holdings(500),AS_OF,TODAY)

def test_validate_duplicates():
    holdings = make_holdings()
    holdings.loc[1,"ticker"] = "T0"
    with pytest.raises(ValueError, match="duplicate"):
        validate_holdings(holdings,AS_OF,TODAY)

def test_validate_missing():
    holdings = make_holdings()
    holdings.loc[1,"price"] = None
    with pytest.raises(ValueError, match="missing"):
        validate_holdings(holdings,AS_OF,TODAY)

def test_validate_non_positive():
    holdings = make_holdings()
    holdings.loc[1,"price"] = 0
    with pytest.raises(ValueError, match="positive"):
        validate_holdings(holdings,AS_OF,TODAY)


def test_validate_unknown_sector():
    holdings = make_holdings()
    holdings.loc[1,"sector"] = "Tech"
    with pytest.raises(ValueError, match="sectors"):
        validate_holdings(holdings,AS_OF,TODAY)

def test_validate_reports_all_failures():
    holdings = make_holdings(500)
    holdings.loc[1,"sector"] = "Tech"
    with pytest.raises(ValueError) as exc:
        validate_holdings(holdings,AS_OF,today=date(2026, 12, 31))
        msg = str(exc.value)
        assert "row count" in msg
        assert "unknown sectors" in msg
        assert "days before today" in msg

def test_validate_stale_as_of():
    holdings = make_holdings()
    with pytest.raises(ValueError, match="as_of"):
        validate_holdings(holdings,as_of=date(2026,11,4),today=date(2026, 11, 22))

def test_validate_future_as_of():
    holdings = make_holdings()
    with pytest.raises(ValueError, match="as_of"):
        validate_holdings(holdings,as_of=date(2026,11,24),today=date(2026, 11, 22))