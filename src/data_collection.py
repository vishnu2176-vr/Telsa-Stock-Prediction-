"""
src/data_collection.py
----------------------
Gets TSLA stock data from three places:
  1. Yahoo Finance  (live download with the yfinance library)
  2. A CSV file the user uploads (fallback when the internet / Yahoo is down)
  3. The small sample file that ships with the project (data/sample_data.csv)

Every function either returns a pandas DataFrame or raises a DataError with a
friendly message. Nothing in this file depends on Streamlit.
"""

from datetime import date, timedelta

import pandas as pd

import config
from utils.helpers import DataError


# ----------------------------------------------------------------------
# Column name clean-up
# ----------------------------------------------------------------------
# Different websites name columns differently ("close", "Close/Last", "adj_close"...).
# We turn each name into lower-case letters only, then look it up here.
_COLUMN_ALIASES = {
    "date": "Date",
    "datetime": "Date",
    "timestamp": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "closelast": "Close",       # Nasdaq.com export
    "adjclose": "Adj Close",
    "adjustedclose": "Adj Close",
    "volume": "Volume",
}


def standardize_columns(df):
    """Rename columns to Date, Open, High, Low, Close, Adj Close, Volume; drop the rest."""
    renamed = {}
    for name in df.columns:
        key = "".join(ch for ch in str(name).lower() if ch.isalpha())
        if key in _COLUMN_ALIASES and _COLUMN_ALIASES[key] not in renamed.values():
            renamed[name] = _COLUMN_ALIASES[key]

    df = df.rename(columns=renamed)
    keep = [c for c in config.REQUIRED_COLUMNS + config.OPTIONAL_COLUMNS if c in df.columns]
    return df[keep].copy()


def check_required_columns(df):
    """Raise a friendly error if Date/Open/High/Low/Close/Volume are not all present."""
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(
            "The data is missing required column(s): " + ", ".join(missing) + ". "
            "A valid file needs these columns: " + ", ".join(config.REQUIRED_COLUMNS) + "."
        )


# ----------------------------------------------------------------------
# 1. Yahoo Finance
# ----------------------------------------------------------------------
def validate_date_range(start, end):
    """Check the dates the user picked. Returns (start, end) as dates."""
    try:
        start = pd.Timestamp(start).date()
        end = pd.Timestamp(end).date()
    except (ValueError, TypeError):
        raise DataError("Please enter valid start and end dates.")

    today = date.today()
    if start >= end:
        raise DataError("The start date must be earlier than the end date.")
    if start > today:
        raise DataError("The start date is in the future. Please pick a date in the past.")
    if end > today:
        end = today                     # quietly clip to today
    if end < pd.Timestamp(config.TSLA_IPO_DATE).date():
        raise DataError(f"TSLA only started trading on {config.TSLA_IPO_DATE}. Pick a later end date.")
    return start, end


def download_tsla(start, end):
    """
    Download daily TSLA prices from Yahoo Finance.
    Returns a DataFrame with Date, Open, High, Low, Close, Adj Close, Volume.
    """
    start, end = validate_date_range(start, end)

    try:
        import yfinance as yf   # imported here so the rest of the project works without it
    except ImportError:
        raise DataError("The 'yfinance' library is not installed. Run: pip install yfinance")

    try:
        # Keep yfinance's SQLite databases beside the project data. The default user
        # cache directory may not be writable in hosted or managed environments.
        cache_dir = config.DATA_DIR / ".yfinance_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))

        # yfinance treats 'end' as exclusive, so add one day to include the end date.
        raw = yf.download(
            config.SYMBOL,
            start=str(start),
            end=str(end + timedelta(days=1)),
            auto_adjust=False,      # keep both Close and Adj Close
            progress=False,
            timeout=20,
        )
    except Exception as error:      # network errors, rate limits, parsing errors ...
        raise DataError(
            "Could not download data from Yahoo Finance. Check your internet connection "
            "and try again, or upload a CSV file / use the sample data instead. "
            f"(Technical detail: {type(error).__name__})"
        )

    if raw is None or raw.empty:
        raise DataError(
            "Yahoo Finance returned no data. This usually means there is no internet "
            "connection, Yahoo is temporarily blocking requests, or the chosen date range "
            "has no trading days. You can try again later, upload a CSV file, or use the sample data."
        )

    # Newer yfinance versions return two-level column names like ('Close', 'TSLA').
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    df = standardize_columns(raw)
    check_required_columns(df)
    return df


# ----------------------------------------------------------------------
# 2. CSV file (upload or path)
# ----------------------------------------------------------------------
def load_csv(file_or_path):
    """
    Read a stock CSV (an uploaded file object or a path).
    Raises DataError for empty, unreadable or wrongly-formatted files.
    """
    try:
        df = pd.read_csv(file_or_path)
    except pd.errors.EmptyDataError:
        raise DataError("The CSV file is empty.")
    except (pd.errors.ParserError, UnicodeDecodeError):
        raise DataError("The file could not be read as a CSV. Please upload a plain .csv file.")
    except FileNotFoundError:
        raise DataError("The CSV file was not found.")
    except Exception as error:
        raise DataError(f"The CSV file could not be read ({type(error).__name__}).")

    if df.empty:
        raise DataError("The CSV file has headers but no rows.")

    df = standardize_columns(df)
    check_required_columns(df)
    return df


# ----------------------------------------------------------------------
# 3. Bundled sample
# ----------------------------------------------------------------------
def load_sample():
    """Load data/sample_data.csv (real TSLA prices, 2015-10-15 to 2018-10-15)."""
    if not config.SAMPLE_CSV.exists():
        raise DataError("The sample file data/sample_data.csv is missing from the project.")
    return load_csv(config.SAMPLE_CSV)


# ----------------------------------------------------------------------
# Quick facts about a data set (shown on the dashboard)
# ----------------------------------------------------------------------
def dataset_summary(df):
    """Return a small dictionary describing a DataFrame with a Date column."""
    dates = pd.to_datetime(df["Date"])
    return {
        "rows": int(len(df)),
        "first_date": dates.min().date(),
        "last_date": dates.max().date(),
        "columns": list(df.columns),
        "missing_values": int(df.isna().sum().sum()),
    }
