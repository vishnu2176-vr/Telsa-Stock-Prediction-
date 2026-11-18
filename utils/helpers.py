"""
utils/helpers.py
----------------
Small helper functions and custom error classes used all over the project.
"""

import json
from datetime import date, datetime

import numpy as np
import pandas as pd

import config


# ----------------------------------------------------------------------
# Custom errors. The app catches these and shows the message to the user
# instead of crashing, so every message must be easy to understand.
# ----------------------------------------------------------------------
class AppError(Exception):
    """Base class for friendly, user-readable errors."""


class DataError(AppError):
    """Problem with the stock data (download, CSV, columns, dates...)."""


class ModelError(AppError):
    """Problem with the model (missing, corrupted, training failed...)."""


# ----------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------
def ensure_folders():
    """Create the output and model folders if they do not exist yet."""
    for folder in (config.MODEL_DIR, config.GRAPH_DIR, config.PREDICTION_DIR, config.DATA_DIR):
        folder.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------
def money(value):
    """12.3456 -> '$12.35'"""
    return f"${value:,.2f}"


def signed_money(value):
    """-1.5 -> '-$1.50', 2 -> '+$2.00'"""
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def signed_percent(value):
    """1.234 -> '+1.23%'"""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def now_text():
    """Current date and time as text (used when we save a model)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------
# Is the latest price historical, delayed or current?
# ----------------------------------------------------------------------
def data_status(last_date, source_kind):
    """
    Decide how fresh the data is.

    Returns (label, explanation) where label is 'Historical', 'Delayed' or 'Current'.
      - Current    : the data contains the most recent trading day.
      - Delayed    : the data is exactly one trading day behind.
      - Historical : the data is older than that (e.g. the bundled sample file).
    Weekends are skipped; stock-market holidays are not (kept simple on purpose).
    """
    last_day = pd.Timestamp(last_date).date()

    if source_kind == "sample":
        return "Historical", f"Bundled sample file - the data ends on {last_day}."

    today = date.today()
    # If today is a weekend, the "latest possible" trading day is the previous Friday.
    latest_possible = np.busday_offset(today, 0, roll="backward")
    gap = int(np.busday_count(last_day, latest_possible))

    if gap <= 0:
        return "Current", (
            f"Includes the latest trading day ({last_day}). "
            "Yahoo Finance quotes can be about 15 minutes delayed while the market is open."
        )
    if gap == 1:
        return "Delayed", f"One trading day behind - the last record is {last_day}."
    return "Historical", f"The last record is {last_day} ({gap} trading days ago)."


# ----------------------------------------------------------------------
# Saving JSON that may contain numpy numbers
# ----------------------------------------------------------------------
def _to_plain(value):
    """Convert numpy / pandas values to plain Python so json can save them."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return str(value)[:10]
    raise TypeError(f"Cannot save {type(value)} to JSON")


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=_to_plain)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
