"""
src/preprocessing.py
--------------------
Turns raw stock data into the arrays the LSTM needs.

Steps (in order):
  1. clean_data()        - dates, numbers, duplicates, missing values, sorting
  2. prepare_data()      - chronological 80/20 split, scaling (fit on training only),
                           60-day sequences, validation set
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config
from utils.helpers import DataError

NUMBER_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]


# ======================================================================
# Step 1 - cleaning
# ======================================================================
def clean_data(df):
    """
    Clean a raw DataFrame. Returns (clean_df, report).
    `report` is a dictionary that says what was fixed (shown on the dashboard).
    """
    if df is None or len(df) == 0:
        raise DataError("The dataset is empty - there is nothing to process.")
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError("The data is missing required column(s): " + ", ".join(missing) + ".")

    df = df.copy()
    report = {"rows_in": int(len(df))}

    # --- Dates -------------------------------------------------------
    date_text = df["Date"].astype(str)
    # Some files contain a row whose "date" is only a time such as "11:34"
    # (an unfinished trading day). pandas would silently turn it into today's date,
    # so we remove such rows first.
    time_only = date_text.str.fullmatch(r"\s*\d{1,2}:\d{2}(:\d{2})?\s*")
    dates = pd.to_datetime(df["Date"].where(~time_only), errors="coerce", format="mixed")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    df["Date"] = dates.dt.normalize()

    bad_dates = int(df["Date"].isna().sum())
    df = df.dropna(subset=["Date"])
    report["invalid_date_rows_removed"] = bad_dates
    if df.empty:
        raise DataError("None of the values in the Date column could be read as dates.")

    # --- Numbers (remove $ and thousands commas, then convert) ---------
    for col in NUMBER_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.replace(r"[$,\s]", "", regex=True)
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Duplicates ----------------------------------------------------
    before = len(df)
    df = df.drop_duplicates()                              # identical rows
    df = df.drop_duplicates(subset="Date", keep="last")    # same day twice
    report["duplicates_removed"] = int(before - len(df))

    # --- Sort oldest -> newest ---------------------------------------
    df = df.sort_values("Date").reset_index(drop=True)

    # --- Missing values ----------------------------------------------
    price_volume_cols = [c for c in NUMBER_COLUMNS if c in df.columns]
    report["missing_values_found"] = int(df[price_volume_cols].isna().sum().sum())
    # Forward-fill uses yesterday's value (never tomorrow's), so no future leak.
    df[price_volume_cols] = df[price_volume_cols].ffill()
    rows_before = len(df)
    df = df.dropna(subset=config.FEATURES)                 # e.g. missing at the very start
    report["rows_dropped_missing"] = int(rows_before - len(df))

    # --- Impossible values ------------------------------------------------
    rows_before = len(df)
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1) & (df["Volume"] >= 0)]
    report["rows_dropped_invalid"] = int(rows_before - len(df))

    df = df.reset_index(drop=True)
    if df.empty:
        raise DataError("No valid rows are left after cleaning the data.")

    report["rows_out"] = int(len(df))
    return df, report


# ======================================================================
# Step 2 - scaling, sequences and the chronological split
# ======================================================================
def check_enough_data(df):
    """Raise a friendly error if there are too few rows to train the LSTM."""
    needed = max(config.MIN_RECORDS, config.WINDOW + 20)
    if len(df) < needed:
        raise DataError(
            f"Only {len(df)} records are available, but at least {needed} are needed "
            f"(the model looks back {config.WINDOW} trading days and also needs data for "
            "training and testing). Choose a longer date range or a bigger file."
        )


def create_sequences(scaled, window, target_index):
    """
    Build samples from a scaled 2-D array (rows = days, columns = features).
    Sample k uses rows k .. k+window-1 as input and the target-column value
    of row k+window as the answer.
    """
    inputs, answers = [], []
    for i in range(window, len(scaled)):
        inputs.append(scaled[i - window:i, :])
        answers.append(scaled[i, target_index])
    return np.array(inputs), np.array(answers)


@dataclass
class PreparedData:
    """Everything the training / evaluation code needs, in one object."""
    scaler: MinMaxScaler
    close_index: int
    split_row: int            # first row (day) that belongs to the test period
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_dates: pd.Series     # the date each test prediction is for
    actual_test_close: np.ndarray   # real test closing prices (original scale)
    previous_close: np.ndarray      # closing price of the day before each test day
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp


def prepare_data(df):
    """
    Split, scale and window the cleaned data without data leakage.

    * The first 80% of the DAYS are training data, the last 20% are test data
      (a time-ordered split - never shuffled).
    * The MinMaxScaler is fitted on the training days only.
    * A test sample's 60-day input window may reach back into the training period
      (that is just the past, so it is fine), but no training sample ever sees a test day.
    """
    check_enough_data(df)

    values = df[config.FEATURES].to_numpy(dtype=float)
    split_row = int(len(df) * config.TRAIN_RATIO)

    scaler = MinMaxScaler()
    scaler.fit(values[:split_row])            # <-- training days only
    scaled = scaler.transform(values)

    close_index = config.FEATURES.index(config.TARGET)
    X_all, y_all = create_sequences(scaled, config.WINDOW, close_index)

    # Sample k predicts row k+WINDOW. Training samples must predict rows < split_row.
    n_train_samples = split_row - config.WINDOW
    X_train_full, y_train_full = X_all[:n_train_samples], y_all[:n_train_samples]
    X_test, y_test = X_all[n_train_samples:], y_all[n_train_samples:]

    # Validation = the LAST part of the training samples (still chronological).
    n_val = max(1, int(len(X_train_full) * config.VALIDATION_RATIO))
    X_train, y_train = X_train_full[:-n_val], y_train_full[:-n_val]
    X_val, y_val = X_train_full[-n_val:], y_train_full[-n_val:]

    close = df[config.TARGET].to_numpy(dtype=float)
    return PreparedData(
        scaler=scaler,
        close_index=close_index,
        split_row=split_row,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        test_dates=df["Date"].iloc[split_row:].reset_index(drop=True),
        actual_test_close=close[split_row:],
        previous_close=close[split_row - 1:-1],
        train_end_date=df["Date"].iloc[split_row - 1],
        test_start_date=df["Date"].iloc[split_row],
    )


def inverse_close(scaled_values, scaler, close_index):
    """
    Turn scaled Close values (0..1) back into dollars.
    MinMaxScaler stores: scaled = price * scale_ + min_  ->  price = (scaled - min_) / scale_
    """
    scaled_values = np.asarray(scaled_values, dtype=float)
    return (scaled_values - scaler.min_[close_index]) / scaler.scale_[close_index]
