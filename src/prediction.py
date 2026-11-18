"""
src/prediction.py
-----------------
Uses a trained model to
  * predict the test period (so we can compare with reality), and
  * forecast the next few trading days (recursive forecasting).
"""

import numpy as np
import pandas as pd

import config
from src.preprocessing import inverse_close
from utils.helpers import DataError, ModelError


def predict_test(model, prepared):
    """
    Predict every test day. Returns a DataFrame in dollars with
    Date, Actual, Predicted, Error (= Predicted - Actual), Abs_Error.
    """
    try:
        scaled_pred = model.predict(prepared.X_test, verbose=0).reshape(-1)
    except Exception as error:
        raise ModelError(f"The model could not make predictions ({type(error).__name__}: {error}).")

    # Convert back to dollars BEFORE any metric is calculated.
    predicted = inverse_close(scaled_pred, prepared.scaler, prepared.close_index)
    actual = prepared.actual_test_close

    result = pd.DataFrame(
        {
            "Date": prepared.test_dates,
            "Actual": actual,
            "Predicted": predicted,
        }
    )
    result["Error"] = result["Predicted"] - result["Actual"]
    result["Abs_Error"] = result["Error"].abs()
    return result


def forecast_future(model, scaler, df, days=config.FORECAST_DAYS):
    """
    Recursive multi-day forecast.

    1. Take the last 60 real days.
    2. Predict the next day's Close.
    3. Add that predicted day to the end of the window and drop the oldest day.
    4. Repeat until `days` predictions exist.

    The model needs all 5 features for each day, but it only predicts Close.
    For a predicted day we therefore set Open = High = Low = Close = predicted Close
    and keep Volume equal to the last real volume. This is a simplification
    (mentioned in the README) and it is one reason long forecasts get less reliable.
    """
    if len(df) < config.WINDOW:
        raise DataError(
            f"At least {config.WINDOW} records are needed to forecast, but only {len(df)} are loaded."
        )

    price_index = {name: i for i, name in enumerate(config.FEATURES)}
    price_columns = [price_index[c] for c in ("Open", "High", "Low", "Close")]
    close_index = price_index[config.TARGET]

    values = df[config.FEATURES].to_numpy(dtype=float)
    window = scaler.transform(values)[-config.WINDOW:].copy()

    scaled_predictions = []
    for _ in range(days):
        batch = window.reshape(1, config.WINDOW, len(config.FEATURES))
        try:
            next_close = float(model.predict(batch, verbose=0)[0, 0])
        except Exception as error:
            raise ModelError(f"The model could not make a forecast ({type(error).__name__}: {error}).")
        scaled_predictions.append(next_close)

        new_row = window[-1].copy()            # keeps the last Volume
        new_row[price_columns] = next_close
        window = np.vstack([window[1:], new_row])

    prices = inverse_close(scaled_predictions, scaler, close_index)

    last_date = pd.Timestamp(df["Date"].iloc[-1])
    last_close = float(df[config.TARGET].iloc[-1])
    # Future dates = next weekdays (stock-market holidays are not removed - kept simple).
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days)

    forecast = pd.DataFrame(
        {
            "Day": range(1, days + 1),
            "Date": future_dates,
            "Predicted_Close": prices,
        }
    )
    forecast["Change_vs_Last_Close"] = forecast["Predicted_Close"] - last_close
    forecast["Change_Percent"] = forecast["Change_vs_Last_Close"] / last_close * 100
    return forecast
