"""
src/evaluation.py
-----------------
Measures how good the predictions are. Every number comes from real model
predictions - nothing here is typed in by hand.

All inputs must already be in dollars (the original price scale),
NOT the 0-1 scaled values.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Plain-language explanations shown in the app and in the README.
METRIC_EXPLANATIONS = {
    "MAE": (
        "Mean Absolute Error - the average size of the mistakes, in dollars. "
        "If MAE is $10, the prediction is on average $10 away from the real price. "
        "Lower is better."
    ),
    "MSE": (
        "Mean Squared Error - the average of the squared mistakes (dollars squared). "
        "Squaring makes big mistakes count much more than small ones, "
        "so it is useful for spotting large misses. Lower is better."
    ),
    "RMSE": (
        "Root Mean Squared Error - the square root of MSE, so it is back in dollars. "
        "It is like MAE but punishes big mistakes more. "
        "RMSE is always equal to or larger than MAE. Lower is better."
    ),
}


def calculate_metrics(actual, predicted):
    """Return MAE, MSE and RMSE for dollar-scale arrays."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": float(np.sqrt(mse))}


def baseline_metrics(actual, previous_close):
    """
    A very simple comparison model: 'tomorrow's price = today's price'.
    If the LSTM cannot beat this, it has not learned anything useful beyond the last price.
    """
    return calculate_metrics(actual, previous_close)


def make_observations(metrics, baseline, actual, predicted):
    """
    Write short observations about the results. Every sentence is built from the
    real numbers of this run, so the wording changes if the results change.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    average_price = float(np.mean(actual))
    mae, rmse = metrics["MAE"], metrics["RMSE"]
    notes = []

    notes.append(
        f"On the test period the predictions were off by ${mae:,.2f} on average (MAE). "
        f"That is about {mae / average_price * 100:.1f}% of the average actual price "
        f"(${average_price:,.2f})."
    )

    if mae > 0 and rmse / mae > 1.3:
        notes.append(
            f"RMSE (${rmse:,.2f}) is clearly larger than MAE (${mae:,.2f}), "
            "which means a few days had much bigger errors than the rest."
        )
    else:
        notes.append(
            f"RMSE (${rmse:,.2f}) is close to MAE (${mae:,.2f}), "
            "so the errors are fairly evenly spread - there are no huge outlier misses."
        )

    base_rmse = baseline["RMSE"]
    if rmse < base_rmse:
        notes.append(
            f"The LSTM beat the simple 'tomorrow = today' baseline "
            f"(RMSE ${rmse:,.2f} vs ${base_rmse:,.2f})."
        )
    else:
        notes.append(
            f"The LSTM did NOT beat the simple 'tomorrow = today' baseline "
            f"(RMSE ${rmse:,.2f} vs ${base_rmse:,.2f}). This is common: stock prices change "
            "very little from one day to the next, so repeating yesterday's price is hard to beat."
        )

    bias = float(np.mean(predicted - actual))
    direction = "higher" if bias > 0 else "lower"
    notes.append(
        f"On average the model predicted ${abs(bias):,.2f} {direction} than the real price, "
        "so it tends to "
        + ("over-estimate" if bias > 0 else "under-estimate")
        + " during this test period."
    )

    # Did the model get the direction (up / down) right?
    actual_move = np.diff(actual)
    predicted_move = np.diff(predicted)
    if len(actual_move) > 0:
        hit = float(np.mean(np.sign(actual_move) == np.sign(predicted_move))) * 100
        notes.append(
            f"The predicted day-to-day direction (up or down) matched the real direction "
            f"on {hit:.1f}% of test days."
        )
    return notes
