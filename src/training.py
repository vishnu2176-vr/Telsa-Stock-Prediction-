"""
src/training.py
---------------
Trains the LSTM, saves / loads the model + scaler, and can run the whole
pipeline from the command line:

    python -m src.training --source sample --epochs 30
    python -m src.training --source yahoo  --start 2015-01-01 --end 2025-01-01
    python -m src.training --source csv    --csv path/to/file.csv
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import random

import joblib
import numpy as np
import pandas as pd

import config
from src import data_collection, evaluation, prediction, preprocessing
from src.lstm_model import build_model
from utils.helpers import DataError, ModelError, ensure_folders, load_json, now_text, save_json


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seeds(seed=config.SEED):
    """Fix every random number generator so a re-run gives (almost) the same result."""
    import keras
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)       # sets python, numpy and backend seeds


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def _make_progress_callback(total_epochs, on_epoch):
    """Wrap a simple function(epoch, total, logs) into a Keras callback."""
    import keras

    class EpochReporter(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if on_epoch is not None:
                on_epoch(epoch + 1, total_epochs, logs or {})

    return EpochReporter()


def train_model(prepared, epochs=config.DEFAULT_EPOCHS, on_epoch=None):
    """
    Train a new LSTM on the training part of `prepared`.
    on_epoch(epoch, total, logs) is called after each epoch (used for the progress bar).
    Returns (model, history_dict).
    """
    import keras

    epochs = int(min(max(epochs, config.MIN_EPOCHS), config.MAX_EPOCHS))
    set_seeds()

    try:
        model = build_model()
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,      # go back to the best epoch
        )
        history = model.fit(
            prepared.X_train, prepared.y_train,
            validation_data=(prepared.X_val, prepared.y_val),
            epochs=epochs,
            batch_size=config.BATCH_SIZE,
            shuffle=True,                   # shuffle the 60-day windows each epoch. This is safe:
                                            # every window is self-contained and the train/test
                                            # split itself stays chronological.
            callbacks=[early_stop, _make_progress_callback(epochs, on_epoch)],
            verbose=0,
        )
    except Exception as error:
        raise ModelError(f"Training failed ({type(error).__name__}: {error}).")

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    return model, history_dict


# ----------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------
def save_artifacts(model, scaler, metadata):
    """Save the model, the scaler and the metadata (history, metrics, data info)."""
    ensure_folders()
    try:
        model.save(config.MODEL_PATH)
        joblib.dump(scaler, config.SCALER_PATH)
        save_json(metadata, config.METADATA_PATH)
    except Exception as error:
        raise ModelError(f"Could not save the trained model ({type(error).__name__}: {error}).")


def saved_model_exists():
    return config.MODEL_PATH.exists() and config.SCALER_PATH.exists() and config.METADATA_PATH.exists()


def load_artifacts():
    """Load (model, scaler, metadata). Raises ModelError with a friendly message."""
    if not saved_model_exists():
        raise ModelError("No saved model was found yet. Open the 'LSTM Training' page and train one first.")
    try:
        import keras
        model = keras.models.load_model(config.MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        metadata = load_json(config.METADATA_PATH)
    except Exception as error:
        raise ModelError(
            "The saved model could not be loaded - it may be corrupted or made with a different "
            f"library version. Please train it again. ({type(error).__name__})"
        )
    return model, scaler, metadata


# ----------------------------------------------------------------------
# Whole pipeline: clean -> prepare -> train -> evaluate -> forecast -> save
# ----------------------------------------------------------------------
def run_pipeline(raw_df, source_label, epochs=config.DEFAULT_EPOCHS, on_epoch=None):
    """
    Run everything on a raw DataFrame and return a dictionary `results` with:
      clean_df, report, prepared, model, history, test_table, metrics, baseline,
      observations, forecast, metadata
    The model, scaler and metadata are also saved to disk.
    """
    clean_df, report = preprocessing.clean_data(raw_df)
    prepared = preprocessing.prepare_data(clean_df)

    model, history = train_model(prepared, epochs=epochs, on_epoch=on_epoch)

    test_table = prediction.predict_test(model, prepared)
    metrics = evaluation.calculate_metrics(test_table["Actual"], test_table["Predicted"])
    baseline = evaluation.baseline_metrics(prepared.actual_test_close, prepared.previous_close)
    observations = evaluation.make_observations(
        metrics, baseline, test_table["Actual"], test_table["Predicted"]
    )
    forecast = prediction.forecast_future(model, prepared.scaler, clean_df)

    metadata = {
        "trained_at": now_text(),
        "data_source": source_label,
        "epochs_requested": int(epochs),
        "epochs_run": len(history["loss"]),
        "best_epoch": int(np.argmin(history["val_loss"]) + 1),
        "final_train_loss": history["loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "history": history,
        "metrics": metrics,
        "baseline_metrics": baseline,
        "observations": observations,
        "rows": int(len(clean_df)),
        "first_date": clean_df["Date"].iloc[0],
        "last_date": clean_df["Date"].iloc[-1],
        "train_end_date": prepared.train_end_date,
        "test_start_date": prepared.test_start_date,
        "train_samples": int(len(prepared.X_train)),
        "validation_samples": int(len(prepared.X_val)),
        "test_samples": int(len(prepared.X_test)),
        "cleaning_report": report,
        "seed": config.SEED,
        "window": config.WINDOW,
        "features": config.FEATURES,
    }
    save_artifacts(model, prepared.scaler, metadata)

    # Save the tables too (sample output for the README / GitHub).
    ensure_folders()
    test_table.to_csv(config.TEST_PREDICTIONS_CSV, index=False, date_format="%Y-%m-%d")
    forecast.to_csv(config.FORECAST_CSV, index=False, date_format="%Y-%m-%d")
    save_json({"metrics": metrics, "baseline": baseline}, config.METRICS_JSON)

    return {
        "clean_df": clean_df, "report": report, "prepared": prepared, "model": model,
        "history": history, "test_table": test_table, "metrics": metrics,
        "baseline": baseline, "observations": observations, "forecast": forecast,
        "metadata": metadata,
    }


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------
def _load_source(args):
    if args.source == "yahoo":
        return data_collection.download_tsla(args.start, args.end), "Yahoo Finance (yfinance)"
    if args.source == "csv":
        if not args.csv:
            raise DataError("Please give a CSV path with --csv.")
        return data_collection.load_csv(args.csv), f"CSV file ({args.csv})"
    return data_collection.load_sample(), "Bundled sample file (data/sample_data.csv)"


def main():
    parser = argparse.ArgumentParser(description="Train the TSLA LSTM from the command line.")
    parser.add_argument("--source", choices=["sample", "yahoo", "csv"], default="sample")
    parser.add_argument("--csv", help="path to a CSV (with --source csv)")
    parser.add_argument("--start", default=config.DEFAULT_START_DATE)
    parser.add_argument("--end", default=str(pd.Timestamp.today().date()))
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    args = parser.parse_args()

    try:
        raw, label = _load_source(args)
        print(f"Data source : {label}")

        def show(epoch, total, logs):
            print(f"  epoch {epoch:>3}/{total}  loss={logs.get('loss', 0):.5f}  val_loss={logs.get('val_loss', 0):.5f}")

        results = run_pipeline(raw, label, epochs=args.epochs, on_epoch=show)
    except (DataError, ModelError) as error:
        print(f"\nERROR: {error}")
        raise SystemExit(1)

    from utils.charts import save_all_graphs
    save_all_graphs(results)

    meta = results["metadata"]
    print(f"\nRows used     : {meta['rows']}  ({meta['first_date']} -> {meta['last_date']})")
    print(f"Epochs run    : {meta['epochs_run']} (best epoch {meta['best_epoch']})")
    for name, value in results["metrics"].items():
        print(f"{name:<5}: {value:,.4f}")
    print("Baseline (tomorrow = today):", {k: round(v, 4) for k, v in results["baseline"].items()})
    forecast = results["forecast"].copy()
    forecast["Date"] = forecast["Date"].dt.strftime("%Y-%m-%d")
    print("\nForecast:\n", forecast.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
