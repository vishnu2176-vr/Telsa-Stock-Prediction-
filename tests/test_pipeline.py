"""
tests/test_pipeline.py
----------------------
Automatic checks for the whole project. Run from the project folder with:

    python -m unittest discover -s tests -v

(Uses only the Python standard library, so nothing extra to install.)
"""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from src import data_collection, evaluation, prediction, preprocessing, training
from src.lstm_model import build_model
from utils.helpers import DataError, ModelError


def small_frame(rows=300, seed=0):
    """A synthetic price table, used ONLY to test code paths (never shown in the app)."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, rows))
    return pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=rows),
        "Open": close - 0.5, "High": close + 1, "Low": close - 1, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, rows),
    })


class DataLoadingTests(unittest.TestCase):
    def test_empty_csv(self):
        with self.assertRaises(DataError):
            data_collection.load_csv(io.StringIO(""))

    def test_header_only_csv(self):
        with self.assertRaises(DataError):
            data_collection.load_csv(io.StringIO("Date,Open,High,Low,Close,Volume\n"))

    def test_missing_columns(self):
        with self.assertRaises(DataError) as ctx:
            data_collection.load_csv(io.StringIO("Date,Open,Close\n2020-01-01,1,2\n"))
        self.assertIn("High", str(ctx.exception))

    def test_garbage_file(self):
        with self.assertRaises(DataError):
            data_collection.load_csv(io.StringIO("\x00\x01 not,a\nreal,,,csv\"\""))

    def test_alternative_column_names(self):
        text = "Date,Close/Last,Volume,Open,High,Low\n01/02/2020,$10.5,1000,10,11,9\n"
        df = data_collection.load_csv(io.StringIO(text))
        self.assertEqual(set(config.REQUIRED_COLUMNS), set(df.columns))

    def test_sample_file_loads(self):
        df = data_collection.load_sample()
        self.assertGreater(len(df), 500)

    def test_invalid_date_ranges(self):
        with self.assertRaises(DataError):
            data_collection.validate_date_range("2022-01-01", "2021-01-01")
        with self.assertRaises(DataError):
            data_collection.validate_date_range("2999-01-01", "3000-01-01")
        with self.assertRaises(DataError):
            data_collection.validate_date_range("not a date", "2021-01-01")


class CleaningTests(unittest.TestCase):
    def test_cleaning_fixes_everything(self):
        df = small_frame(300)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")           # dates as text
        df = pd.concat([df, df.iloc[[5, 6]]])                     # duplicates
        df = df.sample(frac=1, random_state=1)                    # unsorted
        df.loc[df.index[10], "Close"] = np.nan                    # missing value
        df.loc[len(df)] = ["11:34", 1, 1, 1, 1, 1]                # time-only 'date' row
        clean, report = preprocessing.clean_data(df)
        self.assertTrue(clean["Date"].is_monotonic_increasing)
        self.assertEqual(clean["Date"].duplicated().sum(), 0)
        self.assertEqual(clean[config.FEATURES].isna().sum().sum(), 0)
        self.assertEqual(report["invalid_date_rows_removed"], 1)
        self.assertEqual(len(clean), 300)

    def test_empty_frame(self):
        with self.assertRaises(DataError):
            preprocessing.clean_data(pd.DataFrame(columns=config.REQUIRED_COLUMNS))

    def test_too_few_records(self):
        clean, _ = preprocessing.clean_data(small_frame(100))
        with self.assertRaises(DataError) as ctx:
            preprocessing.prepare_data(clean)
        self.assertIn("at least", str(ctx.exception))


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, _ = preprocessing.clean_data(small_frame(400, seed=3))
        cls.p = preprocessing.prepare_data(cls.df)

    def test_shapes(self):
        self.assertEqual(self.p.X_train.shape[1:], (config.WINDOW, len(config.FEATURES)))
        self.assertEqual(self.p.X_test.shape[1:], (config.WINDOW, len(config.FEATURES)))

    def test_scaler_fit_on_training_days_only(self):
        train_rows = self.df[config.FEATURES].iloc[:self.p.split_row]
        np.testing.assert_allclose(self.p.scaler.data_max_, train_rows.max().to_numpy())
        np.testing.assert_allclose(self.p.scaler.data_min_, train_rows.min().to_numpy())

    def test_split_is_chronological_and_80_20(self):
        self.assertEqual(self.p.split_row, int(len(self.df) * 0.8))
        self.assertLess(self.p.train_end_date, self.p.test_start_date)
        self.assertEqual(len(self.p.test_dates), len(self.df) - self.p.split_row)
        self.assertEqual(len(self.p.X_test), len(self.p.test_dates))

    def test_no_training_target_comes_from_test_period(self):
        # Targets of train+validation samples must be exactly the Close of rows WINDOW..split_row-1.
        close = self.df["Close"].to_numpy()
        expected = self.p.scaler.min_[self.p.close_index] + close[config.WINDOW:self.p.split_row] \
            * self.p.scaler.scale_[self.p.close_index]
        got = np.concatenate([self.p.y_train, self.p.y_val])
        np.testing.assert_allclose(got, expected)

    def test_test_targets_are_test_days(self):
        expected = self.p.scaler.min_[self.p.close_index] + self.p.actual_test_close \
            * self.p.scaler.scale_[self.p.close_index]
        np.testing.assert_allclose(self.p.y_test, expected)

    def test_inverse_scaling_roundtrip(self):
        scaled = self.p.scaler.transform(self.df[config.FEATURES].to_numpy())[:, self.p.close_index]
        back = preprocessing.inverse_close(scaled, self.p.scaler, self.p.close_index)
        np.testing.assert_allclose(back, self.df["Close"].to_numpy())


class EvaluationTests(unittest.TestCase):
    def test_metrics_match_manual_calculation(self):
        actual, predicted = np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 33.0])
        m = evaluation.calculate_metrics(actual, predicted)
        self.assertAlmostEqual(m["MAE"], (2 + 2 + 3) / 3)
        self.assertAlmostEqual(m["MSE"], (4 + 4 + 9) / 3)
        self.assertAlmostEqual(m["RMSE"], ((4 + 4 + 9) / 3) ** 0.5)


class ModelTests(unittest.TestCase):
    def test_architecture_matches_brief(self):
        model = build_model()
        kinds = [l.__class__.__name__ for l in model.layers]
        self.assertEqual(kinds, ["LSTM", "Dropout", "LSTM", "Dropout", "Dense", "Dense"])
        self.assertEqual(model.layers[0].units, 50)
        self.assertEqual(model.layers[1].rate, 0.2)
        self.assertEqual(model.layers[4].units, 25)
        self.assertEqual(model.input_shape, (None, 60, 5))
        self.assertEqual(model.output_shape, (None, 1))

    def test_forecast_mechanics(self):
        df, _ = preprocessing.clean_data(small_frame(300))
        p = preprocessing.prepare_data(df)
        model = build_model()
        f1 = prediction.forecast_future(model, p.scaler, df, days=7)
        f2 = prediction.forecast_future(model, p.scaler, df, days=7)
        self.assertEqual(len(f1), 7)
        self.assertTrue(f1["Date"].is_monotonic_increasing)
        self.assertGreater(f1["Date"].iloc[0], df["Date"].iloc[-1])
        np.testing.assert_allclose(f1["Predicted_Close"], f2["Predicted_Close"])   # deterministic

    def test_forecast_needs_enough_history(self):
        df, _ = preprocessing.clean_data(small_frame(300))
        p = preprocessing.prepare_data(df)
        with self.assertRaises(DataError):
            prediction.forecast_future(build_model(), p.scaler, df.head(30))

    def test_missing_saved_model(self):
        original = (config.MODEL_PATH, config.SCALER_PATH, config.METADATA_PATH)
        with tempfile.TemporaryDirectory() as folder:
            config.MODEL_PATH = Path(folder) / "none.keras"
            config.SCALER_PATH = Path(folder) / "none.pkl"
            config.METADATA_PATH = Path(folder) / "none.json"
            try:
                with self.assertRaises(ModelError):
                    training.load_artifacts()
            finally:
                config.MODEL_PATH, config.SCALER_PATH, config.METADATA_PATH = original


@unittest.skipUnless(training.saved_model_exists(), "run training first to test the saved model")
class SavedModelRoundTrip(unittest.TestCase):
    def test_loaded_model_reproduces_saved_metrics(self):
        """Load the saved model from disk, predict again, and compare with the saved metrics."""
        model, scaler, meta = training.load_artifacts()
        df, _ = preprocessing.clean_data(data_collection.load_sample())
        prepared = preprocessing.prepare_data(df)
        table = prediction.predict_test(model, prepared)
        metrics = evaluation.calculate_metrics(table["Actual"], table["Predicted"])
        for name in ("MAE", "MSE", "RMSE"):
            self.assertAlmostEqual(metrics[name], meta["metrics"][name], places=3)
        np.testing.assert_allclose(scaler.data_max_, prepared.scaler.data_max_)


if __name__ == "__main__":
    unittest.main(verbosity=2)
