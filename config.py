"""
config.py
---------
One place for every setting used by the project.
Change a value here and the whole project (app + training scripts) follows.

All paths are built from the location of THIS file, so the project works
on any computer, on Streamlit Community Cloud and on Google Colab
(there are no hard-coded "C:\\Users\\..." style paths).
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Folders and files (all relative to the project folder)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
SAMPLE_CSV = DATA_DIR / "sample_data.csv"

MODEL_DIR = BASE_DIR / "models" / "saved_model"
MODEL_PATH = MODEL_DIR / "lstm_model.keras"      # trained Keras model
SCALER_PATH = MODEL_DIR / "scaler.pkl"           # fitted MinMaxScaler
METADATA_PATH = MODEL_DIR / "metadata.json"      # training history, metrics, data info

OUTPUT_DIR = BASE_DIR / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

TEST_PREDICTIONS_CSV = PREDICTION_DIR / "test_predictions.csv"
FORECAST_CSV = PREDICTION_DIR / "forecast_7_day.csv"
METRICS_JSON = PREDICTION_DIR / "metrics.json"

# ----------------------------------------------------------------------
# Stock information
# ----------------------------------------------------------------------
SYMBOL = "TSLA"
COMPANY_NAME = "Tesla, Inc."
DEFAULT_START_DATE = "2015-01-01"   # default start for the Yahoo Finance download
TSLA_IPO_DATE = "2010-06-29"        # TSLA started trading on this day

# ----------------------------------------------------------------------
# Data columns
# ----------------------------------------------------------------------
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
OPTIONAL_COLUMNS = ["Adj Close"]                 # kept for display if present
FEATURES = ["Open", "High", "Low", "Close", "Volume"]   # inputs to the LSTM
TARGET = "Close"                                 # what we predict

# ----------------------------------------------------------------------
# Pre-processing
# ----------------------------------------------------------------------
WINDOW = 60              # look back 60 trading days
TRAIN_RATIO = 0.80       # first 80% of the timeline = training data
VALIDATION_RATIO = 0.20  # last 20% of the training samples = validation
                         # (10% was too small: validation loss then bottomed out at epoch 2-3)
MIN_RECORDS = 200        # fewest rows we accept for training

# ----------------------------------------------------------------------
# Model and training
# ----------------------------------------------------------------------
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
DENSE_UNITS = 25
BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
MIN_EPOCHS = 5
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10  # stop if validation loss does not improve for 10 epochs
SEED = 42                 # fixed random seed for reproducible results

# ----------------------------------------------------------------------
# Forecasting
# ----------------------------------------------------------------------
FORECAST_DAYS = 7         # we forecast 7 days; the 5-day forecast is the first 5 of them

# ----------------------------------------------------------------------
# Colours (green and blue accents)
# ----------------------------------------------------------------------
GREEN = "#16a34a"
BLUE = "#2563eb"
GREY = "#6b7280"
