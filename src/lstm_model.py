"""
src/lstm_model.py
-----------------
Defines the LSTM neural network (Keras / TensorFlow).

Architecture (as required by the project brief):
    Input  (60 days x 5 features)
    LSTM(50, return_sequences=True)
    Dropout(0.2)
    LSTM(50)
    Dropout(0.2)
    Dense(25, ReLU)
    Dense(1)            -> the predicted (scaled) closing price
"""

import os

# Hide TensorFlow's very chatty start-up messages (must be set before importing it).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
from keras import layers

import config


def build_model(window=config.WINDOW, n_features=len(config.FEATURES)):
    """Create and compile a fresh, untrained LSTM model."""
    model = keras.Sequential(
        [
            layers.Input(shape=(window, n_features)),
            layers.LSTM(config.LSTM_UNITS, return_sequences=True),
            layers.Dropout(config.DROPOUT_RATE),
            layers.LSTM(config.LSTM_UNITS),
            layers.Dropout(config.DROPOUT_RATE),
            layers.Dense(config.DENSE_UNITS, activation="relu"),
            layers.Dense(1),
        ],
        name="tsla_lstm",
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def architecture_table(model):
    """
    Describe each layer as a list of dictionaries (used to show a table in the app).
    Keys: Layer, Type, Output shape, Parameters.
    """
    rows = []
    for layer in model.layers:
        shape = getattr(layer, "output", None)
        shape = tuple(shape.shape) if shape is not None else ("?",)
        rows.append(
            {
                "Layer": layer.name,
                "Type": layer.__class__.__name__,
                "Output shape": str(shape).replace("None", "batch"),
                "Parameters": int(layer.count_params()),
            }
        )
    return rows
