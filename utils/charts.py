"""
utils/charts.py
---------------
All charts in one file.
  * Functions ending in _fig   -> interactive Plotly figures (used by the Streamlit app)
  * save_all_graphs()          -> writes PNG (matplotlib) and HTML (Plotly) files to outputs/graphs

Every chart is drawn from real data or real model output that is passed in.
"""

import matplotlib
matplotlib.use("Agg")          # draw without a screen (needed on servers)
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import config
from utils.helpers import ensure_folders


def _style(fig, title, theme="light", y_title="Price (USD)", height=420):
    """Apply the same look to every Plotly chart."""
    fig.update_layout(
        title=title,
        template="plotly_dark" if theme == "dark" else "plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        yaxis_title=y_title,
    )
    return fig


# ----------------------------------------------------------------------
# Historical charts
# ----------------------------------------------------------------------
def close_history_fig(df, theme="light"):
    fig = go.Figure(go.Scatter(x=df["Date"], y=df["Close"], name="Close",
                               line=dict(color=config.BLUE, width=2)))
    return _style(fig, "TSLA closing price history", theme)


def ohlc_fig(df, theme="light"):
    fig = go.Figure(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=config.GREEN, decreasing_line_color="#dc2626", name="OHLC"))
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _style(fig, "TSLA Open / High / Low / Close", theme)


def volume_fig(df, theme="light"):
    fig = go.Figure(go.Bar(x=df["Date"], y=df["Volume"], name="Volume",
                           marker_color=config.GREEN))
    return _style(fig, "TSLA trading volume", theme, y_title="Shares traded")


# ----------------------------------------------------------------------
# Training / evaluation charts
# ----------------------------------------------------------------------
def loss_fig(history, theme="light"):
    epochs = list(range(1, len(history["loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["loss"], name="Training loss",
                             line=dict(color=config.BLUE, width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Validation loss",
                             line=dict(color=config.GREEN, width=2)))
    fig.update_xaxes(title="Epoch")
    return _style(fig, "Training and validation loss (MSE on scaled data)", theme, y_title="Loss")


def actual_vs_pred_fig(test_table, theme="light"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_table["Date"], y=test_table["Actual"], name="Actual",
                             line=dict(color=config.BLUE, width=2)))
    fig.add_trace(go.Scatter(x=test_table["Date"], y=test_table["Predicted"], name="Predicted",
                             line=dict(color=config.GREEN, width=2, dash="dot")))
    return _style(fig, "Actual vs predicted closing price (test period)", theme)


def error_fig(test_table, theme="light"):
    colors = [config.GREEN if e >= 0 else "#dc2626" for e in test_table["Error"]]
    fig = go.Figure(go.Bar(x=test_table["Date"], y=test_table["Error"], marker_color=colors,
                           name="Error"))
    fig.add_hline(y=0, line_color=config.GREY)
    return _style(fig, "Prediction error (predicted - actual); green = over, red = under",
                  theme, y_title="Error (USD)")


# ----------------------------------------------------------------------
# Forecast chart
# ----------------------------------------------------------------------
def forecast_fig(df, forecast, days=7, history_days=90, theme="light"):
    """Historical actuals (solid blue) and future estimates (dashed green, shaded area)."""
    recent = df.tail(history_days)
    future = forecast.head(days)

    # Join the lines: the forecast line starts from the last real point.
    line_x = [recent["Date"].iloc[-1]] + list(future["Date"])
    line_y = [recent["Close"].iloc[-1]] + list(future["Predicted_Close"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["Date"], y=recent["Close"], name="Historical (actual)",
                             line=dict(color=config.BLUE, width=2)))
    fig.add_trace(go.Scatter(x=line_x, y=line_y, name="Forecast (estimate)",
                             line=dict(color=config.GREEN, width=2, dash="dash"),
                             marker=dict(size=7), mode="lines+markers"))
    fig.add_vrect(x0=recent["Date"].iloc[-1], x1=future["Date"].iloc[-1],
                  fillcolor=config.GREEN, opacity=0.08, line_width=0,
                  annotation_text="Future estimates", annotation_position="top left")
    return _style(fig, f"{days}-day forecast (experimental)", theme)


# ----------------------------------------------------------------------
# Save everything to outputs/graphs
# ----------------------------------------------------------------------
def save_all_graphs(results):
    """
    Save the six required visualisations as PNG (matplotlib) and interactive HTML (Plotly).
    `results` is the dictionary returned by training.run_pipeline().
    """
    ensure_folders()
    df, table = results["clean_df"], results["test_table"]
    history, forecast = results["history"], results["forecast"]
    out = config.GRAPH_DIR

    # ---- interactive HTML (Plotly) ----
    figures = {
        "01_close_history": close_history_fig(df),
        "02_volume_history": volume_fig(df),
        "03_training_validation_loss": loss_fig(history),
        "04_actual_vs_predicted": actual_vs_pred_fig(table),
        "05_future_forecast": forecast_fig(df, forecast, days=7),
        "06_prediction_error": error_fig(table),
    }
    for name, fig in figures.items():
        fig.write_html(out / f"{name}.html", include_plotlyjs="cdn")

    # ---- static PNG (matplotlib) ----
    def finish(fig, ax, title, ylabel, name):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], color=config.BLUE)
    finish(fig, ax, "TSLA closing price history", "Price (USD)", "01_close_history")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["Date"], df["Volume"], color=config.GREEN, width=1.0)
    finish(fig, ax, "TSLA trading volume", "Shares traded", "02_volume_history")

    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = range(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"], color=config.BLUE, label="Training loss")
    ax.plot(epochs, history["val_loss"], color=config.GREEN, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    finish(fig, ax, "Training and validation loss", "Loss (MSE, scaled)", "03_training_validation_loss")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(table["Date"], table["Actual"], color=config.BLUE, label="Actual")
    ax.plot(table["Date"], table["Predicted"], color=config.GREEN, linestyle="--", label="Predicted")
    ax.legend()
    finish(fig, ax, "Actual vs predicted closing price (test period)", "Price (USD)", "04_actual_vs_predicted")

    fig, ax = plt.subplots(figsize=(10, 4))
    recent = df.tail(90)
    ax.plot(recent["Date"], recent["Close"], color=config.BLUE, label="Historical (actual)")
    ax.plot([recent["Date"].iloc[-1], *forecast["Date"]],
            [recent["Close"].iloc[-1], *forecast["Predicted_Close"]],
            color=config.GREEN, linestyle="--", marker="o", label="Forecast (estimate)")
    ax.legend()
    finish(fig, ax, "7-day forecast (experimental)", "Price (USD)", "05_future_forecast")

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [config.GREEN if e >= 0 else "#dc2626" for e in table["Error"]]
    ax.bar(table["Date"], table["Error"], color=colors, width=1.0)
    ax.axhline(0, color=config.GREY)
    finish(fig, ax, "Prediction error (predicted - actual)", "Error (USD)", "06_prediction_error")
