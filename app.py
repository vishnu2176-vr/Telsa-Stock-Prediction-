"""
app.py  -  Tesla (TSLA) Stock Price Prediction with LSTM
Run with:   streamlit run app.py

This file only builds the web pages. All the real work (data download, cleaning,
training, prediction, metrics) happens in the src/ folder.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from datetime import date

import pandas as pd
import streamlit as st

import config
from src import data_collection, evaluation, preprocessing, prediction, training
from src.lstm_model import architecture_table, build_model
from utils import charts
from utils.helpers import (AppError, DataError, ModelError, data_status, money, now_text,
                           signed_money, signed_percent)

PAGES = [
    "Overview", "Historical Analysis", "LSTM Training", "Prediction",
    "Actual vs Predicted", "Model Performance", "Project Information",
]

DISCLAIMER = (
    "**Experimental estimate - not financial advice.** These numbers come from a student "
    "machine-learning project. Stock prices depend on news, company results and the wider "
    "economy, none of which this model can see. Do not use them to buy or sell shares."
)

st.set_page_config(page_title="TSLA LSTM Forecast", page_icon="📈", layout="wide")


# ======================================================================
# Look and feel (optional light / dark theme)
# ======================================================================
def apply_theme_css(theme):
    if theme == "dark":
        bg, text, card, border = "#0f172a", "#e5e7eb", "#1e293b", "#334155"
    else:
        bg, text, card, border = "#f8fafc", "#0f172a", "#ffffff", "#e2e8f0"
    # In light mode we keep Streamlit's own coloured alert boxes; in dark mode we restyle them.
    alert_css = f"background: {card}; border: 1px solid {border};" if theme == "dark" else ""
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {bg}; color: {text}; }}
        [data-testid="stSidebar"] {{ background: {card}; }}
        [data-testid="stHeader"] {{ background: {bg}; }}
        .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp li,
        .stApp div[data-testid="stMarkdownContainer"] {{ color: {text}; }}
        /* inputs, date pickers and alert boxes need their own colours in dark mode */
        [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="input"] input,
        [data-testid="stDateInputField"], [data-testid="stDateInputField"] input {{
            background: {bg} !important; color: {text} !important; }}
        [data-testid="stDateInputField"] * {{ color: {text} !important; }}
        [data-testid="stAlertContainer"] {{ {alert_css} }}
        [data-testid="stAlertContainer"] p, [data-testid="stAlertContainer"] div {{ color: {text} !important; }}
        /* keep Streamlit's own green/red colouring inside the metric delta pill */
        [data-testid="stMetricDelta"] * {{ color: inherit !important; }}
        div[data-testid="stMetric"] {{
            background: {card}; border: 1px solid {border};
            border-left: 5px solid {config.GREEN}; border-radius: 8px; padding: 12px 16px;
        }}
        .badge {{ display:inline-block; padding:2px 12px; border-radius:12px; color:white;
                  font-weight:600; font-size:0.85rem; }}
        .badge-Historical {{ background:{config.GREY}; }}
        .badge-Delayed {{ background:{config.BLUE}; }}
        .badge-Current {{ background:{config.GREEN}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def theme_name():
    return "dark" if st.session_state.get("dark_mode", False) else "light"


def show_chart(fig):
    # theme=None -> use OUR light/dark chart template instead of Streamlit's own override
    st.plotly_chart(fig, width="stretch", theme=None)


def disclaimer():
    st.warning(DISCLAIMER)


# ======================================================================
# Loading data
# ======================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cached_download(start, end):
    """Yahoo downloads are cached for an hour so the app does not hit Yahoo on every click."""
    return data_collection.download_tsla(start, end)


def set_data(raw_df, label, kind):
    """Clean a raw DataFrame and make it the data used by every page."""
    clean_df, report = preprocessing.clean_data(raw_df)   # may raise DataError
    st.session_state.update(
        raw_df=raw_df, clean_df=clean_df, report=report,
        source_label=label, source_kind=kind, loaded_at=now_text(),
    )


def first_load():
    """First visit: try Yahoo Finance, fall back to the sample file with a clear notice."""
    try:
        raw = cached_download(config.DEFAULT_START_DATE, date.today())
        set_data(raw, "Yahoo Finance (yfinance)", "yahoo")
        st.session_state.notice = None
    except AppError as error:
        set_data(data_collection.load_sample(), "Bundled sample file (offline)", "sample")
        st.session_state.notice = (
            f"Live data could not be loaded, so the offline sample file is used instead. "
            f"Reason: {error}"
        )


def load_saved_model():
    """Load the saved model + scaler once at start-up (if a model was trained before)."""
    try:
        model, scaler, meta = _cached_model_files()
    except ModelError as error:
        st.session_state.bundle = None
        st.session_state.model_message = str(error)
        return
    test_table = None
    if config.TEST_PREDICTIONS_CSV.exists():
        try:
            test_table = pd.read_csv(config.TEST_PREDICTIONS_CSV, parse_dates=["Date"])
        except Exception:
            test_table = None
    st.session_state.bundle = {"model": model, "scaler": scaler, "metadata": meta, "test_table": test_table}
    st.session_state.model_message = None


@st.cache_resource(show_spinner="Loading saved model...")
def _cached_model_files():
    return training.load_artifacts()


def sidebar_data_controls():
    st.sidebar.subheader("Data source")
    choice = st.sidebar.radio(
        "Where should the data come from?",
        ["Yahoo Finance (live)", "Upload CSV", "Sample data (offline)"],
        label_visibility="collapsed",
    )
    try:
        if choice == "Yahoo Finance (live)":
            today = date.today()
            start = st.sidebar.date_input("Start date", value=date.fromisoformat(config.DEFAULT_START_DATE),
                                          min_value=date.fromisoformat(config.TSLA_IPO_DATE), max_value=today)
            end = st.sidebar.date_input("End date", value=today,
                                        min_value=date.fromisoformat(config.TSLA_IPO_DATE), max_value=today)
            if st.sidebar.button("Download TSLA data", type="primary"):
                with st.spinner("Downloading from Yahoo Finance..."):
                    set_data(cached_download(start, end), "Yahoo Finance (yfinance)", "yahoo")
                st.session_state.notice = None
                st.sidebar.success("Data loaded.")
        elif choice == "Upload CSV":
            st.sidebar.caption("Needs columns: Date, Open, High, Low, Close, Volume.")
            uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])
            if uploaded is not None and st.sidebar.button("Use this file", type="primary"):
                set_data(data_collection.load_csv(uploaded), f"Uploaded CSV ({uploaded.name})", "csv")
                st.session_state.notice = None
                st.sidebar.success("File loaded.")
        else:
            if st.sidebar.button("Load sample data", type="primary"):
                set_data(data_collection.load_sample(), "Bundled sample file (offline)", "sample")
                st.session_state.notice = None
                st.sidebar.success("Sample loaded.")
    except AppError as error:
        st.sidebar.error(str(error))


# ======================================================================
# Small helpers used by several pages
# ======================================================================
def need_data():
    """Return the clean data, or stop the page with a friendly message."""
    df = st.session_state.get("clean_df")
    if df is None or df.empty:
        st.error("No data is loaded yet. Choose a data source in the sidebar.")
        st.stop()
    return df


def need_model():
    """Return the loaded model bundle, or stop the page with a friendly message."""
    bundle = st.session_state.get("bundle")
    if bundle is None:
        st.info(
            (st.session_state.get("model_message") or "No trained model yet.")
            + " Go to **LSTM Training** and click *Train model*."
        )
        st.stop()
    return bundle


# ======================================================================
# Page 1 - Overview
# ======================================================================
def page_overview():
    st.title("📈 Tesla (TSLA) Stock Price Prediction with LSTM")
    df = need_data()

    latest, previous = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
    change = latest - previous
    label, why = data_status(df["Date"].iloc[-1], st.session_state.source_kind)

    st.markdown(f"### {config.COMPANY_NAME} &nbsp; `{config.SYMBOL}` &nbsp; "
                f'<span class="badge badge-{label}">{label}</span>', unsafe_allow_html=True)
    st.caption(why)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest close", money(latest), signed_money(change))
    c2.metric("Previous close", money(previous))
    c3.metric("Daily change", signed_percent(change / previous * 100), signed_money(change))
    c4.metric("Records", f"{len(df):,}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Last data date", str(df["Date"].iloc[-1].date()))
    c6.metric("First data date", str(df["Date"].iloc[0].date()))
    c7.metric("Loaded at", st.session_state.loaded_at)
    st.write(f"**Data source:** {st.session_state.source_label}")

    show_chart(charts.close_history_fig(df, theme_name()))

    with st.expander("Data quality report (what the cleaning step did)"):
        r = st.session_state.report
        st.write(
            f"- Rows read: **{r['rows_in']}**\n"
            f"- Rows with unreadable dates removed: **{r['invalid_date_rows_removed']}**\n"
            f"- Duplicate rows removed: **{r['duplicates_removed']}**\n"
            f"- Missing values found (forward-filled): **{r['missing_values_found']}**\n"
            f"- Rows dropped because of missing/invalid values: "
            f"**{r['rows_dropped_missing'] + r['rows_dropped_invalid']}**\n"
            f"- Rows kept: **{r['rows_out']}**"
        )
    disclaimer()


# ======================================================================
# Page 2 - Historical Analysis
# ======================================================================
def page_historical():
    st.title("Historical Stock Analysis")
    df = need_data()

    first, last = df["Date"].iloc[0].date(), df["Date"].iloc[-1].date()
    picked = st.date_input("Date range", value=(first, last), min_value=first, max_value=last)
    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        st.info("Pick both a start date and an end date.")
        return
    start, end = picked
    view = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)]
    if view.empty:
        st.warning("There are no trading days in that range. Try a wider range.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days shown", f"{len(view):,}")
    c2.metric("Highest close", money(view["Close"].max()))
    c3.metric("Lowest close", money(view["Close"].min()))
    c4.metric("Average close", money(view["Close"].mean()))

    show_chart(charts.close_history_fig(view, theme_name()))
    show_chart(charts.ohlc_fig(view, theme_name()))
    show_chart(charts.volume_fig(view, theme_name()))

    st.subheader("Data table")
    table = view.sort_values("Date", ascending=False).copy()
    table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("Download this table (CSV)", table.to_csv(index=False), "tsla_history.csv", "text/csv")


# ======================================================================
# Page 3 - LSTM Training
# ======================================================================
@st.cache_resource(show_spinner=False)
def _architecture():
    model = build_model()
    return architecture_table(model), int(model.count_params())


def page_training():
    st.title("LSTM Model Training")
    df = need_data()

    st.subheader("Model architecture")
    rows, total = _architecture()
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(f"Total trainable parameters: {total:,}  |  Input: {config.WINDOW} days x "
               f"{len(config.FEATURES)} features ({', '.join(config.FEATURES)})  |  Optimizer: Adam  |  "
               f"Loss: MSE  |  Batch size: {config.BATCH_SIZE}")

    st.subheader("Train")
    epochs = st.slider("Number of epochs (maximum)", config.MIN_EPOCHS, config.MAX_EPOCHS, config.DEFAULT_EPOCHS,
                       help="An epoch is one full pass over the training data. Early stopping may finish sooner.")
    st.caption(f"Training will use the **{len(df):,} records** currently loaded "
               f"({df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}) from: "
               f"{st.session_state.source_label}.")

    if st.button("Train model", type="primary"):
        bar = st.progress(0.0, text="Preparing data...")

        def on_epoch(epoch, total_epochs, logs):
            bar.progress(epoch / total_epochs,
                         text=f"Epoch {epoch}/{total_epochs} - loss {logs.get('loss', 0):.5f} - "
                              f"validation loss {logs.get('val_loss', 0):.5f}")

        try:
            results = training.run_pipeline(st.session_state.raw_df, st.session_state.source_label,
                                            epochs=epochs, on_epoch=on_epoch)
            charts.save_all_graphs(results)
            _cached_model_files.clear()
            st.session_state.bundle = {
                "model": results["model"], "scaler": results["prepared"].scaler,
                "metadata": results["metadata"], "test_table": results["test_table"],
            }
            st.session_state.model_message = None
            bar.progress(1.0, text="Training finished.")
            st.success("Training finished. The model and scaler were saved to models/saved_model/.")
        except AppError as error:
            bar.empty()
            st.error(str(error))
        except Exception as error:            # anything unexpected: still no crash
            bar.empty()
            st.error(f"Something went wrong during training ({type(error).__name__}): {error}")

    st.subheader("Status")
    bundle = st.session_state.get("bundle")
    if bundle is None:
        st.info((st.session_state.get("model_message") or "No trained model yet.") + " Click *Train model* above.")
        return
    meta = bundle["metadata"]
    st.success(f"Model ready - trained {meta['trained_at']} on {meta['rows']:,} records "
               f"({meta['data_source']}).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Epochs run", f"{meta['epochs_run']} / {meta['epochs_requested']}")
    c2.metric("Best epoch", meta["best_epoch"])
    c3.metric("Final training loss", f"{meta['final_train_loss']:.5f}")
    c4.metric("Final validation loss", f"{meta['final_val_loss']:.5f}")
    st.caption(f"Samples - training: {meta['train_samples']}, validation: {meta['validation_samples']}, "
               f"test: {meta['test_samples']}. Early stopping keeps the weights of the best epoch.")
    show_chart(charts.loss_fig(meta["history"], theme_name()))


# ======================================================================
# Page 4 - Prediction
# ======================================================================
def page_prediction():
    st.title("Stock Price Prediction")
    df = need_data()
    bundle = need_model()
    meta = bundle["metadata"]

    memo_key = f"{meta['trained_at']}|{len(df)}|{df['Date'].iloc[-1]}"
    if st.session_state.get("forecast_key") != memo_key:
        try:
            with st.spinner("Forecasting..."):
                st.session_state.forecast = prediction.forecast_future(bundle["model"], bundle["scaler"], df)
            st.session_state.forecast_key = memo_key
        except AppError as error:
            st.error(str(error))
            return
    forecast = st.session_state.forecast
    last_close = float(df["Close"].iloc[-1])

    st.caption(f"Based on the last {config.WINDOW} trading days of the loaded data (up to "
               f"{df['Date'].iloc[-1].date()}). Last actual close: {money(last_close)}. "
               f"Model trained on data up to {meta['last_date']}.")
    disclaimer()

    nxt = forecast.iloc[0]
    st.metric(f"Next trading day ({nxt['Date'].date()}) - predicted close", money(nxt["Predicted_Close"]),
              f"{signed_money(nxt['Change_vs_Last_Close'])} ({signed_percent(nxt['Change_Percent'])}) vs last close")

    def show_days(days):
        view = forecast.head(days).copy()
        view["Date"] = view["Date"].dt.strftime("%Y-%m-%d")
        view = view.rename(columns={"Predicted_Close": "Predicted close ($)",
                                    "Change_vs_Last_Close": "Change vs last close ($)",
                                    "Change_Percent": "Change (%)"})
        st.dataframe(view.round(2), width="stretch", hide_index=True)
        show_chart(charts.forecast_fig(df, forecast, days=days, theme=theme_name()))

    tab5, tab7 = st.tabs(["5-day forecast", "7-day forecast"])
    with tab5:
        show_days(5)
    with tab7:
        show_days(7)

    st.download_button("Download 7-day forecast (CSV)", forecast.to_csv(index=False, date_format="%Y-%m-%d"),
                       "tsla_forecast.csv", "text/csv")
    st.caption("Forecast dates are the next weekdays; stock-market holidays are not removed. "
               "Each step feeds the previous prediction back in, so errors can grow with every extra day.")


# ======================================================================
# Page 5 - Actual vs Predicted
# ======================================================================
def page_actual_vs_predicted():
    st.title("Actual vs Predicted")
    bundle = need_model()
    table = bundle.get("test_table")
    if table is None or table.empty:
        st.info("No test-period predictions are available. Train the model on the *LSTM Training* page.")
        return
    meta = bundle["metadata"]
    st.caption(f"Test period: {meta['test_start_date']} to {meta['last_date']} - the last 20% of the data, "
               "which the model never saw while training.")

    show_chart(charts.actual_vs_pred_fig(table, theme_name()))
    show_chart(charts.error_fig(table, theme_name()))

    worst = table.loc[table["Abs_Error"].idxmax()]
    c1, c2, c3 = st.columns(3)
    c1.metric("Average error (bias)", signed_money(table["Error"].mean()),
              help="Positive = the model tends to predict too high.")
    c2.metric("Largest miss", money(worst["Abs_Error"]), f"on {pd.Timestamp(worst['Date']).date()}",
              delta_color="off")
    c3.metric("Test days", f"{len(table):,}")

    st.subheader("Prediction table")
    view = table.copy()
    view["Date"] = pd.to_datetime(view["Date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(view.round(2), width="stretch", hide_index=True)


# ======================================================================
# Page 6 - Model Performance
# ======================================================================
def page_performance():
    st.title("Model Performance")
    bundle = need_model()
    meta = bundle["metadata"]
    metrics, baseline = meta["metrics"], meta["baseline_metrics"]

    st.caption("All numbers below were calculated from this model's real predictions on the "
               "test period, after converting prices back to dollars.")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", money(metrics["MAE"]))
    c2.metric("MSE", f"{metrics['MSE']:,.2f}")
    c3.metric("RMSE", money(metrics["RMSE"]))

    st.subheader("What the metrics mean")
    for name, text in evaluation.METRIC_EXPLANATIONS.items():
        st.markdown(f"**{name}** - {text}")

    st.subheader("Comparison with a simple baseline")
    st.caption("Baseline = predict that tomorrow's close equals today's close. A useful model should beat it.")
    st.dataframe(pd.DataFrame({
        "Metric": ["MAE ($)", "MSE", "RMSE ($)"],
        "LSTM": [metrics["MAE"], metrics["MSE"], metrics["RMSE"]],
        "Baseline (tomorrow = today)": [baseline["MAE"], baseline["MSE"], baseline["RMSE"]],
    }).round(2), width="stretch", hide_index=True)

    st.subheader("Observations")
    for note in meta["observations"]:
        st.markdown(f"- {note}")

    with st.expander("Training details"):
        st.write(
            f"- Trained: {meta['trained_at']}\n- Data: {meta['data_source']}, {meta['rows']:,} records "
            f"({meta['first_date']} to {meta['last_date']})\n"
            f"- Training samples: {meta['train_samples']}, validation: {meta['validation_samples']}, "
            f"test: {meta['test_samples']}\n"
            f"- Epochs run: {meta['epochs_run']} of {meta['epochs_requested']} (best epoch {meta['best_epoch']})\n"
            f"- Random seed: {meta['seed']}"
        )
    disclaimer()


# ======================================================================
# Page 7 - Project Information
# ======================================================================
def page_info():
    st.title("Project Information")
    tabs = st.tabs(["Introduction", "What is LSTM?", "Dataset", "Architecture & method",
                    "Technologies", "Limitations & future work"])

    with tabs[0]:
        st.markdown(f"""
### Introduction
This project predicts the **closing price of Tesla ({config.SYMBOL}) shares** using a deep-learning model
called an **LSTM**. It downloads real price history, cleans it, trains the model, measures the errors
honestly and forecasts the next few trading days - all inside this web app.

### Objectives
1. Collect real TSLA data from Yahoo Finance (with a CSV upload and a sample file as back-ups).
2. Clean and prepare the data without data leakage.
3. Build, train and save an LSTM model with TensorFlow/Keras.
4. Evaluate it with MAE, MSE and RMSE using real predictions only.
5. Forecast the next day, 5 days and 7 days, and show them clearly as estimates.
6. Present everything in an easy interactive dashboard that can run in the cloud.
""")

    with tabs[1]:
        st.markdown("""
### What is an LSTM?
**LSTM** stands for **Long Short-Term Memory**. It is a special kind of neural network for *sequences* -
data where the order matters, such as words in a sentence or prices over time. A normal network forgets
what it saw a moment ago; an LSTM has a built-in **memory** that it can keep, update or erase.

### How does it work?
At every time step (here, one trading day) the LSTM cell looks at the new day's data and its own memory
and uses three small "gates":

| Gate | Plain-language job |
|---|---|
| **Forget gate** | Decides which old information is no longer useful and should be dropped. |
| **Input gate** | Decides which new information from today is worth storing. |
| **Output gate** | Decides what part of the memory to pass on as today's result. |

By repeating this for all **60 days** in a window, the LSTM builds a summary of the recent trend
and uses it to estimate the **next day's closing price**.
""")

    with tabs[2]:
        st.markdown(f"""
### Dataset description
- **Live source:** Yahoo Finance, downloaded with the `yfinance` Python library (symbol `{config.SYMBOL}`).
- **Back-up source 1:** any CSV you upload with the columns *Date, Open, High, Low, Close, Volume*.
- **Back-up source 2:** `data/sample_data.csv` - real daily TSLA prices from 2015-10-15 to 2018-10-15
  (756 trading days), taken from the public Plotly datasets repository. It ends in 2018, so the app labels
  it **Historical**.

| Column | Meaning |
|---|---|
| Date | Trading day |
| Open / Close | First / last traded price of the day |
| High / Low | Highest / lowest price of the day |
| Adj Close | Close adjusted for splits and dividends (Yahoo only, kept for reference) |
| Volume | Number of shares traded |

The model uses **Open, High, Low, Close and Volume** as inputs and predicts the next day's **Close**.
""")

    with tabs[3]:
        st.markdown("### System architecture")
        st.code(
            "Data source (Yahoo Finance / CSV / sample)\n"
            "        |\n"
            "        v\n"
            "Cleaning: dates, duplicates, missing values, sorting\n"
            "        |\n"
            "        v\n"
            "Chronological split: 80% train | 20% test\n"
            "        |\n"
            "        v\n"
            "MinMaxScaler (fit on TRAIN only) -> 60-day windows\n"
            "        |\n"
            "        v\n"
            "LSTM(50) -> Dropout -> LSTM(50) -> Dropout -> Dense(25) -> Dense(1)\n"
            "        |\n"
            "        v\n"
            "Inverse-scale to dollars -> MAE / MSE / RMSE, forecasts, charts\n"
            "        |\n"
            "        v\n"
            "Streamlit dashboard",
            language=None,
        )
        st.markdown(f"""
### Methodology
1. **Clean** the data (valid dates, no duplicates, forward-fill gaps, sort oldest to newest).
2. **Split by time**: the first 80% of days train the model, the last 20% test it. Nothing is shuffled
   between train and test, so the model never sees the future.
3. **Scale** every feature to 0-1 with a MinMaxScaler fitted on the training days only.
4. **Windows**: each sample is 60 days of 5 features; its answer is the next day's Close.
5. **Train** with Adam, MSE loss, batch size {config.BATCH_SIZE}. The last {int(config.VALIDATION_RATIO * 100)}% of the
   training windows are used for validation, and **EarlyStopping** (patience {config.EARLY_STOP_PATIENCE}) restores
   the best epoch.
6. **Evaluate** on the test windows after converting predictions back to dollars.
7. **Forecast** recursively: predict one day, feed it back in, repeat.
""")

    with tabs[4]:
        st.markdown("""
### Technologies used
- **Python** - main language
- **Pandas, NumPy** - data handling
- **TensorFlow / Keras** - the LSTM model
- **Scikit-learn** - MinMaxScaler and error metrics
- **yfinance** - Yahoo Finance data
- **Plotly, Matplotlib** - interactive and saved charts
- **Streamlit** - the web dashboard (deployable on Streamlit Community Cloud)
""")

    with tabs[5]:
        st.markdown("""
### Limitations
- Prices depend on news, earnings, interest rates and human behaviour. The model only sees past prices and volume.
- A one-day-ahead model often does little better than "tomorrow = today"; the Model Performance page checks this honestly.
- Recursive forecasts feed predictions back as inputs, so errors add up over 5-7 days.
- For future days, Open/High/Low are assumed equal to the predicted Close and volume is held constant.
- Only one model with one fixed set of settings is trained; there is no extensive hyper-parameter search.
- Forecast dates skip weekends but not market holidays.

### Future improvements
- Add technical indicators (moving averages, RSI) and news-sentiment features.
- Compare with GRU, Transformer and classical models such as ARIMA.
- Tune hyper-parameters and use walk-forward validation.
- Predict a range (confidence interval) instead of a single number.
- Support other stock symbols.
""")
    disclaimer()


# ======================================================================
# Main
# ======================================================================
def main():
    # ---- one-time start-up work ----
    if "clean_df" not in st.session_state:
        with st.spinner("Loading TSLA data..."):
            try:
                first_load()
            except AppError as error:
                st.session_state.clean_df = None
                st.session_state.notice = str(error)
    if "bundle" not in st.session_state:
        load_saved_model()

    # ---- sidebar ----
    st.sidebar.title("TSLA LSTM Forecast")
    st.sidebar.toggle("Dark mode", key="dark_mode")
    apply_theme_css(theme_name())
    page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    sidebar_data_controls()
    st.sidebar.divider()
    st.sidebar.caption("Experimental student project - not financial advice.")

    if st.session_state.get("notice"):
        st.warning(st.session_state.notice)

    # ---- page ----
    pages = {
        "Overview": page_overview, "Historical Analysis": page_historical,
        "LSTM Training": page_training, "Prediction": page_prediction,
        "Actual vs Predicted": page_actual_vs_predicted, "Model Performance": page_performance,
        "Project Information": page_info,
    }
    try:
        pages[page]()
    except AppError as error:
        st.error(str(error))
    except Exception as error:      # last safety net so the app never shows a raw traceback
        st.error(f"Unexpected problem ({type(error).__name__}): {error}. Try reloading the data.")


main()
