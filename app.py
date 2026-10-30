from pathlib import Path
import pandas as pd
import streamlit as st
from engine import evaluate_models, prepare_features, make_demo_data

st.set_page_config(page_title="Options Direction Research Lab", layout="wide")
st.title("Options Direction Research Lab")
st.caption("Free-first research prototype • Educational research only • Not live trading advice")

st.info(
    "This prototype predicts the direction of the underlying price over a chosen horizon. "
    "It does not yet model option-contract P&L, live options flow, or real-time quotes. "
    "The included demo data is synthetic and must not be treated as market evidence."
)

with st.sidebar:
    st.header("Research settings")
    horizon = st.slider("Forecast horizon (bars)", min_value=1, max_value=12, value=3)
    test_fraction = st.slider("Final test fraction", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
    threshold = st.slider("Minimum probability for directional label", min_value=0.50, max_value=0.80, value=0.58, step=0.01)
       data_choice = st.selectbox(
    "Choose data source",
    ["SPY historical data", "Synthetic demo data", "Upload my own CSV"]
)

uploaded = None
if data_choice == "Upload my own CSV":
    uploaded = st.file_uploader("Upload OHLCV CSV", type=["csv"])

st.caption("Required columns: timestamp, open, high, low, close, volume")

if data_choice == "SPY historical data":
    spy_path = Path(__file__).parent / "data" / "spy_5y_daily.csv"
    if spy_path.exists():
        raw = pd.read_csv(spy_path)
        source_label = "SPY historical data"
    else:
        st.error("SPY data file not found in the data folder.")
        st.stop()

elif data_choice == "Synthetic demo data":
    raw = make_demo_data()
    source_label = "Synthetic demo data"

else:
    if uploaded is None:
        st.info("Please upload an OHLCV CSV file to continue.")
        st.stop()
    raw = pd.read_csv(uploaded)
    source_label = "Uploaded CSV"

st.subheader("1. Input data")
st.write(f"Data source: **{source_label}** · {len(raw):,} rows")
st.dataframe(raw.tail(8), use_container_width=True)

try:
    features, y, aligned = prepare_features(raw, horizon=horizon)
    if len(features) < 100:
        st.warning("Use at least 100 usable rows; more history is strongly recommended.")
    if len(features) < 40:
        st.stop()

    results = evaluate_models(features, y, test_fraction=test_fraction, threshold=threshold)
    st.subheader("2. Out-of-sample results")
    st.caption(
        "The final chronological segment is held out from model fitting. Results are still only a preliminary "
        "test; use walk-forward validation and realistic costs before drawing conclusions."
    )
    summary = results["summary"]
    cols = st.columns(min(4, len(summary)))
    for i, (_, row) in enumerate(summary.iterrows()):
        with cols[i % len(cols)]:
            st.metric(row["model"], f'{row["accuracy"]:.1%}', f'{row["n_predictions"]} predictions')
    st.dataframe(summary, use_container_width=True)

    st.subheader("3. Prediction log")
    log = results["predictions"].copy()
    log["actual"] = log["actual"].map({1: "UP", 0: "DOWN"})
    for c in ["baseline_signal", "ensemble_signal"]:
        if c in log:
            log[c] = log[c].map({1: "UP", 0: "DOWN", -1: "NO TRADE"})
    st.dataframe(log.tail(50), use_container_width=True)

    st.subheader("4. Current feature snapshot")
    st.dataframe(features.tail(1).T.rename(columns={features.tail(1).index[0]: "latest value"}), use_container_width=True)
    st.download_button("Download prediction log CSV", log.to_csv(index=False), "prediction_log.csv", "text/csv")
except Exception as e:
    st.error(f"Could not process this file: {e}")
    st.markdown(
        "Check that your CSV has columns named `timestamp`, `open`, `high`, `low`, `close`, and `volume`, "
        "with numeric OHLCV values and timestamps in chronological order."
    )

st.subheader("Research rules")
st.markdown(
    "- Never train on future information.\n"
    "- Keep the final test period untouched until model choices are made.\n"
    "- Compare against a simple baseline, not just against other complex models.\n"
    "- A correct underlying-direction forecast does not guarantee a profitable options trade.\n"
    "- Do not use synthetic demo results to make trading decisions."
)
