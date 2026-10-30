# Options Direction Research Lab — free-first starter

This is a local research prototype for comparing baseline statistical / machine-learning methods on historical OHLCV bars. It is not a live trading system and does not yet analyze option contract pricing or live options flow.

## Run it

1. Install Python 3.10 or newer.
2. Extract this folder.
3. Open a terminal in this folder.
4. Create and activate a virtual environment (recommended).
5. Install dependencies:
   `pip install -r requirements.txt`
6. Start the app:
   `streamlit run app.py`

## Data format

Upload a CSV with these columns:

- `timestamp` — date/time of each bar
- `open`
- `high`
- `low`
- `close`
- `volume`

Use chronological, consistently spaced bars. For the intended first experiment, 5-minute SPY bars are a reasonable starting point, but any supported OHLCV dataset can be used.

## What it currently does

- Builds price/volume features without using future values as inputs.
- Predicts whether the underlying close will be higher after a selected number of bars.
- Trains logistic regression, random forest, and gradient boosting models on the earlier segment.
- Evaluates on a later chronological holdout segment.
- Compares an equal-weight ensemble against a simple train-majority baseline.
- Exports a prediction log.

## Important limitations

- Included demo data is synthetic; its results are not evidence of predictive edge.
- A single holdout split is only a first check. Add walk-forward tests before relying on results.
- Bars overlapping across multi-bar forecast horizons create dependent labels.
- The app does not yet account for bid/ask spread, slippage, commissions, option Greeks, volatility surface, exercise/assignment, or option-contract returns.
- Probabilities are not yet calibrated; accuracy alone is insufficient.
- Do not use for live or real-money trading.

## Next engineering milestones

1. Verify data quality and timestamp/time-zone conventions.
2. Add walk-forward validation and confidence intervals.
3. Add calibration plots and benchmark against unconditional / momentum baselines.
4. Add leakage-safe regime analysis and signal ablation.
5. Add options-chain features only when reliable historical data is available.
6. Add paper-trading capture and a full audit trail.
7. Run Track B experiments in a separate module; promote only when it beats the incumbent on untouched periods.
