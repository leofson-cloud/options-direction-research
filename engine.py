import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss

REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]

def make_demo_data(n=1800, seed=7):
    """Synthetic OHLCV-like data for checking the app workflow only."""
    rng = np.random.default_rng(seed)
    # Regime shifts and noise are deliberately synthetic; not intended to mimic a tradable series.
    regimes = np.repeat([0.00008, -0.00004, 0.00002, -0.00007, 0.00006], [360, 300, 420, 300, 420])[:n]
    shocks = rng.normal(0, 0.0012, n)
    returns = regimes + shocks
    close = 500 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    intraday_noise = np.abs(rng.normal(0.001, 0.0007, n))
    high = np.maximum(open_, close) * (1 + intraday_noise)
    low = np.minimum(open_, close) * (1 - intraday_noise)
    volume = rng.lognormal(12, 0.35, n).astype(int)
    ts = pd.date_range("2025-01-02 09:30", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})

def _validate(raw):
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = raw[REQUIRED].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if (df["close"] <= 0).any():
        raise ValueError("Close prices must be positive.")
    return df

def prepare_features(raw, horizon=3):
    df = _validate(raw)
    close = df["close"]
    ret1 = close.pct_change()
    f = pd.DataFrame(index=df.index)
    f["ret_1"] = ret1
    f["ret_3"] = close.pct_change(3)
    f["ret_6"] = close.pct_change(6)
    f["sma_gap_5"] = close / close.rolling(5).mean() - 1
    f["sma_gap_15"] = close / close.rolling(15).mean() - 1
    f["ema_gap_8"] = close / close.ewm(span=8, adjust=False).mean() - 1
    f["volatility_10"] = ret1.rolling(10).std()
    f["volatility_30"] = ret1.rolling(30).std()
    f["range_pct"] = (df["high"] - df["low"]) / close
    f["volume_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    f["volume_z20"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    future_ret = close.shift(-horizon) / close - 1
    y = (future_ret > 0).astype(int)
    valid = f.notna().all(axis=1) & future_ret.notna()
    return f.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), df.loc[valid].reset_index(drop=True)

def evaluate_models(X, y, test_fraction=0.25, threshold=0.58):
    n = len(X)
    split = int(n * (1 - test_fraction))
    if split < 50 or n - split < 20:
        raise ValueError("Not enough data for a chronological train/test split.")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random forest": RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=15,
                                                class_weight="balanced_subsample", random_state=7, n_jobs=-1),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                                                             learning_rate=0.06, l2_regularization=1.0, random_state=7),
    }
    pred_cols = {}
    metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "brier_score": brier_score_loss(y_test, prob),
            "n_predictions": len(y_test),
        })
        pred_cols[name] = (prob, pred)

    probs = np.column_stack([v[0] for v in pred_cols.values()])
    ensemble_prob = probs.mean(axis=1)
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    metrics.append({
        "model": "Equal-weight ensemble",
        "accuracy": accuracy_score(y_test, ensemble_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, ensemble_pred),
        "brier_score": brier_score_loss(y_test, ensemble_prob),
        "n_predictions": len(y_test),
    })
    majority = (ensemble_prob >= threshold).astype(int)
    signal = np.where(ensemble_prob >= threshold, 1, np.where(ensemble_prob <= 1-threshold, 0, -1))
    # A naive majority baseline based only on training class prevalence.
    base_class = int(y_train.mean() >= 0.5)
    metrics.append({
        "model": "Train-majority baseline",
        "accuracy": accuracy_score(y_test, np.full(len(y_test), base_class)),
        "balanced_accuracy": balanced_accuracy_score(y_test, np.full(len(y_test), base_class)),
        "brier_score": brier_score_loss(y_test, np.full(len(y_test), y_train.mean())),
        "n_predictions": len(y_test),
    })
    log = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "ensemble_up_probability": ensemble_prob,
        "ensemble_direction_at_50pct": ensemble_pred,
        "ensemble_signal": signal,
        "logistic_up_probability": pred_cols["Logistic regression"][0],
        "random_forest_up_probability": pred_cols["Random forest"][0],
        "gradient_boosting_up_probability": pred_cols["Gradient boosting"][0],
    })
    return {"summary": pd.DataFrame(metrics).sort_values("model").reset_index(drop=True),
            "predictions": log, "split_index": split}
