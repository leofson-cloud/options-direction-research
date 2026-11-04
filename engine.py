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

def evaluate_walk_forward(X, y, n_splits=5, initial_train_fraction=0.50, horizon=3):
    """
    Expanding-window chronological validation.

    Each fold trains only on earlier observations.
    The final `horizon` training rows are purged to prevent
    training labels from looking into the test period.
    """
    n = len(X)

    if n_splits < 2:
        raise ValueError("Use at least 2 walk-forward splits.")

    initial_train = int(n * initial_train_fraction)

    if initial_train - horizon < 50:
        raise ValueError("Not enough initial training data.")

    remaining = n - initial_train

    if remaining < n_splits * 20:
        raise ValueError("Not enough observations for the requested test windows.")

    models_template = {
        "Logistic regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "Random forest": lambda: RandomForestClassifier(
            n_estimators=250,
            max_depth=5,
            min_samples_leaf=15,
            class_weight="balanced_subsample",
            random_state=7,
            n_jobs=-1
        ),
        "Gradient boosting": lambda: HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=15,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=7
        ),
    }

    fold_results = []
    prediction_results = []

    for fold in range(n_splits):
        test_start = initial_train + (remaining * fold // n_splits)
        test_end = initial_train + (remaining * (fold + 1) // n_splits)

        train_end = test_start - horizon

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        if len(X_test) == 0 or y_train.nunique() < 2:
            raise ValueError(
                f"Insufficient data or only one training class in fold {fold + 1}."
            )

        probabilities = {}

        for name, factory in models_template.items():
            model = factory()
            model.fit(X_train, y_train)

            probabilities[name] = model.predict_proba(X_test)[:, 1]

        ensemble_prob = np.mean(
            np.column_stack(list(probabilities.values())),
            axis=1
        )
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)

        fold_models = dict(probabilities)
        fold_models["Equal-weight ensemble"] = ensemble_prob

        base_probability = float(y_train.mean())
        baseline_pred = int(base_probability >= 0.5)

        fold_models["Train-majority baseline"] = np.full(
            len(y_test), base_probability
        )

        for name, prob in fold_models.items():
            pred = (prob >= 0.5).astype(int)

            fold_results.append({
                "fold": fold + 1,
                "model": name,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "accuracy": accuracy_score(y_test, pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                "brier_score": brier_score_loss(y_test, prob),
            })

        prediction_results.append(pd.DataFrame({
            "fold": fold + 1,
            "actual": y_test.to_numpy(),
            "ensemble_up_probability": ensemble_prob,
            "ensemble_direction": ensemble_pred,
        }, index=y_test.index))

    folds = pd.DataFrame(fold_results)

    summary = (
        folds.groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            brier_score_mean=("brier_score", "mean"),
            brier_score_std=("brier_score", "std"),
            total_test_rows=("test_rows", "sum"),
        )
        .reset_index()
        .sort_values("model")
    )

    predictions = pd.concat(prediction_results).sort_index()

    return {
        "summary": summary,
        "folds": folds,
        "predictions": predictions.reset_index(drop=True),
    }
def analyze_market_regimes(
    raw,
    adx_period=14,
    volatility_period=20,
    trend_short=20,
    trend_long=50
):
    """
    Classify historical market conditions using information available
    at each point in time. No future prices are used.
    """
    df = _validate(raw)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Trend direction: short and long moving averages
    sma_short = close.rolling(
        trend_short,
        min_periods=trend_short
    ).mean()

    sma_long = close.rolling(
        trend_long,
        min_periods=trend_long
    ).mean()

    # Average True Range (ATR)
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / adx_period,
        adjust=False,
        min_periods=adx_period
    ).mean()

    # Directional movement for ADX
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0),
        0.0
    )

    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0),
        0.0
    )

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / adx_period,
            adjust=False,
            min_periods=adx_period
        ).mean()
        / atr
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / adx_period,
            adjust=False,
            min_periods=adx_period
        ).mean()
        / atr
    )

    directional_sum = (plus_di + minus_di).replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / directional_sum
    )

    adx = dx.ewm(
        alpha=1 / adx_period,
        adjust=False,
        min_periods=adx_period
    ).mean()

    # Rolling volatility, measured as a percentage
    returns = close.pct_change()

    volatility = (
        returns.rolling(
            volatility_period,
            min_periods=volatility_period
        ).std()
        * 100
    )

    # Compare volatility with the previous 252 bars.
    # Shifting by one bar prevents the current value from
    # influencing its own comparison threshold.
    volatility_threshold = (
        volatility.rolling(
            252,
            min_periods=60
        ).quantile(0.75).shift(1)
    )

    # Only classify rows with sufficient historical information
    valid = (
        sma_long.notna()
        & adx.notna()
        & volatility.notna()
        & volatility_threshold.notna()
    )

    regime = pd.Series(
        "Insufficient history",
        index=df.index,
        dtype="object"
    )

    # Trend conditions
    uptrend = (
        (close > sma_short)
        & (sma_short > sma_long)
        & (adx >= 20)
    )

    downtrend = (
        (close < sma_short)
        & (sma_short < sma_long)
        & (adx >= 20)
    )

    # Start with sideways conditions for valid rows
    regime.loc[valid] = "Sideways"

    regime.loc[valid & uptrend] = "Uptrend"
    regime.loc[valid & downtrend] = "Downtrend"

    # High volatility takes priority over trend classification
    high_volatility = (
        valid
        & (volatility > volatility_threshold)
    )

    regime.loc[high_volatility] = "High volatility"

    # Return the measurements and classification
    result = df[["timestamp", "close"]].copy()

    result["sma_20"] = sma_short
    result["sma_50"] = sma_long
    result["adx_14"] = adx
    result["volatility_20_pct"] = volatility
    result["volatility_threshold_pct"] = volatility_threshold
    result["market_regime"] = regime

    return result
