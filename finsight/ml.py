"""Leakage-aware supervised research on adjusted daily stock prices.

The prediction is a probability of a precisely defined historical outcome, not
personal investment advice. Chronological, purged partitions keep all training
labels before calibration features, and all calibration labels before validation
features. Models are selected on validation only and never refitted after the
final test is opened. A lack of measured evidence produces an explicit abstention.
"""
from __future__ import annotations

import math
from numbers import Integral
import warnings as warning_control

import numpy as np
import pandas as pd


BUY_THRESHOLD = .60
AVOID_THRESHOLD = .40
MINIMUM_INDEPENDENT_TEST = 8
MINIMUM_INDEPENDENT_BUY_CALLS = 5
RANDOM_STATE = 157
MAXIMUM_HISTORY_SESSIONS = 5040

_MODEL_NAMES = {
    "baseline": "Training class-prior baseline",
    "logistic_regression": "Regularized logistic regression",
    "random_forest": "Shallow random forest",
    "hist_gradient_boosting": "Histogram gradient boosting",
}


def _validate_parameters(horizon, cost_bps):
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or not 1 <= horizon <= 252:
        raise ValueError("The prediction horizon must be an integer from 1 to 252 trading sessions.")
    if isinstance(cost_bps, bool):
        raise ValueError("Round-trip costs must be a finite nonnegative number of basis points.")
    try:
        cost = float(cost_bps)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Round-trip costs must be a finite nonnegative number of basis points.") from None
    if not math.isfinite(cost) or not 0 <= cost <= 10000:
        raise ValueError("Round-trip costs must be between 0 and 10,000 basis points.")
    return int(horizon), cost


def _prices(frame):
    """Keep bad bars missing rather than filling them or shortening return windows."""
    if (not isinstance(frame, pd.DataFrame) or frame.empty
            or isinstance(frame.columns, pd.MultiIndex) or not frame.columns.is_unique):
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="Date"))
    column = "Adj Close" if "Adj Close" in frame else "Close"
    if column not in frame:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="Date"))
    if isinstance(frame.index, pd.DatetimeIndex):
        dates = frame.index.tz_localize(None).normalize()
    else:
        dates = []
        for value in frame.index:
            try:
                # Numeric row numbers are not actual historical trading dates.
                dates.append(pd.NaT if isinstance(value, (int, float, np.number))
                             else pd.Timestamp(value).tz_localize(None).normalize())
            except (TypeError, ValueError, OverflowError):
                dates.append(pd.NaT)
        dates = pd.DatetimeIndex(dates)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    result = pd.Series(values, index=dates, name="adjusted_close")
    result = result.loc[~result.index.isna()]
    result = result.loc[~result.index.duplicated(keep="last")].sort_index()
    result.loc[~np.isfinite(result) | (result <= 0)] = np.nan
    result.index.name = "Date"
    return result


def _features(close, benchmark_close=None):
    """Every feature at t uses prices available at or before t only."""
    features = pd.DataFrame(index=close.index)
    for window in (5, 21, 63, 126):
        features[f"return_{window}"] = close.pct_change(window, fill_method=None)
    for window in (20, 50, 200):
        features[f"distance_ma_{window}"] = close / close.rolling(window, min_periods=window).mean() - 1
    log_returns = np.log(close).diff()
    for window in (21, 63):
        features[f"volatility_{window}"] = log_returns.rolling(window, min_periods=window).std() * np.sqrt(252)
    features["drawdown_63"] = close / close.rolling(63, min_periods=63).max() - 1
    change = close.diff()
    gains = change.clip(lower=0).rolling(14, min_periods=14).mean()
    losses = (-change.clip(upper=0)).rolling(14, min_periods=14).mean()
    rsi = 100 - 100 / (1 + gains / losses.replace(0, np.nan))
    rsi = rsi.mask((losses == 0) & (gains > 0), 100.)
    rsi = rsi.mask((losses == 0) & (gains == 0), 50.)
    features["rsi_14"] = rsi / 100
    if benchmark_close is not None:
        for window in (21, 63):
            features[f"relative_return_{window}"] = (
                close.pct_change(window, fill_method=None)
                - benchmark_close.pct_change(window, fill_method=None))
    return features.replace([np.inf, -np.inf], np.nan)


def _date(value):
    return pd.Timestamp(value).date().isoformat()


def build_ml_dataset(history, benchmark=None, *, horizon=63, cost_bps=20):
    """Return trailing features, binary outcomes and aligned, JSON-safe audit data.

    DataFrame ``Close`` must already be split/dividend adjusted (the market-data
    loader requests this explicitly); ``Adj Close`` takes precedence if supplied.
    A signal after session t enters at session t+1 close and exits h sessions later.
    Outcome 1 means stock return exceeds the disclosed round-trip cost AND exceeds
    the gross benchmark return by that cost. No benchmark uses an explicitly
    different absolute positive-net-return target and cannot authorize a decision.
    """
    horizon, cost_bps = _validate_parameters(horizon, cost_bps)
    close = _prices(history)
    benchmark_source = _prices(benchmark)
    has_benchmark = not benchmark_source.empty
    benchmark_close = benchmark_source.reindex(close.index) if has_benchmark else None
    features = _features(close, benchmark_close)
    entry = close.shift(-1)
    exit_price = close.shift(-(horizon + 1))
    forward = (exit_price / entry - 1).replace([np.inf, -np.inf], np.nan)
    benchmark_forward = (benchmark_close.shift(-(horizon + 1)) / benchmark_close.shift(-1) - 1
                         if has_benchmark else pd.Series(np.nan, index=close.index))
    benchmark_forward = benchmark_forward.replace([np.inf, -np.inf], np.nan)
    # Validate every bar in the labelled holding window, including endpoints.
    valid_prices = close.notna().astype(int)
    window_valid = valid_prices.iloc[::-1].rolling(horizon + 1, min_periods=horizon + 1).sum().iloc[::-1].shift(-1)
    valid = features.notna().all(axis=1) & forward.notna() & (window_valid == horizon + 1)
    if has_benchmark:
        benchmark_valid = benchmark_close.notna().astype(int)
        benchmark_window = benchmark_valid.iloc[::-1].rolling(horizon + 1, min_periods=horizon + 1).sum().iloc[::-1].shift(-1)
        valid &= benchmark_forward.notna() & (benchmark_window == horizon + 1)
    cost = cost_bps / 10000
    target = forward > cost
    if has_benchmark:
        target &= forward - benchmark_forward > cost
    positions = np.flatnonzero(valid.to_numpy())
    X = features.loc[valid].copy()
    y = target.loc[valid].astype(int).rename("buy_outcome")
    metadata = {
        "source_rows": int(len(close)), "usable_rows": int(len(X)),
        "feature_names": list(features.columns), "horizon": horizon, "cost_bps": cost_bps,
        "benchmark_available": has_benchmark,
        "target_mode": "positive_net_and_benchmark_outperformance" if has_benchmark else "absolute_positive_net_return",
        "original_positions": [int(value) for value in positions],
        "label_entry_positions": [int(value + 1) for value in positions],
        "label_end_positions": [int(value + horizon + 1) for value in positions],
        "feature_dates": [_date(close.index[value]) for value in positions],
        "label_entry_dates": [_date(close.index[value + 1]) for value in positions],
        "label_end_dates": [_date(close.index[value + horizon + 1]) for value in positions],
        "stock_forward_returns": [float(value) for value in forward.loc[valid]],
        "benchmark_forward_returns": [float(value) for value in benchmark_forward.loc[valid]] if has_benchmark else [],
        "price_basis": "Split/dividend-adjusted daily close; Adj Close preferred, otherwise caller-supplied adjusted Close",
    }
    return X, y, metadata


def _make_partitions(X, metadata, horizon=63):
    """Allocate 40/15/15/30% chronologically, then purge ending labels strictly."""
    count = len(X)
    cuts = [0, int(count * .40), int(count * .55), int(count * .70), count]
    names = ("train", "calibration", "validation", "test")
    partitions = {name: np.arange(cuts[i], cuts[i + 1], dtype=int) for i, name in enumerate(names)}
    label_ends = pd.to_datetime(metadata["label_end_dates"])
    for i, name in enumerate(names[:-1]):
        following = partitions[names[i + 1]]
        if len(following):
            boundary = X.index[following[0]]
            positions = partitions[name]
            partitions[name] = positions[label_ends[positions] < boundary]
    return partitions


def _evaluate(y, probabilities):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                                 log_loss, roc_auc_score)
    y = np.asarray(y, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    predicted = probabilities >= .5
    positive_calls = probabilities >= BUY_THRESHOLD
    both_classes = len(np.unique(y)) == 2
    with warning_control.catch_warnings():
        warning_control.simplefilter("ignore", UserWarning)
        balanced = float(balanced_accuracy_score(y, predicted)) if both_classes else None
    return {
        "observations": int(len(y)), "brier": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, predicted)), "balanced_accuracy": balanced,
        "roc_auc": float(roc_auc_score(y, probabilities)) if both_classes else None,
        "positive_rate": float(y.mean()), "mean_probability": float(probabilities.mean()),
        "positive_calls": int(positive_calls.sum()),
        "positive_precision": float(y[positive_calls].mean()) if positive_calls.any() else None,
    }


def _logits(probabilities):
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values)).reshape(-1, 1)


def _wilson(successes, count):
    if not count:
        return {"lower": None, "upper": None, "confidence": .95}
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return {"lower": float(max(0., center - half_width)), "upper": float(min(1., center + half_width)), "confidence": .95}


def _trade_diagnostics(indices, probabilities, metadata, cost_bps):
    """Fixed disjoint slots; unselected slots stay in zero-return cash."""
    stock = np.asarray([metadata["stock_forward_returns"][index] for index in indices])
    has_benchmark = metadata["benchmark_available"]
    benchmark = (np.asarray([metadata["benchmark_forward_returns"][index] for index in indices])
                 if has_benchmark else np.zeros(len(indices)))
    net = stock - cost_bps / 10000
    excess = net - benchmark
    calls = np.asarray(probabilities) >= BUY_THRESHOLD
    def summarize(mask):
        count = int(mask.sum())
        selected_net = net[mask]
        selected_excess = excess[mask]
        return {
            "n_calls": count, "available_slots": len(indices),
            "win_rate": float((selected_net > 0).mean()) if count else None,
            "mean_net_return": float(selected_net.mean()) if count else None,
            "mean_net_excess_return": float(selected_excess.mean()) if count and has_benchmark else None,
            "target_hit_precision": float(((selected_net > 0) & (selected_excess > 0)).mean()) if count else None,
            "mean_return_per_slot_with_cash": float(np.where(mask, net, 0.).mean()) if len(indices) else None,
            "fraction_slots_invested": float(mask.mean()) if len(indices) else 0.,
        }
    trades = [{
        "feature_date": metadata["feature_dates"][index],
        "entry_date": metadata["label_entry_dates"][index],
        "exit_date": metadata["label_end_dates"][index],
        "probability": float(probabilities[offset]), "buy_call": bool(calls[offset]),
        "gross_stock_return": float(stock[offset]), "net_stock_return": float(net[offset]),
        "gross_benchmark_return": float(benchmark[offset]) if has_benchmark else None,
        "net_excess_return": float(excess[offset]) if has_benchmark else None,
    } for offset, index in enumerate(indices)]
    return {
        "selected_strategy": summarize(calls),
        "always_buy_stock": summarize(np.ones(len(indices), dtype=bool)),
        "trades": trades, "round_trip_cost_bps": cost_bps,
        "cash_return_assumption": 0., "non_overlapping": True,
        "methodology": "Buy calls use fixed probability >= 60% on preselected disjoint holding windows. Enter next close, exit after the selected horizon, charge the assumed stock round-trip cost once. No call stays in zero-return cash. Always-buy compares the same stock and slots after the same cost; benchmark comparison uses gross buy-and-hold returns. No taxes, FX, dividends beyond adjusted closes, market impact, liquidity constraints or cash interest are simulated. These are historical diagnostics, not a portfolio backtest or guaranteed future returns.",
    }


def _independent_indices(test_indices, metadata):
    """Select fixed, disjoint holding windows using dates, never label values."""
    result = []
    previous_exit = -1
    source_positions = metadata["original_positions"]
    for index in test_indices:
        if source_positions[index] >= previous_exit:
            result.append(int(index))
            previous_exit = metadata["label_end_positions"][index]
    return np.asarray(result, dtype=int)


def _partition_audit(indices, X, metadata):
    if not len(indices):
        return {"rows": 0, "feature_start": None, "feature_end": None, "label_end": None,
                "indices": [], "feature_dates": [], "label_entry_dates": [], "label_end_dates": []}
    return {
        "rows": int(len(indices)), "feature_start": _date(X.index[indices[0]]),
        "feature_end": _date(X.index[indices[-1]]),
        "label_start": metadata["label_entry_dates"][indices[0]],
        "label_end": metadata["label_end_dates"][indices[-1]],
        "indices": [int(index) for index in indices],
        "original_positions": [metadata["original_positions"][index] for index in indices],
        "feature_dates": [metadata["feature_dates"][index] for index in indices],
        "label_entry_dates": [metadata["label_entry_dates"][index] for index in indices],
        "label_end_dates": [metadata["label_end_dates"][index] for index in indices],
    }


def _unavailable(reason, horizon, cost_bps, data=None, warnings=None):
    return {
        "status": "unavailable", "eligible_for_decision": False, "signal": "NO_RELIABLE_EDGE",
        "probability": None, "horizon": horizon, "cost_bps": cost_bps,
        "selected_model": None, "target_description": "Unavailable until a usable historical outcome can be defined.",
        "validation": {"models": []}, "test": {"models": [], "selected": {}, "baseline": {},
                                                   "independent_observations": 0},
        "data": data or {}, "feature_importance": [], "reasons": [reason],
        "warnings": warnings or [], "methodology": _methodology(),
    }


def _methodology():
    return (
        "Fixed regularized logistic regression, shallow random forest and histogram gradient boosting candidates; "
        "a training-prior baseline; trailing price features only. Train/calibration/validation/final-test partitions "
        "are chronological (40/15/15/30% before label purges). Each earlier label must end strictly before the "
        "next partition's first feature date. Fit models on training only and sigmoid calibrators on calibration only; "
        "select the candidate by validation Brier score, then log loss. Open the final test only after selection and "
        "never refit for the current prediction. Require out-of-sample improvement and disjoint outcome diagnostics "
        "before using a signal in a purchase assessment. Daily test outcomes overlap; they are not independent "
        "trades. Disjoint windows and a Wilson interval provide conservative diagnostics, not statistical proof. "
        "Single-stock historical research is exposed to regime change, selection bias and provider revisions. "
        "CPU only; no GPU, API key or generative AI is used."
    )


def predict_buy_signal(history, benchmark=None, *, horizon=63, cost_bps=20):
    """Bound native CPU threads on Windows, macOS and Linux during all ML work."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return _unavailable("The threadpoolctl/scikit-learn dependency is unavailable; install the project's requirements.",
                            None, None)
    with threadpool_limits(limits=1):
        return _predict_buy_signal(history, benchmark, horizon=horizon, cost_bps=cost_bps)


def _predict_buy_signal(history, benchmark=None, *, horizon=63, cost_bps=20):
    """Fit and audit a frozen model; abstain unless its held-out evidence passes.

    All returned data are plain JSON types. Returns ``status='available'`` for an
    evaluated model, even when evidence is insufficient for a trading decision.
    Only ``eligible_for_decision`` authorizes the signal to influence a verdict.
    """
    try:
        horizon, cost_bps = _validate_parameters(horizon, cost_bps)
    except ValueError as exc:
        return _unavailable(str(exc), None, None)
    try:
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return _unavailable("The scikit-learn dependency is unavailable; install the project's requirements.", horizon, cost_bps)
    source_prices = _prices(history)
    # Very old regimes are not allowed to push a frozen training fit decades
    # further into the past. The public dataset builder keeps its input intact.
    model_history = source_prices.iloc[-MAXIMUM_HISTORY_SESSIONS:].to_frame("Close")
    X, y, metadata = build_ml_dataset(model_history, benchmark, horizon=horizon, cost_bps=cost_bps)
    partitions = _make_partitions(X, metadata, horizon)
    audit = {name: _partition_audit(indices, X, metadata) for name, indices in partitions.items()}
    data = {"source_rows": metadata["source_rows"], "usable_rows": len(X),
            "features": metadata["feature_names"], "price_basis": metadata["price_basis"],
            "benchmark_available": metadata["benchmark_available"], "partitions": audit,
            "purged_rows": int(len(X) - sum(len(part) for part in partitions.values())),
            "input_source_rows": int(len(source_prices)),
            "maximum_history_sessions": MAXIMUM_HISTORY_SESSIONS,
            "latest_feature_date": None}
    minimums = {"train": 250, "calibration": 50, "validation": 50, "test": 50}
    short = [f"{name} has {len(partitions[name])} rows (needs {minimum})"
             for name, minimum in minimums.items() if len(partitions[name]) < minimum]
    if short:
        return _unavailable("Insufficient history after warm-up and label purges: " + "; ".join(short) + ". Request a longer history.",
                            horizon, cost_bps, data)
    for name in ("train", "calibration"):
        if y.iloc[partitions[name]].nunique() < 2:
            return _unavailable(f"The {name} partition has only one outcome class; reliable supervised fitting/calibration is unavailable.",
                                horizon, cost_bps, data)
    close = _prices(model_history)
    bench_source = _prices(benchmark)
    bench_close = bench_source.reindex(close.index) if metadata["benchmark_available"] else None
    all_features = _features(close, bench_close)
    # The current signal must describe the latest source session, never a stale
    # conveniently complete row from further in the past.
    if all_features.empty or all_features.iloc[-1].isna().any():
        return _unavailable("The latest session lacks complete trailing price/benchmark features; no stale signal is substituted.",
                            horizon, cost_bps, data)
    latest = all_features.iloc[[-1]]
    data["latest_feature_date"] = _date(latest.index[-1])
    train, calibration, validation, test = (partitions[name] for name in ("train", "calibration", "validation", "test"))
    X_train, y_train = X.iloc[train], y.iloc[train]
    X_cal, y_cal = X.iloc[calibration], y.iloc[calibration]
    candidates = {
        "baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=RANDOM_STATE)),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20,
                                                random_state=RANDOM_STATE, n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=80, max_depth=3,
                                                                min_samples_leaf=20, early_stopping=False,
                                                                random_state=RANDOM_STATE),
    }
    models = {}
    validation_rows = []
    failures = []
    for name, model in candidates.items():
        try:
            model.fit(X_train, y_train)
            calibrator = None
            if name != "baseline":
                calibrator = LogisticRegression(C=1., max_iter=1000, random_state=RANDOM_STATE)
                calibrator.fit(_logits(model.predict_proba(X_cal)[:, 1]), y_cal)
            def probability(frame, fitted=model, sigmoid=calibrator):
                values = fitted.predict_proba(frame)[:, 1]
                return sigmoid.predict_proba(_logits(values))[:, 1] if sigmoid is not None else values
            values = probability(X.iloc[validation])
            metrics = _evaluate(y.iloc[validation], values)
            validation_rows.append({"model": name, "name": _MODEL_NAMES[name], **metrics})
            models[name] = {"model": model, "calibrator": calibrator, "probability": probability}
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            failures.append(f"{_MODEL_NAMES[name]} could not be evaluated ({type(exc).__name__}).")
    if "baseline" not in models or len(models) < 2:
        return _unavailable("Not enough candidate models fitted successfully.", horizon, cost_bps, data, failures)
    # Freeze selection before any access to test outcomes. Never select a model,
    # calibrator, horizon, feature or cutoff using final-test measurements.
    winner = min(validation_rows, key=lambda row: (row["brier"], row["log_loss"], row["model"]))
    selected_name = winner["model"]
    selected = models[selected_name]
    current_probability = float(selected["probability"](latest)[0])
    independent = _independent_indices(test, metadata)
    independent_offsets = np.searchsorted(test, independent)
    test_rows = []
    independent_rows = []
    selected_test_probabilities = None
    for name, candidate in models.items():
        values = candidate["probability"](X.iloc[test])
        test_rows.append({"model": name, "name": _MODEL_NAMES[name], **_evaluate(y.iloc[test], values)})
        independent_rows.append({"model": name, "name": _MODEL_NAMES[name],
                                 **_evaluate(y.iloc[independent], values[independent_offsets])})
        if name == selected_name:
            selected_test_probabilities = values
    by_name = {row["model"]: row for row in test_rows}
    independent_by_name = {row["model"]: row for row in independent_rows}
    selected_metrics, baseline_metrics = by_name[selected_name], by_name["baseline"]
    independent_metrics = independent_by_name[selected_name]
    prior = float(y_train.mean())
    brier_improvement = ((baseline_metrics["brier"] - selected_metrics["brier"]) / baseline_metrics["brier"]
                         if baseline_metrics["brier"] > 0 else 0.)
    independent_probabilities = selected_test_probabilities[independent_offsets]
    buy_calls = independent_probabilities >= BUY_THRESHOLD
    buy_count = int(buy_calls.sum())
    successes = int(y.iloc[independent].to_numpy()[buy_calls].sum())
    interval = _wilson(successes, buy_count)
    trading = _trade_diagnostics(independent, independent_probabilities, metadata, cost_bps)
    baseline_validation = next(row for row in validation_rows if row["model"] == "baseline")
    checks = []
    def check(key, passed, value, threshold, failure):
        checks.append({"key": key, "passed": bool(passed), "value": value,
                       "threshold": threshold, "reason": None if passed else failure})
    check("benchmark", metadata["benchmark_available"], metadata["benchmark_available"], True,
          "A usable benchmark is required before this research signal can affect a buy assessment.")
    check("validation_improvement", selected_name != "baseline" and winner["brier"] < baseline_validation["brier"],
          selected_name, "A supervised model beats the training-prior baseline on validation Brier score",
          "No supervised candidate beat the class-prior baseline on validation.")
    check("test_brier_improvement", brier_improvement >= .02, float(brier_improvement), ">= 2% relative improvement",
          "The frozen model did not improve final-test Brier score by at least 2% relative to the training-prior baseline.")
    check("test_auc", selected_metrics["roc_auc"] is not None and selected_metrics["roc_auc"] > .5,
          selected_metrics["roc_auc"], "> 0.50",
          "Final-test discrimination did not exceed chance, or the test contains only one outcome class.")
    check("independent_observations", len(independent) >= MINIMUM_INDEPENDENT_TEST, len(independent), ">= 8 disjoint holding windows",
          "Fewer than 8 disjoint final-test holding windows are available; overlapping daily labels are not independent evidence.")
    independent_targets = y.iloc[independent].to_numpy()
    class_counts = {"positive": int(independent_targets.sum()), "negative": int(len(independent_targets) - independent_targets.sum())}
    check("independent_classes", min(class_counts.values()) >= 2, class_counts, ">= 2 of each outcome in disjoint test windows",
          "The disjoint final-test windows do not contain at least two observations of each outcome.")
    check("independent_buy_calls", buy_count >= MINIMUM_INDEPENDENT_BUY_CALLS, buy_count, ">= 5 disjoint calls at probability >= 60%",
          "Too few disjoint high-probability buy calls are available to assess precision.")
    check("buy_precision", interval["lower"] is not None and interval["lower"] > prior,
          interval["lower"], f"Wilson 95% lower bound > training outcome rate {prior:.1%}",
          "The lower confidence bound on disjoint buy-call precision does not exceed the training-prior success rate.")
    economic_excess = trading["selected_strategy"]["mean_net_excess_return"]
    check("positive_net_excess", economic_excess is not None and economic_excess > 0,
          economic_excess, "> 0 average stock return after costs minus gross benchmark return on disjoint buy calls",
          "The disjoint high-probability buy calls did not earn positive average excess return after assumed costs.")
    eligible = all(item["passed"] for item in checks)
    if not eligible:
        signal = "NO_RELIABLE_EDGE"
    elif current_probability >= BUY_THRESHOLD:
        signal = "BUY_SUPPORT"
    elif current_probability <= AVOID_THRESHOLD:
        signal = "AVOID_SUPPORT"
    else:
        signal = "NEUTRAL"
    reasons = [item["reason"] for item in checks if not item["passed"]]
    if eligible:
        reasons = ["The selected frozen model passes the disclosed final-test and disjoint-window evidence checks.",
                   f"Current estimated target-outcome probability is {current_probability:.1%}; buy support starts at 60%, avoid support at 40% or below."]
    result_warnings = list(failures)
    result_warnings.extend([
        "Daily held-out outcomes overlap. Read the disjoint-window diagnostics before interpreting sample count or accuracy.",
        "A single-stock historical probability does not establish future profitability or account for taxes, liquidity, FX, market impact or personal risk tolerance.",
        "The validated model is frozen at its training/calibration dates; recent labelled observations are not used to refit it.",
    ])
    if not metadata["benchmark_available"]:
        result_warnings.append("Benchmark unavailable: target is positive stock return after assumed costs only, not market outperformance; decision eligibility is disabled.")
    importance = []
    if selected_name == "logistic_regression":
        coefficients = selected["model"].steps[-1][1].coef_[0]
        importance = [{"feature": name, "importance": float(value), "method": "standardized logistic coefficient before sigmoid calibration"}
                      for name, value in zip(X.columns, coefficients)]
    elif selected_name == "random_forest":
        importance = [{"feature": name, "importance": float(value), "method": "training impurity importance; may be biased and is not causal"}
                      for name, value in zip(X.columns, selected["model"].feature_importances_)]
    elif selected_name == "hist_gradient_boosting":
        # HGB exposes no public native importance. Validation permutation avoids
        # opening the final test again, and is explanatory rather than causal.
        from sklearn.inspection import permutation_importance
        class _CalibratedEstimator:
            _estimator_type = "classifier"
            classes_ = np.array([0, 1])
            def fit(self, frame, target):
                return self
            def predict_proba(self, frame):
                values = selected["probability"](frame)
                return np.column_stack([1 - values, values])
        try:
            result = permutation_importance(_CalibratedEstimator(), X.iloc[validation], y.iloc[validation],
                                            scoring=lambda estimator, frame, target: -float(np.mean((target - estimator.predict_proba(frame)[:, 1]) ** 2)),
                                            n_repeats=2, random_state=RANDOM_STATE, n_jobs=1)
            importance = [{"feature": name, "importance": float(value), "method": "validation permutation Brier increase; correlated features may hide importance"}
                          for name, value in zip(X.columns, result.importances_mean)]
        except (ValueError, TypeError, AttributeError):
            result_warnings.append("Feature importance is unavailable for this fitted estimator.")
    importance.sort(key=lambda row: abs(row["importance"]), reverse=True)
    target_description = (
        f"Enter at the next session's adjusted close, exit {horizon} trading sessions later. "
        f"Success means a positive stock return after {cost_bps:g} bps assumed round-trip trading cost"
        + (" and a net stock return greater than the gross benchmark return. The active stock trade bears incremental costs versus the buy-and-hold benchmark."
           if metadata["benchmark_available"] else ". Benchmark missing: this is an absolute-return target only.")
    )
    return {
        "status": "available", "eligible_for_decision": eligible, "signal": signal,
        "probability": current_probability, "horizon": horizon, "cost_bps": cost_bps,
        "selected_model": selected_name, "selected_model_name": _MODEL_NAMES[selected_name],
        "target_description": target_description, "target_mode": metadata["target_mode"],
        "validation": {"selection_metric": "brier", "models": validation_rows, "selected_model": selected_name},
        "test": {"models": test_rows, "selected": selected_metrics, "baseline": baseline_metrics,
                 "brier_relative_improvement": float(brier_improvement), "overlapping_daily_labels": True,
                 "independent_observations": int(len(independent)), "independent_models": independent_rows,
                 "independent_selected": independent_metrics, "independent_baseline": independent_by_name["baseline"],
                 "independent_class_counts": class_counts, "independent_buy_calls": buy_count,
                 "trading": trading,
                 "buy_precision_interval": interval, "training_prior": prior,
                 "evaluation_indices": [int(index) for index in independent],
                 "evaluation_dates": [metadata["feature_dates"][index] for index in independent],
                 "evaluation_end_dates": [metadata["label_end_dates"][index] for index in independent]},
        "eligibility_checks": checks, "data": data, "feature_importance": importance,
        "reasons": reasons, "warnings": result_warnings, "methodology": _methodology(),
    }
