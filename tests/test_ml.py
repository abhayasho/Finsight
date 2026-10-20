"""Historical ML must earn its vote without looking into future prices or labels."""

import json

import numpy as np
import pandas as pd
import pytest

from finsight import ml


def price_history(size=2200, seed=15):
    """Reproducible changing regimes, not a crafted guaranteed predictive edge."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2014-01-02", periods=size, name="Date")
    market_returns = rng.normal(.00015, .008, size)
    stock_returns = (market_returns + rng.normal(.00005, .014, size)
                     + .0015 * np.sin(np.arange(size) / 31))
    market = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(market_returns))}, index=index)
    stock = pd.DataFrame({"Close": 80 * np.exp(np.cumsum(stock_returns))}, index=index)
    return stock, market


def test_target_means_profitable_and_beats_benchmark_after_cost_from_next_close():
    stock, market = price_history(620)
    horizon, cost_bps = 21, 37
    X, y, metadata = ml.build_ml_dataset(stock, market, horizon=horizon, cost_bps=cost_bps)
    positions = np.asarray(metadata["original_positions"], dtype=int)
    entries = np.asarray(metadata["label_entry_positions"], dtype=int)
    exits = np.asarray(metadata["label_end_positions"], dtype=int)
    assert len(X) == len(y) == len(positions) > 100
    np.testing.assert_array_equal(entries, positions + 1)
    np.testing.assert_array_equal(exits, positions + 1 + horizon)
    stock_return = stock["Close"].to_numpy()[exits] / stock["Close"].to_numpy()[entries] - 1
    market_return = market["Close"].to_numpy()[exits] / market["Close"].to_numpy()[entries] - 1
    np.testing.assert_allclose(metadata["stock_forward_returns"], stock_return)
    np.testing.assert_allclose(metadata["benchmark_forward_returns"], market_return)
    expected = (stock_return > cost_bps / 10_000) & (stock_return - market_return > cost_bps / 10_000)
    np.testing.assert_array_equal(y.to_numpy(), expected.astype(int))
    assert y.nunique() == 2
    assert X.index.equals(y.index)
    assert np.isfinite(X.to_numpy()).all()


def test_features_are_prefix_invariant_when_future_prices_change():
    stock, market = price_history(700)
    prefix_length = 510
    full_X, _, _ = ml.build_ml_dataset(stock, market, horizon=21)
    prefix_X, _, _ = ml.build_ml_dataset(stock.iloc[:prefix_length], market.iloc[:prefix_length], horizon=21)
    pd.testing.assert_frame_equal(full_X.loc[prefix_X.index], prefix_X)
    altered_stock, altered_market = stock.copy(), market.copy()
    altered_stock.loc[stock.index[prefix_length]:, "Close"] *= np.linspace(1., 9., len(stock) - prefix_length)
    altered_market.loc[market.index[prefix_length]:, "Close"] *= np.linspace(1., .3, len(market) - prefix_length)
    altered_X, _, _ = ml.build_ml_dataset(altered_stock, altered_market, horizon=21)
    common_past = full_X.index[full_X.index < stock.index[prefix_length]]
    pd.testing.assert_frame_equal(full_X.loc[common_past], altered_X.loc[common_past])


def test_present_day_fundamentals_and_news_are_not_backfilled_into_training():
    stock, market = price_history(500)
    X, y, _ = ml.build_ml_dataset(stock, market, horizon=21)
    contaminated = stock.assign(trailingPE=-1000., revenueGrowth=999., sentiment=1., target=1)
    other_X, other_y, _ = ml.build_ml_dataset(contaminated, market, horizon=21)
    pd.testing.assert_frame_equal(X, other_X)
    pd.testing.assert_series_equal(y, other_y)


def test_input_sorting_duplicates_and_invalid_prices_do_not_change_valid_features():
    stock, market = price_history(500)
    clean_X, clean_y, _ = ml.build_ml_dataset(stock, market, horizon=21)
    extra_dates = pd.to_datetime(["2013-12-20", "2013-12-23", "2013-12-24"])
    invalid = pd.DataFrame({"Close": [np.nan, -1., np.inf]}, index=extra_dates)
    dirty = pd.concat([stock, stock.iloc[[20, 80]], invalid]).sample(frac=1, random_state=4)
    dirty_X, dirty_y, _ = ml.build_ml_dataset(dirty, market, horizon=21)
    pd.testing.assert_frame_equal(clean_X, dirty_X, check_freq=False)
    pd.testing.assert_series_equal(clean_y, dirty_y, check_freq=False)


def test_target_with_benchmark_gaps_uses_original_stock_sessions():
    stock, market = price_history(650)
    market = market.drop(market.index[251::97])
    X, y, metadata = ml.build_ml_dataset(stock, market, horizon=21)
    for feature_date, entry_date, exit_date, position in zip(
        X.index, metadata["label_entry_dates"], metadata["label_end_dates"], metadata["original_positions"]
    ):
        position = int(position)
        assert pd.Timestamp(feature_date) == stock.index[position]
        assert pd.Timestamp(entry_date) == stock.index[position + 1]
        assert pd.Timestamp(exit_date) == stock.index[position + 22]
        assert pd.Timestamp(entry_date) in market.index
        assert pd.Timestamp(exit_date) in market.index
    assert len(X) == len(y) > 100


def test_all_partition_boundaries_purge_forward_labels_using_real_dates():
    stock, market = price_history(2400)
    # Missing benchmark dates make X row offsets different from source offsets.
    market = market.drop(market.index[220::211])
    X, _, metadata = ml.build_ml_dataset(stock, market, horizon=63)
    partitions = ml._make_partitions(X, metadata, 63)
    ends = pd.to_datetime(metadata["label_end_dates"])
    names = ("train", "calibration", "validation", "test")
    for name in names:
        positions = np.asarray(partitions[name], dtype=int)
        assert len(positions) > 0
        assert np.all(np.diff(positions) > 0)
    for first, second in zip(names, names[1:]):
        previous = np.asarray(partitions[first], dtype=int)
        following = np.asarray(partitions[second], dtype=int)
        assert ends[previous].max() < X.index[following].min()
    all_positions = np.concatenate([partitions[name] for name in names])
    assert len(set(all_positions)) == len(all_positions)


@pytest.fixture(scope="module")
def fitted_result():
    stock, market = price_history()
    return ml.predict_buy_signal(stock, market, horizon=21)


def test_real_cpu_models_report_unseen_validation_and_test_evidence(fitted_result):
    result = fitted_result
    assert result["selected_model"]
    assert result["validation"]["models"]
    assert {row["model"] for row in result["validation"]["models"]} == {
        "baseline", "logistic_regression", "random_forest", "hist_gradient_boosting"
    }
    assert result["test"]["models"]
    assert result["test"]["selected"]
    assert result["test"]["baseline"]
    assert result["test"]["independent_observations"] > 0
    assert 0 <= result["probability"] <= 1
    assert result["horizon"] == 21 and result["cost_bps"] == 20
    assert result["target_description"] and result["methodology"]
    assert isinstance(result["warnings"], list)
    assert isinstance(result["reasons"], list)
    json.dumps(result, allow_nan=False)


def test_final_holdout_evaluation_uses_nonoverlapping_trades(fitted_result):
    result = fitted_result
    dates = pd.to_datetime(result["test"]["evaluation_dates"])
    ends = pd.to_datetime(result["test"]["evaluation_end_dates"])
    assert len(dates) == len(ends) == result["test"]["independent_observations"]
    # A new signal may follow that day's previous exit; its entry is next close.
    assert all(end <= following for end, following in zip(ends[:-1], dates[1:]))


@pytest.mark.parametrize("stock", [None, pd.DataFrame(), pd.DataFrame({"Close": [100.] * 100}),
                                     pd.DataFrame({"Close": [np.nan, -1., np.inf]})])
def test_short_invalid_history_abstains_without_nonfinite_json(stock):
    result = ml.predict_buy_signal(stock, None)
    assert result["eligible_for_decision"] is False
    assert result["signal"] not in {"BUY", "DO_NOT_BUY"}
    assert result["reasons"]
    json.dumps(result, allow_nan=False)


def test_constant_price_single_class_history_abstains():
    index = pd.bdate_range("2012-01-02", periods=2200)
    stock = pd.DataFrame({"Close": 100.}, index=index)
    market = pd.DataFrame({"Close": 100.}, index=index)
    result = ml.predict_buy_signal(stock, market, horizon=21)
    assert result["eligible_for_decision"] is False
    assert result["signal"] not in {"BUY", "DO_NOT_BUY"}
    json.dumps(result, allow_nan=False)


def test_missing_benchmark_does_not_claim_benchmark_outperformance():
    stock, _ = price_history(500)
    result = ml.predict_buy_signal(stock, None, horizon=21)
    assert result["eligible_for_decision"] is False
    assert result["reasons"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("changes", [{"horizon": 0}, {"horizon": True}, {"horizon": "63"},
                                     {"cost_bps": -1}, {"cost_bps": np.nan}, {"cost_bps": np.inf}])
def test_invalid_model_settings_abstain_as_json_instead_of_fitting(changes):
    stock, market = price_history(400)
    result = ml.predict_buy_signal(stock, market, **changes)
    assert result["eligible_for_decision"] is False
    assert result["reasons"]
    json.dumps(result, allow_nan=False)


def test_no_learned_edge_cannot_become_a_buy_just_from_a_large_probability(fitted_result):
    result = fitted_result
    checks_passed = all(check["passed"] for check in result["eligibility_checks"])
    assert result["eligible_for_decision"] is checks_passed
    if not checks_passed:
        assert result["signal"] == "NO_RELIABLE_EDGE"
    if result["selected_model"] == "baseline":
        assert result["eligible_for_decision"] is False


def test_probability_metrics_match_hand_computed_predictions():
    metrics = ml._evaluate(np.array([0, 1]), np.array([.2, .8]))
    assert metrics["brier"] == pytest.approx(.04)
    assert metrics["log_loss"] == pytest.approx(-np.log(.8))
    assert metrics["accuracy"] == 1.
    assert metrics["balanced_accuracy"] == 1.
    assert metrics["roc_auc"] == 1.


def test_disjoint_trade_costs_cash_and_benchmark_returns_match_hand_calculation():
    metadata = {
        "stock_forward_returns": [.10, .20, -.05],
        "benchmark_forward_returns": [.04, .08, -.02], "benchmark_available": True,
        "feature_dates": ["2024-01-01", "2024-02-01", "2024-03-01"],
        "label_entry_dates": ["2024-01-02", "2024-02-02", "2024-03-04"],
        "label_end_dates": ["2024-01-30", "2024-02-29", "2024-04-02"],
    }
    result = ml._trade_diagnostics(np.array([0, 1, 2]), np.array([.8, .3, .7]), metadata, 100)
    selected, always = result["selected_strategy"], result["always_buy_stock"]
    assert selected["n_calls"] == 2
    assert selected["win_rate"] == .5
    assert selected["target_hit_precision"] == .5
    assert selected["mean_net_return"] == pytest.approx(.015)
    assert selected["mean_net_excess_return"] == pytest.approx(.005)
    assert selected["mean_return_per_slot_with_cash"] == pytest.approx(.01)
    assert selected["fraction_slots_invested"] == pytest.approx(2 / 3)
    assert always["n_calls"] == 3
    assert always["mean_net_return"] == pytest.approx((.09 + .19 - .06) / 3)
    assert result["trades"][0]["net_stock_return"] == pytest.approx(.09)
    assert result["trades"][0]["net_excess_return"] == pytest.approx(.05)
    json.dumps(result, allow_nan=False)


def test_no_disjoint_buy_calls_report_cash_and_missing_precision():
    metadata = {
        "stock_forward_returns": [.10], "benchmark_forward_returns": [.04],
        "benchmark_available": True, "feature_dates": ["2024-01-01"],
        "label_entry_dates": ["2024-01-02"], "label_end_dates": ["2024-01-30"],
    }
    result = ml._trade_diagnostics(np.array([0]), np.array([.2]), metadata, 20)
    selected = result["selected_strategy"]
    assert selected["n_calls"] == 0
    assert selected["target_hit_precision"] is None
    assert selected["mean_net_return"] is None
    assert selected["mean_return_per_slot_with_cash"] == 0.
    assert selected["fraction_slots_invested"] == 0.
    json.dumps(result, allow_nan=False)


def test_prediction_caps_old_history_before_any_training(monkeypatch):
    stock, market = price_history(6000)
    original_build = ml.build_ml_dataset
    captured = {}

    def capture_model_window(history, benchmark, **kwargs):
        captured["index"] = history.index.copy()
        X, y, metadata = original_build(history, benchmark, **kwargs)
        # Stop before fitting; this test checks the historical input boundary.
        return X, y * 0, metadata

    monkeypatch.setattr(ml, "build_ml_dataset", capture_model_window)
    result = ml.predict_buy_signal(stock, market, horizon=21)
    assert captured["index"].equals(stock.index[-5040:])
    assert result["data"]["source_rows"] == 5040
    assert result["data"]["input_source_rows"] == 6000
    assert result["data"]["maximum_history_sessions"] == 5040


def test_high_probability_is_vetoed_when_final_test_has_no_predictive_edge(monkeypatch):
    from sklearn.linear_model import LogisticRegression

    stock, market = price_history()
    original_build = ml.build_ml_dataset

    def validation_favors_high_predictions(*args, **kwargs):
        X, y, metadata = original_build(*args, **kwargs)
        partitions = ml._make_partitions(X, metadata, kwargs.get("horizon", 63))
        y = y.copy()
        y.iloc[partitions["validation"]] = 1
        return X, y, metadata

    def overconfident_probability(self, frame):
        return np.tile([.01, .99], (len(frame), 1))

    monkeypatch.setattr(ml, "build_ml_dataset", validation_favors_high_predictions)
    monkeypatch.setattr(LogisticRegression, "predict_proba", overconfident_probability)
    result = ml.predict_buy_signal(stock, market, horizon=21)
    assert result["selected_model"] != "baseline"
    assert result["probability"] == pytest.approx(.99)
    assert result["eligible_for_decision"] is False
    assert result["signal"] == "NO_RELIABLE_EDGE"
    assert any(not check["passed"] for check in result["eligibility_checks"]
               if check["key"] in {"test_auc", "test_brier_improvement"})


def test_changing_only_final_test_labels_cannot_select_or_refit_the_model(monkeypatch, fitted_result):
    stock, market = price_history()
    original_build = ml.build_ml_dataset

    def changed_test_labels(*args, **kwargs):
        X, y, metadata = original_build(*args, **kwargs)
        partitions = ml._make_partitions(X, metadata, kwargs.get("horizon", 63))
        y = y.copy()
        y.iloc[partitions["test"]] = 1 - y.iloc[partitions["test"]]
        return X, y, metadata

    monkeypatch.setattr(ml, "build_ml_dataset", changed_test_labels)
    changed = ml.predict_buy_signal(stock, market, horizon=21)
    assert changed["selected_model"] == fitted_result["selected_model"]
    assert changed["validation"] == fitted_result["validation"]
    # Test labels may veto trust, but must not affect the fitted prediction.
    assert changed["probability"] == pytest.approx(fitted_result["probability"], abs=1e-12)


def test_overflowed_future_return_is_missing_not_a_positive_label():
    stock, market = price_history()
    stock = stock.copy()
    stock.iloc[400, stock.columns.get_loc("Close")] = 1e-300
    stock.iloc[421, stock.columns.get_loc("Close")] = 1e300
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        features, target, metadata = ml.build_ml_dataset(stock, market, horizon=21)
    assert 399 not in metadata["original_positions"]
    assert np.isfinite(metadata["stock_forward_returns"]).all()
    json.dumps(metadata, allow_nan=False)
