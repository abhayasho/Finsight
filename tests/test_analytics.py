"""Math, missing-data, chronology and robustness checks for research models."""

import json
import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from finsight import analytics
from finsight.analytics import analyze_stock, dcf_valuation, forecast_prices


def history(prices, start="2023-01-02"):
    return pd.DataFrame({"Close": prices}, index=pd.bdate_range(start, periods=len(prices)))


def sample_history(size=300):
    rng = np.random.default_rng(42)
    return history(100 * np.exp(np.cumsum(rng.normal(.0005, .012, size))))


def complete_info(**changes):
    info = {
        "symbol": "TEST", "quoteType": "EQUITY", "sector": "Technology", "industry": "Software",
        "currency": "USD", "financialCurrency": "USD", "currentPrice": 100.,
        "marketCap": 100_000., "sharesOutstanding": 1_000., "trailingEps": 5., "trailingPE": 20.,
        "freeCashflow": 5_000., "totalRevenue": 50_000., "enterpriseToEbitda": 12.,
        "returnOnEquity": .20, "returnOnAssets": .10, "operatingMargins": .20,
        "revenueGrowth": .10, "earningsGrowth": .15, "totalDebt": 10_000., "totalCash": 5_000.,
        "ebitda": 10_000., "currentRatio": 2.,
    }
    info.update(changes)
    return info


def test_complete_score_has_all_factors_bounded_and_coverage_one():
    result = analyze_stock(complete_info(), sample_history())
    assert result["score"]["status"] == "available"
    assert result["score"]["coverage"] == pytest.approx(1.)
    assert 0 <= result["score"]["overall"] <= 100
    assert sum(row["weight"] for row in result["score"]["factors"].values()) == pytest.approx(1.)
    for factor in result["score"]["factors"].values():
        assert 0 <= factor["score"] <= 100
        assert factor["coverage"] == 1


def test_missing_fundamentals_abstain_instead_of_neutral_scores():
    result = analyze_stock({}, sample_history())
    assert result["score"]["overall"] is None
    assert result["score"]["coverage"] == pytest.approx(.30)
    for name in ("value", "quality", "growth", "financial_strength"):
        assert result["score"]["factors"][name]["score"] is None
        assert result["score"]["factors"][name]["coverage"] == 0


def test_coverage_uses_metric_weights_not_available_factor_count():
    full = analyze_stock(complete_info(), sample_history())
    partial_info = complete_info(enterpriseToEbitda=None)
    partial = analyze_stock(partial_info, sample_history())
    assert partial["score"]["factors"]["value"]["coverage"] == pytest.approx(4/5)
    assert partial["score"]["coverage"] == pytest.approx(full["score"]["coverage"] - .20/5)


def test_factor_includes_all_evidence_and_reweights_missing_metrics():
    result = analyze_stock(complete_info(enterpriseToEbitda=None), sample_history())
    value = result["score"]["factors"]["value"]
    available = [metric for metric in value["metrics"] if metric["available"]]
    expected = sum(metric["score"] * metric["weight"] for metric in available) / sum(metric["weight"] for metric in available)
    assert value["score"] == pytest.approx(expected)
    missing = next(metric for metric in value["metrics"] if not metric["available"])
    assert missing["score"] is None and missing["reason"]


def test_score_scale_is_continuous_and_monotonic():
    low = analyze_stock(complete_info(revenueGrowth=.10), sample_history())
    high = analyze_stock(complete_info(revenueGrowth=.11), sample_history())
    difference = high["score"]["overall"] - low["score"]["overall"]
    assert 0 < difference < 1
    extremes = analyze_stock(complete_info(revenueGrowth=1e20, earningsGrowth=-1e20), sample_history())
    assert 0 <= extremes["score"]["factors"]["growth"]["score"] <= 100


def test_nonfinite_and_boolean_fundamentals_are_missing_and_json_safe():
    info = complete_info(trailingEps=np.nan, trailingPE=np.inf, freeCashflow=-np.inf,
                         returnOnEquity=True, currentRatio="not a number", revenueGrowth=np.nan)
    result = analyze_stock(info, sample_history())
    for name, key in (("value", "earnings_yield"), ("quality", "roe"), ("growth", "revenue_growth")):
        metric = next(item for item in result["score"]["factors"][name]["metrics"] if item["key"] == key)
        assert not metric["available"]
    json.dumps(result, allow_nan=False)


def test_financials_use_actual_sector_and_skip_inapplicable_ratios():
    info = complete_info(sector="Financial Services", industry="Banks - Regional", priceToBook=1.2,
                         profitMargins=.25, totalAssets=1_000_000., totalStockholderEquity=100_000.)
    result = analyze_stock(info, sample_history())
    assert result["sector"] == "Financial Services"
    assert result["industry"] == "Banks - Regional"
    assert result["financial_company"]
    keys = {metric["key"] for row in result["score"]["factors"].values() for metric in row["metrics"]}
    assert "equity_assets" in keys and "price_to_book" in keys
    assert not {"ev_ebitda", "fcf_yield", "current_ratio", "debt_ebitda"} & keys
    assert result["valuation"]["status"] == "unavailable"
    assert any("Regulatory capital" in warning for warning in result["warnings"])


def test_non_equity_never_gets_company_fundamental_score():
    result = analyze_stock(complete_info(quoteType="ETF"), sample_history())
    assert result["score"]["overall"] is None
    assert result["score"]["factors"]["value"]["score"] is None
    assert result["valuation"]["status"] == "unavailable"


def test_cross_currency_financial_ratios_excluded():
    result = analyze_stock(complete_info(financialCurrency="EUR"), sample_history())
    metrics = {metric["key"]: metric for factor in result["score"]["factors"].values() for metric in factor["metrics"]}
    assert metrics["fcf_yield"]["value"] is None
    assert metrics["net_debt_market_cap"]["value"] is None
    assert result["valuation"]["status"] == "unavailable"


def test_history_cleaning_sort_duplicates_invalid_and_adjusted_column():
    index = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-04", "2024-01-05"])
    frame = pd.DataFrame({"Close": [999] * 5, "Adj Close": [102, 100, 101, np.inf, -1]}, index=index)
    result = analyze_stock({}, frame)
    assert result["technical"]["observations"] == 2
    assert result["technical"]["latest_close"] == 102
    assert any("duplicate" in warning for warning in result["warnings"])
    assert any("Excluded 2" in warning for warning in result["warnings"])


def test_short_and_empty_history_do_not_invent_technicals():
    for frame in (pd.DataFrame(), history([100., 101.])):
        result = analyze_stock({}, frame)
        technical = result["technical"]
        assert technical["moving_averages"]["ma20"] is None
        assert technical["rsi14"] is None
        assert technical["annualized_volatility"] is None
        assert result["score"]["overall"] is None
        json.dumps(result, allow_nan=False)


def test_numeric_dates_are_rejected_instead_of_becoming_1970_sessions():
    result = analyze_stock({}, pd.DataFrame({"Close": [100.] * 300}))
    assert result["technical"]["observations"] == 0
    assert any("numeric row labels" in warning for warning in result["warnings"])


def test_rsi_wilder_seed_and_flat_up_down_edges():
    prices = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
              45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    result = analyze_stock({}, history(prices))["technical"]
    assert result["rsi14"] == pytest.approx(70.464135, abs=1e-5)
    for values, expected in (([100.] * 20, 50.), (np.arange(1., 21.), 100.), (np.arange(20., 0., -1), 0.)):
        assert analyze_stock({}, history(values))["technical"]["rsi14"] == expected


def test_risk_statistics_match_hand_computed_returns():
    returns = np.tile([.01, -.02, .03, -.01], 40)
    prices = 100 * np.cumprod(np.r_[1., 1 + returns])
    technical = analyze_stock({}, history(prices))["technical"]
    assert technical["annualized_volatility"] == pytest.approx(np.std(returns, ddof=1) * math.sqrt(252))
    assert technical["downside_volatility"] == pytest.approx(np.sqrt(np.mean(np.minimum(returns, 0) ** 2)) * math.sqrt(252))
    assert technical["historical_var95"] == pytest.approx(.02)
    assert technical["historical_cvar95"] == pytest.approx(.02)
    assert technical["max_drawdown"] == pytest.approx(np.min(prices / np.maximum.accumulate(prices) - 1))


def test_all_positive_returns_have_zero_historical_loss_and_downside():
    technical = analyze_stock({}, history(100 * 1.01 ** np.arange(100)))["technical"]
    assert technical["historical_var95"] == 0
    assert technical["historical_cvar95"] == 0
    assert technical["downside_volatility"] == 0
    assert technical["max_drawdown"] == 0


def test_beta_aligns_prices_before_returns_when_benchmark_dates_missing():
    market = sample_history()
    stock = market * 2
    benchmark = market.iloc[::2]
    technical = analyze_stock({}, stock, benchmark)["technical"]
    assert technical["benchmark_observations"] == len(benchmark) - 1
    assert technical["beta"] == pytest.approx(1., abs=1e-12)


def test_flat_or_short_benchmark_beta_abstains():
    stock = sample_history()
    flat = history([100.] * len(stock))
    assert analyze_stock({}, stock, flat)["technical"]["beta"] is None
    assert analyze_stock({}, stock, stock.iloc[:50])["technical"]["beta"] is None


@pytest.mark.parametrize("rate", [float("nan"), "invalid", -1., None])
def test_invalid_risk_free_rate_omits_risk_adjusted_statistics(rate):
    result = analyze_stock({}, sample_history(), risk_free_rate=rate)
    assert result["technical"]["sharpe"] is None
    assert result["technical"]["sortino"] is None
    json.dumps(result, allow_nan=False)


def test_constant_equity_cashflow_dcf_is_perpetuity_without_debt_double_count():
    info = complete_info(freeCashflow=1_000., sharesOutstanding=100., totalDebt=100_000., totalCash=10_000.)
    result = dcf_valuation(info, growth=0., discount_rate=.10, terminal_growth=0., years=5)
    assert result["status"] == "available"
    assert result["fair_value_per_share"] == pytest.approx(100.)
    changed_balances = dcf_valuation(dict(info, totalDebt=0, totalCash=999_999.), growth=0., discount_rate=.10, terminal_growth=0.)
    assert changed_balances["fair_value_per_share"] == pytest.approx(100.)
    assert result["assumptions"]["net_borrowing"] == 0
    assert "No debt deduction" in result["assumptions"]["debt_cash_treatment"]


def test_dcf_share_units_and_equity_value_are_consistent():
    original = dcf_valuation(complete_info())
    doubled_shares = dcf_valuation(complete_info(sharesOutstanding=2_000.))
    assert doubled_shares["fair_value_per_share"] == pytest.approx(original["fair_value_per_share"] / 2)
    base = original["scenarios"][1]
    assert base["equity_value"] == pytest.approx(base["present_value_cashflows"] + base["present_value_terminal"])
    assert base["fair_value_per_share"] == pytest.approx(base["equity_value"] / 1_000.)
    assert 0 < base["terminal_value_share"] < 1


def test_dcf_scenarios_and_sensitivity_monotonic():
    result = dcf_valuation(complete_info())
    bear, base, bull = result["scenarios"]
    assert bear["fair_value_per_share"] < base["fair_value_per_share"] < bull["fair_value_per_share"]
    cells = result["sensitivity"]
    assert len(cells) == 9
    assert cells[0]["fair_value_per_share"] > cells[6]["fair_value_per_share"]
    assert cells[0]["fair_value_per_share"] < cells[2]["fair_value_per_share"]


@pytest.mark.parametrize("changes", [{"freeCashflow": 0}, {"freeCashflow": -100}, {"sharesOutstanding": 0},
                                     {"financialCurrency": "EUR"}, {"sector": "Financial Services"}, {"freeCashflow": np.inf}])
def test_invalid_dcf_data_abstains_with_reasons(changes):
    result = dcf_valuation(complete_info(**changes))
    assert result["status"] == "unavailable"
    assert result["fair_value_per_share"] is None
    assert result["reasons"]


@pytest.mark.parametrize("parameters", [{"discount_rate": .02, "terminal_growth": .03}, {"growth": -1},
                                        {"years": 2.5}, {"years": 0}, {"years": True}, {"discount_rate": np.nan}])
def test_invalid_dcf_parameters_abstain(parameters):
    result = dcf_valuation(complete_info(), **parameters)
    assert result["status"] == "unavailable"
    json.dumps(result, allow_nan=False)


def test_dcf_invalid_bull_or_sensitivity_cell_is_not_silently_repaired():
    result = dcf_valuation(complete_info(), discount_rate=.04, terminal_growth=.035)
    assert result["status"] == "available"
    assert result["scenarios"][2]["fair_value_per_share"] is None
    assert result["scenarios"][2]["reason"]
    assert any(cell["fair_value_per_share"] is None for cell in result["sensitivity"])


def test_forecast_constant_history_retains_baseline_and_exact_bands():
    result = forecast_prices(history([100.] * 250), horizon=21)
    assert result["status"] == "available"
    assert result["selected_model"] == "random_walk"
    assert len(result["forecast"]) == 21
    for row in result["forecast"]:
        for key in ("point_forecast", "median", "lower_80", "upper_80", "lower_95", "upper_95"):
            assert row[key] == pytest.approx(100.)
    assert result["backtest"]["models"][0]["rmse"] == pytest.approx(0., abs=1e-12)


def test_forecast_trend_can_select_drift_at_same_horizon():
    result = forecast_prices(history(100 * np.exp(.001 * np.arange(250))), horizon=21)
    assert result["status"] == "available"
    assert result["selected_model"] != "random_walk"
    assert result["backtest"]["rmse_improvement_over_baseline"] > .99
    assert all(row["horizon"] == 21 for row in result["backtest"]["models"])


def test_forecast_baseline_error_is_true_future_endpoint_error():
    frame = sample_history(250)
    result = forecast_prices(frame, horizon=21)
    backtest = result["backtest"]
    first = backtest["first_train_observations"]
    last = len(frame) - 21
    origins = list(range(first, last + 1, backtest["origin_stride"]))
    if origins[-1] != last:
        origins.append(last)
    origins = origins[-60:]
    prices = frame["Close"].to_numpy()
    errors = np.array([prices[origin - 1] - prices[origin + 20] for origin in origins])
    baseline = next(row for row in backtest["models"] if row["name"] == "random_walk")
    assert baseline["mae"] == pytest.approx(np.mean(np.abs(errors)))
    assert baseline["rmse"] == pytest.approx(np.sqrt(np.mean(errors ** 2)))
    assert baseline["n_predictions"] == len(origins)
    assert not backtest["independent_final_test"]


def test_forecast_every_backtest_fit_uses_only_its_training_prefix():
    frame = sample_history(250)
    full_logs = np.log(frame["Close"].to_numpy())
    calls = []
    original = analytics._forecast_log

    def observe(prefix, name, horizon):
        calls.append((prefix.copy(), name, horizon))
        return original(prefix, name, horizon)

    with patch.object(analytics, "_forecast_log", side_effect=observe):
        result = forecast_prices(frame, horizon=10)
    n_origins = result["backtest"]["n_origins"]
    assert len(calls) == 4 * n_origins + 1
    for prefix, name, horizon in calls[:-1]:
        assert len(prefix) <= len(full_logs) - 10
        np.testing.assert_array_equal(prefix, full_logs[:len(prefix)])
        assert horizon == 10
    assert len(calls[-1][0]) == len(full_logs)  # Only the actual future forecast refits on all observations.


def test_ridge_scaling_is_fit_on_training_data_only():
    train = np.log(sample_history(100)["Close"].to_numpy())
    model = analytics._fit_model(train, "ridge_ar5")
    returns = np.diff(train)
    x = np.array([returns[index-5:index][::-1] for index in range(5, len(returns))])
    np.testing.assert_allclose(model["mean"], x.mean(axis=0))
    np.testing.assert_allclose(model["scale"], x.std(axis=0))
    future_extreme = np.r_[train, train[-1] + 1.]
    changed = analytics._fit_model(future_extreme, "ridge_ar5")
    assert not np.allclose(changed["mean"], model["mean"])


def test_forecast_bands_order_reproducibility_and_serialization():
    first = forecast_prices(sample_history(300), horizon=10)
    second = forecast_prices(sample_history(300), horizon=10)
    assert first == second
    for row in first["forecast"]:
        assert 0 < row["lower_95"] <= row["lower_80"] <= row["median"] <= row["upper_80"] <= row["upper_95"]
    json.dumps(first, allow_nan=False)
    assert first["backtest"]["overlapping_targets"]
    assert any("not calibrated" in warning for warning in first["warnings"])


@pytest.mark.parametrize("horizon", [0, 127, 1.5, True, None])
def test_invalid_forecast_horizon_abstains(horizon):
    result = forecast_prices(sample_history(), horizon=horizon)
    assert result["status"] == "insufficient_data"
    assert not result["forecast"]


def test_short_history_and_long_horizon_forecast_limits():
    result = forecast_prices(sample_history(99), horizon=21)
    assert result["status"] == "insufficient_data"
    sufficient = forecast_prices(sample_history(190), horizon=126)
    assert sufficient["status"] == "available"
    assert sufficient["backtest"]["n_origins"] >= 5
    assert len(sufficient["forecast"]) == 126


def test_non_daily_and_corrupt_extreme_histories_do_not_forecast():
    monthly = sample_history(150)
    monthly.index = pd.date_range("2000-01-01", periods=150, freq="MS")
    assert forecast_prices(monthly)["status"] == "insufficient_data"
    corrupt = sample_history()
    corrupt.iloc[100, 0] *= 100
    assert forecast_prices(corrupt)["status"] == "insufficient_data"


def test_history_is_never_mutated_by_analysis_or_forecast():
    frame = sample_history()
    saved = frame.copy(deep=True)
    analyze_stock(complete_info(), frame, frame)
    forecast_prices(frame)
    pd.testing.assert_frame_equal(frame, saved)


def test_random_walk_has_no_directional_prediction():
    result = forecast_prices(sample_history(), horizon=10)
    baseline = result["backtest"]["models"][0]
    assert baseline["directional_accuracy"] is None
    assert baseline["directional_coverage"] == 0


def test_negative_book_equity_and_earnings_do_not_get_favorable_ratios():
    result = analyze_stock(complete_info(bookValue=-10, returnOnEquity=.8,
                           trailingEps=-2, netIncomeToCommon=-200, earningsGrowth=4), sample_history())
    metrics = {m["key"]: m for f in result["score"]["factors"].values() for m in f["metrics"]}
    assert not metrics["roe"]["available"]
    assert not metrics["earnings_growth"]["available"]
    assert metrics["earnings_yield"]["value"] < 0
