"""Final research verdicts must follow disclosed evidence and abstain safely."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from finsight.analytics import dcf_valuation
from finsight.decision import investment_decision


NOW = datetime(2026, 10, 19, 16, 0, tzinfo=timezone.utc)


def evidence():
    """Complete independent evidence isolates decision rules from factor math."""
    info = {
        "symbol": "TEST", "quoteType": "EQUITY", "sector": "Technology",
        "industry": "Software", "currency": "USD", "financialCurrency": "USD",
        "currentPrice": 75., "freeCashflow": 5_000., "sharesOutstanding": 1_000.,
        "trailingEps": 4., "netIncomeToCommon": 4_000.,
        "totalStockholderEquity": 20_000., "bookValue": 20.,
    }
    factors = {
        name: {"score": score, "coverage": 1., "weight": weight,
               "available_metrics": 3, "total_metrics": 3, "metrics": []}
        for name, score, weight in (
            ("value", 70., .20), ("quality", 75., .20), ("growth", 65., .15),
            ("momentum", 65., .15), ("financial_strength", 70., .15),
            ("risk", 65., .15),
        )
    }
    analysis = {
        "schema_version": 1, "symbol": "TEST", "financial_company": False,
        "currency": "USD", "warnings": [],
        "score": {"overall": 75., "coverage": 1., "status": "available", "factors": factors},
        "technical": {"as_of": "2026-10-19", "latest_close": 40., "observations": 756},
    }
    valuation = dcf_valuation(info)
    valuation["fair_value_per_share"] = 100.
    for case, fair_value in zip(valuation["scenarios"], (85., 100., 120.)):
        case["fair_value_per_share"] = fair_value
        case["upside"] = fair_value / info["currentPrice"] - 1
    dates = {
        "price_as_of": "2026-10-19", "quote_as_of": "2026-10-19T15:55:00+00:00",
        "fundamentals_as_of": "2026-06-30T00:00:00+00:00", "now": NOW,
    }
    return analysis, valuation, info, dates


def decide(analysis, valuation, info, dates):
    return investment_decision(analysis, valuation, info, **dates)


def test_complete_strong_evidence_returns_explained_buy_and_json_safe_schema():
    analysis, valuation, info, dates = evidence()
    result = decide(analysis, valuation, info, dates)
    assert result["verdict"] == "BUY"
    assert result["actionable"] is True
    assert result["is_demo"] is False
    assert result["title"] and result["summary"] and result["methodology"]
    assert result["reasons"] and all(isinstance(reason, str) for reason in result["reasons"])
    assert isinstance(result["conditions_to_improve"], list)
    assert result["thresholds"]
    assert result["checks"]
    for check in result["checks"]:
        assert {"key", "label", "passed", "value", "threshold"} <= check.keys()
        assert check["passed"] in (True, False, None)
    assert result["evidence"]["quote"] == 75.
    assert result["evidence"]["base_fair_value"] == 100.
    assert result["evidence"]["margin_of_safety"] == pytest.approx(.25)
    assert result["evidence"]["coverage"] == 1.
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(("quote", "expected"), [(80., "BUY"), (80.01, "WAIT"),
                                                  (120., "DO_NOT_BUY"), (119.99, "WAIT")])
def test_margin_of_safety_uses_fair_value_denominator_and_exact_boundaries(quote, expected):
    analysis, valuation, info, dates = evidence()
    info["currentPrice"] = quote
    result = decide(analysis, valuation, info, dates)
    assert result["verdict"] == expected
    assert result["evidence"]["margin_of_safety"] == pytest.approx(1 - quote / 100.)


@pytest.mark.parametrize(("key", "threshold"), [("overall", 70.), ("quality", 65.),
                                                ("financial_strength", 60.), ("risk", 50.), ("value", 50.)])
def test_buy_factor_thresholds_are_inclusive_but_require_every_condition(key, threshold):
    analysis, valuation, info, dates = evidence()
    container = analysis["score"] if key == "overall" else analysis["score"]["factors"][key]
    score_key = "overall" if key == "overall" else "score"
    container[score_key] = threshold
    assert decide(analysis, valuation, info, dates)["verdict"] == "BUY"
    container[score_key] = threshold - .01
    assert decide(analysis, valuation, info, dates)["verdict"] == "WAIT"


@pytest.mark.parametrize(("key", "threshold"), [("overall", 40.), ("quality", 35.),
                                                ("financial_strength", 30.), ("risk", 25.)])
def test_weak_evidence_thresholds_are_strict_and_explained(key, threshold):
    analysis, valuation, info, dates = evidence()
    container = analysis["score"] if key == "overall" else analysis["score"]["factors"][key]
    score_key = "overall" if key == "overall" else "score"
    container[score_key] = threshold
    assert decide(analysis, valuation, info, dates)["verdict"] == "WAIT"
    container[score_key] = threshold - .01
    result = decide(analysis, valuation, info, dates)
    assert result["verdict"] == "DO_NOT_BUY"
    assert result["reasons"]


def test_global_coverage_threshold_is_required_even_with_every_factor_present():
    analysis, valuation, info, dates = evidence()
    analysis["score"]["coverage"] = .85
    assert decide(analysis, valuation, info, dates)["verdict"] == "BUY"
    analysis["score"]["coverage"] = .8499
    result = decide(analysis, valuation, info, dates)
    assert result["verdict"] == "INSUFFICIENT_DATA"
    assert result["actionable"] is False
    assert result["conditions_to_improve"]


@pytest.mark.parametrize(("factor", "minimum"), [("value", .8), ("quality", .75),
                                                ("financial_strength", 2/3), ("risk", 1.)])
def test_required_factor_coverage_cannot_be_masked_by_high_global_coverage(factor, minimum):
    analysis, valuation, info, dates = evidence()
    analysis["score"]["factors"][factor]["coverage"] = minimum
    assert decide(analysis, valuation, info, dates)["verdict"] == "BUY"
    analysis["score"]["factors"][factor]["coverage"] = minimum - .0001
    assert decide(analysis, valuation, info, dates)["verdict"] == "INSUFFICIENT_DATA"


@pytest.mark.parametrize("factor", ["value", "quality", "growth", "momentum", "financial_strength", "risk"])
def test_missing_any_factor_abstains_instead_of_renormalizing_to_buy(factor):
    analysis, valuation, info, dates = evidence()
    analysis["score"]["factors"][factor]["score"] = None
    assert decide(analysis, valuation, info, dates)["verdict"] == "INSUFFICIENT_DATA"


@pytest.mark.parametrize("value", [None, 0., -5., float("nan"), float("inf"), True])
def test_no_valid_current_quote_cannot_fall_back_to_adjusted_close(value):
    analysis, valuation, info, dates = evidence()
    info["currentPrice"] = value
    info.pop("regularMarketPrice", None)
    assert analysis["technical"]["latest_close"] > 0
    result = decide(analysis, valuation, info, dates)
    assert result["verdict"] == "INSUFFICIENT_DATA"
    assert result["evidence"]["quote"] is None
    json.dumps(result, allow_nan=False)


def test_regular_market_quote_is_accepted_when_current_price_is_missing():
    analysis, valuation, info, dates = evidence()
    info.pop("currentPrice")
    info["regularMarketPrice"] = 75.
    result = decide(analysis, valuation, info, dates)
    assert result["verdict"] == "BUY"
    assert result["evidence"]["quote"] == 75.


@pytest.mark.parametrize("key", ["price_as_of", "quote_as_of", "fundamentals_as_of"])
@pytest.mark.parametrize("value", [None, "not-a-date", "2026-10-20T00:00:00+00:00"])
def test_missing_invalid_and_future_dates_force_abstention(key, value):
    analysis, valuation, info, dates = evidence()
    dates[key] = value
    if key == "price_as_of":
        analysis["technical"]["as_of"] = value
    assert decide(analysis, valuation, info, dates)["verdict"] == "INSUFFICIENT_DATA"


def test_future_quote_time_on_same_calendar_day_abstains():
    analysis, valuation, info, dates = evidence()
    dates["quote_as_of"] = (NOW + timedelta(hours=1)).isoformat()
    assert decide(analysis, valuation, info, dates)["verdict"] == "INSUFFICIENT_DATA"


def test_timezone_offset_is_normalized_before_future_quote_check():
    analysis, valuation, info, dates = evidence()
    # Although its local date is tomorrow, this is today at 11:00 UTC.
    dates["quote_as_of"] = "2026-10-20T01:00:00+14:00"
    assert decide(analysis, valuation, info, dates)["verdict"] == "BUY"


@pytest.mark.parametrize(("key", "limit"), [("price_as_of", 7), ("quote_as_of", 7), ("fundamentals_as_of", 180)])
def test_date_recency_boundaries_use_calendar_days(key, limit):
    analysis, valuation, info, dates = evidence()
    dates[key] = (NOW.date() - timedelta(days=limit)).isoformat()
    assert decide(analysis, valuation, info, dates)["verdict"] == "BUY"
    dates[key] = (NOW.date() - timedelta(days=limit + 1)).isoformat()
    assert decide(analysis, valuation, info, dates)["verdict"] == "INSUFFICIENT_DATA"


@pytest.mark.parametrize(("key", "value"), [("currency", None), ("financialCurrency", None),
                                             ("financialCurrency", "EUR"), ("currency", "GBp"),
                                             ("quoteType", None), ("quoteType", "ETF"), ("quoteType", "MUTUALFUND")])
def test_missing_or_incompatible_instrument_and_currency_metadata_abstains(key, value):
    analysis, valuation, info, dates = evidence()
    info[key] = value
    assert decide(analysis, valuation, info, dates)["verdict"] == "INSUFFICIENT_DATA"


def test_pence_and_pounds_are_not_silently_treated_as_matching_currency_units():
    analysis, valuation, info, dates = evidence()
    info.update(currency="GBp", financialCurrency="GBP")
    assert decide(analysis, valuation, info, dates)["verdict"] == "INSUFFICIENT_DATA"


def test_supplied_valuation_must_also_use_the_verified_quote_currency():
    analysis, valuation, info, dates = evidence()
    valuation["currency"] = "EUR"
    assert decide(analysis, valuation, info, dates)["verdict"] == "INSUFFICIENT_DATA"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1., 101.])
def test_invalid_overall_scores_abstain_and_remain_json_safe(value):
    analysis, valuation, info, dates = evidence()
    analysis["score"]["overall"] = value
    result = decide(analysis, valuation, info, dates)
    assert result["verdict"] == "INSUFFICIENT_DATA"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1., 1.01])
def test_invalid_global_coverage_abstains_and_remains_json_safe(value):
    analysis, valuation, info, dates = evidence()
    analysis["score"]["coverage"] = value
    result = decide(analysis, valuation, info, dates)
    assert result["verdict"] == "INSUFFICIENT_DATA"
    json.dumps(result, allow_nan=False)


def test_financial_company_requires_specialist_valuation_instead_of_negative_verdict():
    analysis, valuation, info, dates = evidence()
    analysis["financial_company"] = True
    info.update(sector="Financial Services", industry="Banks - Regional")
    result = decide(analysis, valuation, info, dates)
    assert result["verdict"] == "INSUFFICIENT_DATA"


@pytest.mark.parametrize("invalid", ["unavailable", "missing_base", "missing_bear", "missing_bull", "negative_bear", "invalid_base"])
def test_incomplete_or_invalid_valuation_cannot_produce_any_actionable_verdict(invalid):
    analysis, valuation, info, dates = evidence()
    if invalid == "unavailable":
        valuation["status"] = "unavailable"
    elif invalid.startswith("missing_"):
        name = invalid.removeprefix("missing_").title()
        valuation["scenarios"] = [case for case in valuation["scenarios"] if case["name"] != name]
    elif invalid == "negative_bear":
        valuation["scenarios"][0]["fair_value_per_share"] = -1.
    else:
        valuation["fair_value_per_share"] = float("inf")
        valuation["scenarios"][1]["fair_value_per_share"] = float("inf")
    result = decide(analysis, valuation, info, dates)
    assert result["verdict"] == "INSUFFICIENT_DATA"
    assert result["actionable"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("key", ["trailingEps", "netIncomeToCommon", "totalStockholderEquity", "bookValue"])
@pytest.mark.parametrize("value", [0., -1.])
def test_nonpositive_earnings_and_book_equity_block_buy_without_inventing_a_rejection(key, value):
    analysis, valuation, info, dates = evidence()
    info[key] = value
    assert decide(analysis, valuation, info, dates)["verdict"] == "WAIT"


@pytest.mark.parametrize("value", [None, 0., -1., float("nan")])
def test_unusable_reported_cashflow_blocks_even_an_injected_available_valuation(value):
    analysis, valuation, info, dates = evidence()
    info["freeCashflow"] = value
    assert decide(analysis, valuation, info, dates)["verdict"] == "INSUFFICIENT_DATA"


def test_overflowing_derived_valuation_ratio_abstains_and_is_json_safe():
    analysis, valuation, info, dates = evidence()
    info["currentPrice"] = 1e308
    valuation["fair_value_per_share"] = 1e-308
    for case in valuation["scenarios"]:
        case["fair_value_per_share"] = 1e-308
    result = decide(analysis, valuation, info, dates)
    assert result["verdict"] == "INSUFFICIENT_DATA"
    assert result["evidence"]["margin_of_safety"] is None
    json.dumps(result, allow_nan=False)


def test_missing_data_takes_precedence_over_weak_scores_or_overvaluation():
    analysis, valuation, info, dates = evidence()
    info["currentPrice"] = 200.
    analysis["score"]["overall"] = 10.
    dates["quote_as_of"] = None
    result = decide(analysis, valuation, info, dates)
    assert result["verdict"] == "INSUFFICIENT_DATA"
    assert result["actionable"] is False


def test_editing_current_valuation_changes_verdict_without_using_saved_analysis_valuation():
    analysis, valuation, info, dates = evidence()
    analysis["valuation"] = deepcopy(valuation)
    assert decide(analysis, valuation, info, dates)["verdict"] == "BUY"
    valuation["fair_value_per_share"] = 50.
    for case, value in zip(valuation["scenarios"], (40., 50., 60.)):
        case["fair_value_per_share"] = value
    assert decide(analysis, valuation, info, dates)["verdict"] == "DO_NOT_BUY"


def test_demo_verdict_is_explicitly_non_actionable():
    analysis, valuation, info, dates = evidence()
    result = investment_decision(analysis, valuation, info, **dates, is_demo=True)
    assert result["is_demo"] is True
    assert result["actionable"] is False
    text = " ".join([result["title"], result["summary"], *result["reasons"]]).lower()
    assert "demo" in text or "synthetic" in text


def test_forecast_and_news_do_not_vote_or_claim_probabilities_of_profit():
    analysis, valuation, info, dates = evidence()
    baseline = decide(analysis, valuation, info, dates)
    analysis["forecast"] = {"probability_of_profit": 0., "predicted_return": -1.}
    analysis["news"] = {"sentiment": -1., "recommendation": "SELL"}
    result = decide(analysis, valuation, info, dates)
    assert result["verdict"] == baseline["verdict"] == "BUY"
    narrative = str(result["methodology"]).lower() + " " + result["summary"].lower()
    assert "rule" in narrative or "heuristic" in narrative
    assert "guaranteed" not in result["summary"].lower()


def test_decision_does_not_mutate_inputs():
    analysis, valuation, info, dates = evidence()
    before = deepcopy((analysis, valuation, info, dates))
    decide(analysis, valuation, info, dates)
    assert (analysis, valuation, info, dates) == before
