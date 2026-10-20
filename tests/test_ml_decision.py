"""A learned prediction must earn its role in the final user-facing assessment."""
from copy import deepcopy
import json
import pytest

from finsight.analytics import analyze_stock, dcf_valuation
from finsight.data import demo_market_data
from finsight.decision import machine_learning_decision


@pytest.fixture
def evidence():
    bundle = demo_market_data()
    info = dict(bundle["info"], currentPrice=50., regularMarketPrice=50.)
    analysis = analyze_stock(info, bundle["history"], bundle["benchmark"])
    valuation = dcf_valuation(info)
    model = {"status": "available", "eligible_for_decision": True, "probability": .7,
             "signal": "BUY_SUPPORT", "horizon": 63, "cost_bps": 20,
             "selected_model": "logistic_regression", "target_description": "Positive net return and benchmark outperformance.",
             "data": {"latest_feature_date": bundle["price_as_of"]}, "reasons": []}
    args = dict(price_as_of=bundle["price_as_of"], quote_as_of=bundle["quote_as_of"],
                fundamentals_as_of=bundle["fundamentals_as_of"], now=bundle["fetched_at"])
    return analysis, valuation, info, model, args


def decide(evidence):
    analysis, valuation, info, model, args = evidence
    return machine_learning_decision(analysis, valuation, info, model, **args)


def test_validated_positive_ml_can_authorize_buy_without_fixed_factor_buy(evidence):
    result = decide(evidence)
    assert result["verdict"] == "BUY"
    assert result["evidence"]["ml_probability"] == .7
    assert result["basis"] == "machine_learning_with_risk_checks"
    assert result["fundamental_assessment"]["verdict"] != "BUY"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("probability", [.01, .70, .99])
def test_probability_without_earned_test_evidence_never_authorizes_call(evidence, probability):
    evidence[3].update(eligible_for_decision=False, probability=probability)
    result = decide(evidence)
    assert result["verdict"] == "INSUFFICIENT_DATA"
    assert not result["actionable"]


@pytest.mark.parametrize(("probability", "signal", "expected"), [(.2, "AVOID_SUPPORT", "DO_NOT_BUY"), (.5, "NEUTRAL", "WAIT")])
def test_qualified_model_signal_drives_the_final_assessment(evidence, probability, signal, expected):
    evidence[3].update(probability=probability, signal=signal)
    assert decide(evidence)["verdict"] == expected


def test_inconsistent_probability_and_signal_abstains(evidence):
    evidence[3]["probability"] = .1
    assert decide(evidence)["verdict"] == "INSUFFICIENT_DATA"


def test_cashflow_overvaluation_moderates_positive_model(evidence):
    evidence[2]["currentPrice"] = 1000.
    result = decide(evidence)
    assert result["verdict"] == "WAIT"
    assert any("valuation conflicts" in reason for reason in result["reasons"])


def test_negative_earnings_moderate_a_positive_model(evidence):
    evidence[2]["trailingEps"] = -1.
    assert decide(evidence)["verdict"] == "WAIT"


def test_extreme_risk_vetoes_positive_model(evidence):
    evidence[0]["score"]["factors"]["risk"]["score"] = 10.
    assert decide(evidence)["verdict"] == "DO_NOT_BUY"


def test_absent_dcf_does_not_masquerade_as_negative_ml_prediction(evidence):
    evidence[1].clear()
    result = decide(evidence)
    assert result["verdict"] == "BUY"
    assert result["fundamental_assessment"]["verdict"] == "INSUFFICIENT_DATA"


@pytest.mark.parametrize("change", ["stale_features", "stale_quote", "unknown_currency", "not_equity", "missing_risk", "invalid_risk"])
def test_fresh_applicable_data_are_required_even_with_validated_model(evidence, change):
    if change == "stale_features": evidence[3]["data"]["latest_feature_date"] = "2025-01-01"
    elif change == "stale_quote": evidence[4]["quote_as_of"] = "2025-01-01"
    elif change == "unknown_currency": evidence[2].pop("currency")
    elif change == "not_equity": evidence[2]["quoteType"] = "ETF"
    elif change == "invalid_risk": evidence[0]["score"]["factors"]["risk"]["score"] = 200.
    else: evidence[0]["score"]["factors"]["risk"]["score"] = None
    assert decide(evidence)["verdict"] == "INSUFFICIENT_DATA"


def test_demo_is_explicitly_synthetic_and_inputs_are_not_mutated(evidence):
    saved = deepcopy(evidence)
    analysis, valuation, info, model, args = evidence
    result = machine_learning_decision(analysis, valuation, info, model, **args, is_demo=True)
    assert not result["actionable"] and "DEMO" in result["title"]
    assert evidence == saved
