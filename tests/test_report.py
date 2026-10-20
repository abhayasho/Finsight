import json
import numpy as np
from finsight.report import export_report


def test_report_is_strict_json_and_excludes_secret_fields():
    output = export_report({"symbol": "TEST", "api_key": "secret", "info": {"key": "secret"}},
                           {"value": np.float64(1.2), "bad": float("nan"), "infinite": float("inf")})
    report = json.loads(output)
    assert report["analysis"] == {"value": 1.2, "bad": None, "infinite": None}
    assert "secret" not in output


def test_final_assessment_and_its_valuation_are_exported_together():
    valuation = {"fair_value_per_share": 100, "assumptions": {"discount_rate": .1}}
    decision = {"verdict": "BUY", "evidence": {"base_fair_value": 100, "margin_of_safety": .25}}
    report = json.loads(export_report({"symbol": "TEST"}, {}, valuation=valuation, decision=decision))
    assert report["decision"] == decision
    assert report["valuation"] == valuation
    assert report["version"] == "3.0.0"


def test_ml_evidence_is_exported_with_the_final_assessment():
    model = {"selected_model": "random_forest", "eligible_for_decision": False,
             "test": {"selected": {"brier": .25}}, "probability": .8}
    report = json.loads(export_report({"symbol": "TEST"}, {}, ml_result=model,
                       decision={"verdict": "INSUFFICIENT_DATA"}))
    assert report["machine_learning"] == model
    assert report["decision"]["verdict"] == "INSUFFICIENT_DATA"
