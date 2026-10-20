"""Portable strict JSON exports with an explicit source-data allowlist."""
from datetime import date, datetime
import json
import math
from numbers import Real, Integral


def json_value(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def export_report(bundle, analysis, valuation=None, forecast=None, news=None, decision=None, ml_result=None):
    """Exclude credentials and raw provider metadata from downloadable reports."""
    report = {
        "app": "Finsight", "version": "3.0.0",
        "symbol": bundle.get("symbol"), "is_demo": bundle.get("is_demo", False),
        "data": {key: bundle.get(key) for key in (
            "fetched_at", "price_as_of", "quote_as_of", "fundamentals_as_of", "price_basis", "currency", "benchmark_symbol", "warnings"
        )},
        "analysis": analysis, "valuation": valuation, "forecast": forecast, "news": news, "decision": decision, "machine_learning": ml_result,
        "interpretation": "Research rules and scenarios; final buy assessment is conditional on disclosed assumptions, not personal suitability or a probability of profit.",
    }
    return json.dumps(json_value(report), indent=2, allow_nan=False)
