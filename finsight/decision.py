"""Explainable buy-entry assessment using disclosed rules, not an AI opinion.

This assesses a prospective purchase under the selected valuation assumptions.
It does not assess whether to sell an existing holding or personal suitability.
Missing evidence abstains; unsupported securities are not automatically rejected
as bad investments. All rates/coverage use fractions, scores use 0..100.
"""
from datetime import date, datetime, timedelta, timezone
from copy import deepcopy
import math


THRESHOLDS = {
    "minimum_coverage": .85,
    "factor_minimum_coverage": {"value": .8, "quality": .75, "financial_strength": 2 / 3, "risk": 1.},
    "buy_minimum_scores": {"overall": 70., "quality": 65., "financial_strength": 60., "risk": 50., "value": 50.},
    "avoid_below_scores": {"overall": 40., "quality": 35., "financial_strength": 30., "risk": 25.},
    "minimum_margin_of_safety": .20,
    "avoid_margin_of_safety_at_or_below": -.20,
    "maximum_price_age_days": 7,
    "maximum_reported_quarter_age_days": 180,
}


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _timestamp(value):
    try:
        if isinstance(value, datetime):
            result = value
        elif isinstance(value, date):
            result = datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            result = datetime.fromtimestamp(value, timezone.utc)
        elif isinstance(value, str):
            result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        else:
            return None
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def investment_decision(analysis, valuation, info, *, price_as_of=None, quote_as_of=None,
                        fundamentals_as_of=None, is_demo=False, now=None):
    """Return BUY / WAIT / DO_NOT_BUY / INSUFFICIENT_DATA plus auditable evidence.

    BUY requires complete, recent, applicable evidence; strong research factors;
    positive available earnings/equity; and price <= 80% of base scenario value.
    A missing valuation never becomes a negative verdict. Forecasts and news do
    not enter this rule. The exact same edited valuation should be exported.
    """
    analysis = analysis if isinstance(analysis, dict) else {}
    valuation = valuation if isinstance(valuation, dict) else {}
    info = info if isinstance(info, dict) else {}
    current = _timestamp(now) or datetime.now(timezone.utc)
    score = analysis.get("score") or {}
    factors = score.get("factors") or {}
    overall, coverage = _number(score.get("overall")), _number(score.get("coverage"))
    scores = {name: _number((factors.get(name) or {}).get("score"))
              for name in ("value", "quality", "growth", "momentum", "financial_strength", "risk")}
    scores["overall"] = overall
    quote = _number(info.get("currentPrice"))
    if quote is None or quote <= 0:
        quote = _number(info.get("regularMarketPrice"))
    quote = quote if quote is not None and quote > 0 else None
    fair = _number(valuation.get("fair_value_per_share"))
    fair = fair if fair is not None and fair > 0 else None
    # Scenario records, when available, are authoritative for the base case.
    scenarios = {str(item.get("name", "")).lower(): item for item in valuation.get("scenarios", []) if isinstance(item, dict)}
    base_fair = _number(scenarios.get("base", {}).get("fair_value_per_share"))
    if base_fair is not None and base_fair > 0:
        fair = base_fair
    margin = _number(1 - quote / fair) if quote is not None and fair is not None else None
    checks, missing, buy_failures, avoid_reasons = [], [], [], []
    conditions = []

    def gate(key, label, passed, value=None, threshold=None, reason=None):
        checks.append({"key": key, "label": label, "passed": bool(passed), "value": value, "threshold": threshold})
        if not passed:
            missing.append(reason or label)

    def dated(key, label, value, maximum):
        stamp = _timestamp(value)
        age = (current.date() - stamp.date()).days if stamp is not None else None
        valid = (stamp is not None and 0 <= age <= maximum
                 and stamp <= current + timedelta(minutes=5))
        gate(key, label, valid, age, f"0–{maximum} calendar days",
             f"{label}: a valid recent provider date is required (at most {maximum} calendar days old).")
        return age

    gate("instrument", "Supported operating-company equity", str(info.get("quoteType") or "").upper() == "EQUITY",
         info.get("quoteType"), "EQUITY", "The final assessment requires explicit operating-company equity metadata.")
    sector = str(info.get("sector") or "").lower()
    industry = str(info.get("industry") or "").lower()
    financial = bool(analysis.get("financial_company")) or sector in {"financial services", "financials", "financial"} or any(
        word in industry for word in ("banks", "banking", "insurance", "credit services", "capital markets", "mortgage reit"))
    gate("sector_model", "Applicable valuation model", not financial, info.get("sector"), "Non-financial operating company",
         "Financial companies need a specialist valuation and regulatory-capital assessment before a buy decision.")
    currency, financial_currency = info.get("currency"), info.get("financialCurrency")
    currencies_match = bool(isinstance(currency, str) and currency.strip() and isinstance(financial_currency, str)
                            and financial_currency.strip() and currency == financial_currency)
    gate("currency", "Verified matching currency units", currencies_match,
         {"quote": currency, "financials": financial_currency}, "Exact match; no implicit FX conversion",
         "Quote and financial-statement currency units must both be known and match.")
    gate("quote", "Positive current market quote", quote is not None, quote, "> 0",
         "A positive current market quote is required; adjusted historical prices are not a substitute.")
    quote_age = dated("quote_date", "Quote date", quote_as_of if quote_as_of is not None else info.get("regularMarketTime"), 7)
    history_age = dated("history_date", "Price-history date", price_as_of, 7)
    quarter_age = dated("quarter_date", "Reported-quarter end", fundamentals_as_of if fundamentals_as_of is not None else info.get("mostRecentQuarter"), 180)
    complete_scores = all(value is not None and 0 <= value <= 100 for value in scores.values())
    gate("scores", "Available bounded research scores", score.get("status") == "available" and complete_scores,
         overall, "Overall and all six factors available", "The overall score and all six factor scores must be available and valid.")
    gate("coverage", "Weighted model data coverage", coverage is not None and .85 - 1e-12 <= coverage <= 1,
         coverage, THRESHOLDS["minimum_coverage"], "At least 85% weighted model input coverage is required for a final assessment.")
    for name, minimum in THRESHOLDS["factor_minimum_coverage"].items():
        available = _number((factors.get(name) or {}).get("coverage"))
        gate(f"{name}_coverage", f"{name.replace('_', ' ').title()} input coverage",
             available is not None and minimum - 1e-12 <= available <= 1,
             available, minimum, f"More {name.replace('_', ' ')} inputs are needed; overall coverage cannot replace them.")
    complete_scenarios = all((_number(scenarios.get(name, {}).get("fair_value_per_share")) or 0) > 0
                             for name in ("bear", "base", "bull"))
    valuation_currency = valuation.get("currency")
    valuation_valid = (valuation.get("status") == "available" and fair is not None and margin is not None and complete_scenarios
                       and valuation_currency == currency and currencies_match)
    gate("valuation", "Supported complete cashflow scenarios", valuation_valid, fair,
         "Positive bear/base/bull values in quote currency",
         "A supported valuation with valid bear, base and bull scenarios in the quote currency is required.")
    fcf = _number(info.get("freeCashflow"))
    gate("cashflow", "Positive reported free cashflow", fcf is not None and fcf > 0, fcf, "> 0",
         "Positive reported cashflow is required for this valuation method; another method is needed otherwise.")

    for key, minimum in THRESHOLDS["buy_minimum_scores"].items():
        value = scores[key]
        passed = value is not None and value >= minimum
        label = f"{key.replace('_', ' ').title()} score"
        checks.append({"key": f"buy_{key}", "label": label, "passed": passed if value is not None else None,
                       "value": value, "threshold": f">= {minimum:g} / 100"})
        if not passed:
            buy_failures.append(f"{label} is {value:.1f}/100; the buy rule requires at least {minimum:g}." if value is not None else f"{label} is unavailable.")
            conditions.append(f"{label} reaches {minimum:g}/100 with sufficient supporting data.")
    for key, limit in THRESHOLDS["avoid_below_scores"].items():
        value = scores[key]
        if value is not None and value < limit:
            avoid_reasons.append(f"{key.replace('_', ' ').title()} score is {value:.1f}/100, below the {limit:g} minimum for considering a new purchase.")
    margin_pass = margin is not None and margin >= .20 - 1e-12
    checks.append({"key": "margin_of_safety", "label": "Margin of safety to base scenario", "passed": margin_pass if margin is not None else None,
                   "value": margin, "threshold": ">= 20%; market quote <= 80% of base scenario value"})
    if not margin_pass:
        buy_failures.append(f"Margin of safety is {margin:.1%}; the buy rule requires 20%." if margin is not None else "Margin of safety is unavailable.")
        if fair is not None:
            conditions.append(f"Market price is at or below {fair * .8:,.2f} {currency or ''}, provided the valuation and other evidence still hold.")
    if margin is not None and margin <= -.20 + 1e-12:
        avoid_reasons.append(f"The current quote is at least 20% above the base cashflow scenario value (margin of safety {margin:.1%}).")
    losses = []
    for key, label in (("trailingEps", "Trailing EPS"), ("netIncomeToCommon", "Net income"),
                       ("bookValue", "Book equity per share"), ("totalStockholderEquity", "Reported equity"),
                       ("stockholdersEquity", "Reported equity")):
        value = _number(info.get(key))
        if value is not None and value <= 0:
            losses.append(f"{label} is nonpositive; the buy rule excludes this condition.")
    buy_failures.extend(dict.fromkeys(losses))
    if losses:
        conditions.append("Reported earnings and book equity are positive; review the original financial statements.")
    if missing:
        verdict, title = "INSUFFICIENT_DATA", "Cannot decide — more data needed"
        summary = "The evidence does not support a reliable buy-or-no-buy assessment with this model."
        reasons = missing
        conditions = ["Resolve the missing, stale or unsupported evidence listed above before interpreting a final assessment."]
    elif avoid_reasons:
        verdict, title = "DO_NOT_BUY", "No — do not buy at this price"
        summary = "The current evidence fails a disclosed valuation, business-strength or risk rule for a new purchase."
        reasons = avoid_reasons
    elif not buy_failures:
        verdict, title = "BUY", "Yes — buy candidate"
        summary = "The stock passes the buy rules under the selected cashflow assumptions and current market quote."
        reasons = [f"Overall score {overall:.1f}/100 with {coverage:.0%} input coverage and all required factor inputs.",
                   f"Quality {scores['quality']:.1f}, financial strength {scores['financial_strength']:.1f}, risk {scores['risk']:.1f} and value {scores['value']:.1f} pass the buy thresholds.",
                   f"The current quote has {margin:.1%} margin of safety to the base cashflow scenario value."]
        conditions = ["Reassess if the price rises, reported results change, risk worsens, or your cashflow assumptions change."]
    else:
        verdict, title = "WAIT", "Not yet — wait before buying"
        summary = "Evidence is usable but mixed, or the price lacks the required margin of safety."
        reasons = buy_failures
    if is_demo:
        title = "DEMO · " + title
        summary = "Synthetic example only. " + summary
    return {
        "verdict": verdict, "title": title, "summary": summary,
        "reasons": list(dict.fromkeys(reasons)), "conditions_to_improve": list(dict.fromkeys(conditions)),
        "checks": checks, "thresholds": deepcopy(THRESHOLDS),
        "evidence": {"quote": quote, "currency": currency, "base_fair_value": fair,
                     "margin_of_safety": margin, "buy_price_at_current_assumptions": fair * .8 if fair is not None else None,
                     "coverage": coverage, "scores": scores, "quote_age_days": quote_age,
                     "history_age_days": history_age, "reported_quarter_age_days": quarter_age,
                     "valuation_assumptions": valuation.get("assumptions", {})},
        "actionable": not is_demo and verdict != "INSUFFICIENT_DATA", "is_demo": bool(is_demo),
        "as_of": current.isoformat(),
        "methodology": "Rule-based assessment of a new long-term stock purchase under current valuation assumptions. Thresholds are subjective, not empirically calibrated; no forecast or news/LLM vote. Coverage is not confidence. No personal goals, portfolio, time horizon or risk tolerance are known; no inference about whether to sell an existing holding.",
    }


def machine_learning_decision(analysis, valuation, info, ml_result, *, price_as_of=None,
                              quote_as_of=None, fundamentals_as_of=None, is_demo=False, now=None):
    """Use an eligible supervised prediction as the primary buy signal.

    The existing fundamental assessment remains separate context. Its fixed
    score/margin thresholds are not relabeled as learned predictions. Quote and
    feature freshness gates still apply; severe measured risk vetoes a purchase,
    while conflicting financial evidence downgrades a positive ML signal to Wait.
    """
    context = investment_decision(analysis, valuation, info, price_as_of=price_as_of,
        quote_as_of=quote_as_of, fundamentals_as_of=fundamentals_as_of, is_demo=is_demo, now=now)
    result = deepcopy(context)
    result["basis"] = "machine_learning_with_risk_checks"
    result["fundamental_assessment"] = {key: context[key] for key in ("verdict", "title", "reasons", "checks", "thresholds")}
    ml_result = ml_result if isinstance(ml_result, dict) else {}
    probability = _number(ml_result.get("probability"))
    signal = ml_result.get("signal")
    horizon = ml_result.get("horizon")
    usable = (ml_result.get("status") == "available" and ml_result.get("eligible_for_decision") is True
              and probability is not None and 0 <= probability <= 1
              and isinstance(horizon, int) and not isinstance(horizon, bool) and horizon > 0
              and signal in {"BUY_SUPPORT", "AVOID_SUPPORT", "NEUTRAL"})
    if usable:
        usable = ((signal == "BUY_SUPPORT" and probability >= .60 - 1e-12)
                  or (signal == "AVOID_SUPPORT" and probability <= .40 + 1e-12)
                  or (signal == "NEUTRAL" and .40 < probability < .60))
    checks = [deepcopy(check) for check in context["checks"]
              if check["key"] in {"instrument", "quote", "quote_date", "history_date"}]
    currency = info.get("currency") if isinstance(info, dict) else None
    checks.append({"key": "quote_currency", "label": "Known quote currency", "passed": bool(isinstance(currency, str) and currency.strip()),
                   "value": currency, "threshold": "Provider quote currency supplied"})
    feature_date = (ml_result.get("data") or {}).get("latest_feature_date")
    feature_matches = (_timestamp(feature_date) is not None and _timestamp(price_as_of) is not None
                       and _timestamp(feature_date).date() == _timestamp(price_as_of).date())
    checks.append({"key": "ml_feature_date", "label": "Model features match the latest price session", "passed": feature_matches,
                   "value": feature_date, "threshold": price_as_of})
    checks.append({"key": "ml_validation", "label": "ML passed its untouched test qualification", "passed": usable,
                   "value": ml_result.get("selected_model"), "threshold": "Disclosed model eligibility checks passed"})
    missing = [f"{check['label']} is unavailable or does not meet its requirement." for check in checks if not check["passed"]]
    result["evidence"].update({"ml_probability": probability, "ml_signal": signal, "ml_horizon_sessions": horizon,
        "ml_selected_model": ml_result.get("selected_model"), "ml_target": ml_result.get("target_description"),
        "ml_eligible": usable, "ml_feature_date": feature_date, "ml_cost_bps": ml_result.get("cost_bps")})
    factor_evidence = ((analysis.get("score") or {}).get("factors") or {}) if isinstance(analysis, dict) else {}
    risk = _number((factor_evidence.get("risk") or {}).get("score"))
    conflicts = []
    if isinstance(info, dict):
        for key, name in (("trailingEps", "Trailing EPS"), ("netIncomeToCommon", "Reported net income"),
                          ("bookValue", "Reported book equity per share")):
            value = _number(info.get(key))
            if value is not None and value <= 0:
                conflicts.append(f"{name} is nonpositive; review the financial condition before taking a positive price-model signal.")
    margin = result["evidence"].get("margin_of_safety")
    if (valuation or {}).get("status") == "available" and margin is not None and margin <= -.20 + 1e-12:
        conflicts.append("The quote is at least 20% above your base cashflow scenario; the valuation conflicts with the positive price-model signal.")
    if risk is None or not 0 <= risk <= 100:
        missing.append("Historical risk evidence is unavailable.")
    if missing:
        verdict, title = "INSUFFICIENT_DATA", "No reliable ML buy signal"
        summary = "The learned model has not established a usable signal for this snapshot."
        reasons = list(ml_result.get("reasons") or []) + missing
        conditions = ["Use ML lab to inspect data coverage, chronological test results and failed qualification checks.",
                      "More history or future out-of-sample observations may change the evidence; a model does not have to produce a buy call."]
    elif risk < 25:
        verdict, title = "DO_NOT_BUY", "No — historical risk vetoes a purchase"
        summary = "The risk check overrides the learned signal."
        reasons = [f"Historical risk score {risk:.1f}/100 is below the 25/100 minimum; higher scores represent lower measured risk."]
        conditions = ["Reassess if realized volatility, drawdowns and tail losses materially improve."]
    elif signal == "AVOID_SUPPORT":
        verdict, title = "DO_NOT_BUY", f"No — ML does not support buying for this {horizon}-session target"
        summary = "The qualified classifier assigns a low probability to the defined profitable-outperformance outcome."
        reasons = [f"Estimated target probability is {probability:.1%}, at or below the 40% avoidance threshold.",
                   "This predicts the defined outcome, not a certain loss or an instruction to sell an existing holding."]
        conditions = ["Reassess when new price observations change the model signal or the test evidence."]
    elif signal == "BUY_SUPPORT" and conflicts:
        verdict, title = "WAIT", "Wait — ML and financial evidence disagree"
        summary = "The model supports the selected return target, but conflicting financial evidence needs review."
        reasons = [f"Estimated target probability is {probability:.1%}."] + conflicts
        conditions = ["Review the stated financial conflicts and cashflow assumptions before considering the positive signal."]
    elif signal == "BUY_SUPPORT":
        verdict, title = "BUY", f"Yes — ML buy candidate for a {horizon}-session horizon"
        summary = "The selected classifier passed the historical test requirements and currently supports the defined return target."
        reasons = [f"Estimated target probability is {probability:.1%}, meeting the 60% positive-signal threshold.",
                   str(ml_result.get("target_description") or "The target includes the selected transaction-cost assumption."),
                   "Historical test qualification and risk checks passed; this does not establish future performance."]
        conditions = ["Reassess as new data arrive, model test performance changes, or the financial context deteriorates."]
    else:
        verdict, title = "WAIT", "Wait — ML evidence is mixed"
        summary = "The qualified model's probability is between the positive and negative decision thresholds."
        reasons = [f"Estimated target probability is {probability:.1%}; the model needs at least 60% for Buy and at most 40% for Do not buy."]
        conditions = ["New observations may move the model probability outside the 40%–60% wait range."]
    result.update({"verdict": verdict, "title": ("DEMO · " if is_demo else "") + title,
        "summary": ("Synthetic example only. " if is_demo else "") + summary,
        "reasons": list(dict.fromkeys(reasons)), "conditions_to_improve": conditions,
        "checks": checks, "actionable": not is_demo and verdict != "INSUFFICIENT_DATA",
        "thresholds": {"buy_probability": .60, "avoid_probability": .40, "minimum_risk_score": 25,
                       "valuation_conflict_margin": -.20, "maximum_price_age_days": 7},
        "methodology": "Primary signal from a supervised, chronologically evaluated classifier. Current factor/valuation rules are separate context and risk checks, not ML features or claimed training labels. Forecast-lab price fits and news AI do not vote. The target horizon, costs and benchmark define what probability means; personal suitability is unknown."})
    return result
