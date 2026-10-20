"""Deterministic, auditable stock research models; no network or UI dependencies.

Public functions return JSON-compatible dictionaries (missing numbers are None).
Rates, returns and coverage use fractions, scores use 0..100, prices use the
security's quote currency. Input history must contain dated, adjusted daily
Close prices (Adj Close takes precedence when both columns are supplied).

``analyze_stock`` returns score/factors/metric evidence, technical statistics,
valuation scenarios, warnings and methodology. ``forecast_prices`` returns an
expanding-window, horizon-matched comparison and bootstrap scenario bands.
``dcf_valuation`` returns assumptions, bear/base/bull and sensitivity records.
These are research heuristics and scenarios, not trained investment ratings.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS = 252
FACTOR_WEIGHTS = {
    "value": 0.20, "quality": 0.20, "growth": 0.15,
    "momentum": 0.15, "financial_strength": 0.15, "risk": 0.15,
}
SOURCES = [
    {"title": "Damodaran: valuation and cash-flow models", "url":
     "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/val.html"},
    {"title": "Damodaran: characteristics of financial service firms", "url":
     "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/littlebook/financialsvccompanies.htm"},
    {"title": "scikit-learn: time-series validation", "url":
     "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html"},
]


def _number(value: Any) -> float | None:
    """Reject booleans, NaN and infinity rather than assigning neutral values."""
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _positive(value: Any) -> float | None:
    result = _number(value)
    return result if result is not None and result > 0 else None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    top, bottom = _number(numerator), _positive(denominator)
    return _number(top / bottom) if top is not None and bottom is not None else None


def _plain(value: Any) -> Any:
    """Final serialization boundary: allow json.dumps(..., allow_nan=False)."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _number(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _close_series(history: pd.DataFrame | None) -> tuple[pd.Series, list[str]]:
    """Sort dates, keep last duplicate session, discard invalid prices, never fill."""
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]), name="Close")
    if history is None or not isinstance(history, pd.DataFrame) or history.empty:
        return empty, ["No usable adjusted price history was supplied."]
    column = "Adj Close" if "Adj Close" in history.columns else "Close"
    if column not in history.columns:
        return empty, ["Price history has no Close or Adj Close column."]
    values = history[column]
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            return empty, ["Ambiguous multi-security price history; select one security."]
        values = values.iloc[:, 0]
    if pd.api.types.is_numeric_dtype(history.index.dtype):
        return empty, ["Price history needs calendar dates rather than numeric row labels."]
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(history.index, errors="coerce"))
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        dates = dates.normalize()
        prices = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError):
        return empty, ["Price dates or Close values could not be read."]
    valid = ~dates.isna() & np.isfinite(prices) & (prices > 0)
    removed = int(len(prices) - valid.sum())
    series = pd.Series(prices[valid], index=dates[valid], name="Close")
    duplicates = int(series.index.duplicated(keep="last").sum())
    series = series[~series.index.duplicated(keep="last")].sort_index()
    warnings = []
    if removed:
        warnings.append(f"Excluded {removed} rows with invalid dates or nonpositive/nonfinite prices; no values were filled.")
    if duplicates:
        warnings.append(f"Kept the last valid price for {duplicates} duplicate trading-session rows.")
    if len(series) > 2 and np.median(np.diff(series.index.asi8)) / 8.64e13 > 4:
        warnings.append("History does not appear to be daily; daily annualization and forecast horizons may be inappropriate.")
    if len(series) < 63:
        warnings.append("Fewer than 63 price observations: risk estimates are unavailable or unstable.")
    return series, warnings


def _financial_company(info: dict) -> bool:
    sector = str(info.get("sector") or "").lower()
    industry = str(info.get("industry") or "").lower()
    return (sector in {"financial services", "financials", "financial"}
            or any(word in industry for word in ("banks", "banking", "insurance", "credit services", "capital markets", "mortgage reit")))


def _equity(info: dict) -> bool:
    return str(info.get("quoteType") or "EQUITY").upper() == "EQUITY"


def _rsi(prices: np.ndarray, period: int = 14) -> float | None:
    if len(prices) <= period:
        return None
    changes = np.diff(prices)
    gains, losses = np.maximum(changes, 0), np.maximum(-changes, 0)
    avg_gain, avg_loss = gains[:period].mean(), losses[:period].mean()
    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = ((period - 1) * avg_gain + gain) / period
        avg_loss = ((period - 1) * avg_loss + loss) / period
    if avg_loss == 0:
        return 50.0 if avg_gain == 0 else 100.0
    return float(100 - 100 / (1 + avg_gain / avg_loss))


def _technical(close: pd.Series, benchmark: pd.DataFrame | None,
               risk_free_rate: float) -> dict:
    prices = close.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    count = len(close)
    data = {
        "as_of": close.index[-1].date().isoformat() if count else None,
        "observations": count, "latest_close": float(prices[-1]) if count else None,
        "moving_averages": {f"ma{window}": float(prices[-window:].mean()) if count >= window else None
                            for window in (20, 50, 200)},
        "rsi14": _rsi(prices),
        "annualized_volatility": None, "downside_volatility": None,
        "max_drawdown": None, "historical_var95": None, "historical_cvar95": None,
        "sharpe": None, "sortino": None, "beta": None,
        "benchmark_observations": 0, "annualization_sessions": TRADING_DAYS,
        "risk_window_observations": min(count, 756),
        "returns": {label: _number(prices[-1] / prices[-days - 1] - 1) if count > days else None
                    for label, days in (("one_month", 21), ("three_month", 63), ("six_month", 126), ("one_year", 252))},
    }
    # Consistent trailing risk window; momentum and moving averages use specified lookbacks.
    recent = close.iloc[-756:]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        r = recent.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(recent):
        data["max_drawdown"] = float((recent / recent.cummax() - 1).min())
    if len(r) >= 62:
        vol = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))
        downside = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(TRADING_DAYS))
        quantile = float(np.quantile(r, .05))
        tail = r[r <= quantile]
        data.update(annualized_volatility=vol, downside_volatility=downside,
                    historical_var95=max(0., -quantile),
                    historical_cvar95=max(0., -float(tail.mean())))
        rf = _number(risk_free_rate)
        if rf is not None and rf > -1:
            daily_rf = math.expm1(math.log1p(rf) / TRADING_DAYS)
            excess = float(np.mean(r) - daily_rf) * TRADING_DAYS
            data["sharpe"] = excess / vol if vol > 1e-12 else None
            # Sortino's target is the supplied daily risk-free return.
            target_downside = float(np.sqrt(np.mean(np.minimum(r - daily_rf, 0) ** 2)) * np.sqrt(TRADING_DAYS))
            data["sortino"] = excess / target_downside if target_downside > 1e-12 else None
    if benchmark is not None and count:
        other, _ = _close_series(benchmark)
        # Align prices FIRST: returns must span the same dates for both assets.
        aligned = pd.concat([recent.rename("stock"), other.rename("market")], axis=1, join="inner").dropna()
        with np.errstate(over="ignore", invalid="ignore"):
            paired = aligned.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
        data["benchmark_observations"] = len(paired)
        if len(paired) >= 60:
            variance = float(paired["market"].var(ddof=1))
            if variance > 1e-15:
                data["beta"] = float(paired["stock"].cov(paired["market"]) / variance)
    return _plain(data)


def _metric(key: str, label: str, value: Any, low: float, high: float,
            *, weight: float = 1., higher: bool = True,
            unit: str = "ratio", reason: str | None = None) -> dict:
    value = _number(value)
    score = None if value is None else float(np.clip((value - low) / (high - low), 0, 1) * 100)
    if score is not None and not higher:
        score = 100 - score
    return {"key": key, "label": label, "value": value, "score": score,
            "weight": weight, "available": score is not None,
            "reason": None if score is not None else (reason or "Required data is missing or not meaningful."),
            "unit": unit, "scale_low": low, "scale_high": high, "higher_is_better": higher}


def _factor(metrics: list[dict], weight: float) -> dict:
    total = sum(metric["weight"] for metric in metrics)
    available = sum(metric["weight"] for metric in metrics if metric["available"])
    score = (sum(metric["score"] * metric["weight"] for metric in metrics if metric["available"]) / available
             if available else None)
    return {"score": score, "coverage": available / total if total else 0.,
            "weight": weight, "available_metrics": sum(metric["available"] for metric in metrics),
            "total_metrics": len(metrics), "metrics": metrics}


def analyze_stock(info: dict, history: pd.DataFrame, benchmark: pd.DataFrame | None = None,
                  *, risk_free_rate: float = .04) -> dict:
    """Build six continuous absolute heuristic factors and disclosed technicals.

    Overall uses factor_weight * metric_coverage and renormalizes only available
    evidence. Coverage below 50%, fewer than three factors, or fewer than two
    fundamental factors causes abstention. A missing value never becomes 50.
    Financial firms exclude generic EV/EBITDA, FCF and liquidity/leverage ratios.
    """
    info = info if isinstance(info, dict) else {}
    close, warnings = _close_series(history)
    technical = _technical(close, benchmark, risk_free_rate)
    financial = _financial_company(info)
    equity = _equity(info)
    # Preserve point-in-time provider values; never merge future financials into backtests.
    current = _positive(info.get("currentPrice")) or _positive(info.get("regularMarketPrice"))
    latest = technical["latest_close"]
    price = current or latest
    market_cap = _positive(info.get("marketCap"))
    shares = _positive(info.get("sharesOutstanding"))
    if market_cap is None and price is not None and shares is not None:
        market_cap = _number(price * shares)
    currency = str(info.get("currency") or "") or None
    financial_currency = str(info.get("financialCurrency") or "") or None
    currency_match = not (currency and financial_currency and currency != financial_currency)
    eps = _number(info.get("trailingEps"))
    pe = _positive(info.get("trailingPE"))
    earnings_yield = _ratio(eps, price) if eps is not None and currency_match else (_ratio(1., pe) if pe else None)
    fcf = _number(info.get("freeCashflow"))
    revenue = _number(info.get("totalRevenue"))
    debt, cash = _number(info.get("totalDebt")), _number(info.get("totalCash"))
    debt = debt if debt is not None and debt >= 0 else None
    cash = cash if cash is not None and cash >= 0 else None
    ebitda = _positive(info.get("ebitda"))
    net_debt = debt - cash if debt is not None and cash is not None else None
    value_metrics = [_metric("earnings_yield", "Trailing earnings yield", earnings_yield, -.02, .10, weight=2., unit="fraction")]
    if financial:
        value_metrics.append(_metric("price_to_book", "Price / book (positive book equity only)",
                                     _positive(info.get("priceToBook")), .5, 3., higher=False, unit="multiple"))
    else:
        value_metrics.extend([
            _metric("fcf_yield", "Reported free-cash-flow yield", _ratio(fcf, market_cap) if currency_match else None,
                    -.03, .10, weight=2., unit="fraction"),
            _metric("ev_ebitda", "Enterprise value / EBITDA", _positive(info.get("enterpriseToEbitda")),
                    5., 25., higher=False, unit="multiple"),
        ])
    reported_equity = _number(info.get("totalStockholderEquity"))
    if reported_equity is None:
        reported_equity = _number(info.get("stockholdersEquity"))
    book_value = _number(info.get("bookValue"))
    nonpositive_equity = ((reported_equity is not None and reported_equity <= 0)
                          or (book_value is not None and book_value <= 0))
    if nonpositive_equity:
        warnings.append("Return on equity is excluded because reported book equity is nonpositive.")
    quality_metrics = [
        _metric("roe", "Return on equity", None if nonpositive_equity else info.get("returnOnEquity"),
                -.05, .20 if financial else .30, unit="fraction", reason="Return on equity is not meaningful with nonpositive book equity." if nonpositive_equity else None),
        _metric("roa", "Return on assets", info.get("returnOnAssets"), -.005 if financial else -.02,
                .02 if financial else .15, unit="fraction"),
        _metric("margin", "Net margin" if financial else "Operating margin",
                info.get("profitMargins" if financial else "operatingMargins"), -.05 if financial else -.10,
                .30 if financial else .35, unit="fraction"),
    ]
    if not financial:
        quality_metrics.append(_metric("fcf_margin", "Reported free-cash-flow margin", _ratio(fcf, revenue), -.10, .25, unit="fraction"))
    net_income = _number(info.get("netIncomeToCommon"))
    negative_earnings = ((eps is not None and eps <= 0) or (net_income is not None and net_income <= 0))
    growth_metrics = [
        _metric("revenue_growth", "Reported revenue growth", info.get("revenueGrowth"), -.10, .25, unit="fraction"),
        _metric("earnings_growth", "Reported earnings growth", None if negative_earnings else info.get("earningsGrowth"),
                -.20, .30, unit="fraction", reason="Nonpositive current earnings make percentage earnings growth ambiguous; historical statements are needed." if negative_earnings else None),
    ]
    ma200 = technical["moving_averages"]["ma200"]
    momentum_metrics = [
        _metric("return_3m", "63-session total price return", technical["returns"]["three_month"], -.20, .30, unit="fraction"),
        _metric("return_6m", "126-session total price return", technical["returns"]["six_month"], -.25, .40, unit="fraction"),
        _metric("distance_ma200", "Price relative to 200-session average", (_ratio(latest, ma200) - 1) if _ratio(latest, ma200) is not None else None,
                -.20, .30, unit="fraction"),
    ]
    if financial:
        book_equity = _number(info.get("totalStockholderEquity"))
        if book_equity is None:
            book_equity = _number(info.get("stockholdersEquity"))
        strength_metrics = [_metric("equity_assets", "Accounting equity / assets (not regulatory capital)",
                                    _ratio(book_equity, info.get("totalAssets")), .03, .15, unit="fraction")]
        warnings.append("Financial-company handling excludes generic enterprise, cash-flow and debt/liquidity ratios. Regulatory capital, asset quality and loss reserves are not modeled.")
    else:
        strength_metrics = [
            _metric("current_ratio", "Current ratio", _positive(info.get("currentRatio")), .5, 2.5, unit="multiple"),
            _metric("debt_ebitda", "Total debt / positive EBITDA", _ratio(debt, ebitda), 0., 5., higher=False, unit="multiple"),
            _metric("net_debt_market_cap", "Net debt / market capitalization", _ratio(net_debt, market_cap) if currency_match else None,
                    -.25, 1., higher=False, unit="fraction"),
        ]
    risk_metrics = [
        _metric("volatility", "Annualized realized volatility", technical["annualized_volatility"], .10, .70, higher=False, unit="fraction"),
        _metric("max_drawdown", "Trailing maximum drawdown magnitude", -technical["max_drawdown"] if len(close) >= 63 else None,
                .05, .60, higher=False, unit="fraction"),
        _metric("cvar95", "Historical daily 95% tail loss", technical["historical_cvar95"], .01, .08, higher=False, unit="fraction"),
    ]
    metrics_by_factor = {"value": value_metrics, "quality": quality_metrics, "growth": growth_metrics,
                         "momentum": momentum_metrics, "financial_strength": strength_metrics, "risk": risk_metrics}
    if not equity:
        warnings.append("Fundamental equity scores and cash-flow valuation are unavailable for this non-equity instrument.")
        for name in ("value", "quality", "growth", "financial_strength"):
            for metric in metrics_by_factor[name]:
                metric.update(value=None, score=None, available=False, reason="This model is for operating-company equities.")
    factors = {name: _factor(metrics, FACTOR_WEIGHTS[name]) for name, metrics in metrics_by_factor.items()}
    coverage = sum(factor["weight"] * factor["coverage"] for factor in factors.values())
    count_factors = sum(factor["score"] is not None for factor in factors.values())
    fundamentals = sum(factors[name]["score"] is not None for name in ("value", "quality", "growth", "financial_strength"))
    enough = coverage >= .50 and count_factors >= 3 and fundamentals >= 2
    overall = (sum(factor["weight"] * factor["coverage"] * factor["score"] for factor in factors.values() if factor["score"] is not None)
               / coverage if enough else None)
    if not enough:
        warnings.append("Insufficient weighted evidence for an overall score; review available metrics individually.")
    if len(close) < 253:
        warnings.append("Less than a full 252-return trading year; long-horizon momentum or risk context is incomplete.")
    if not currency_match:
        warnings.append("Quote and financial-statement currencies differ; cross-currency yields, net-debt ratios and valuation were excluded.")
    if benchmark is not None and technical["beta"] is None:
        warnings.append("Beta requires at least 60 aligned return observations and a benchmark with nonzero variance.")
    checked_rf = _number(risk_free_rate)
    if checked_rf is None or checked_rf <= -1:
        warnings.append("Invalid risk-free rate; risk-adjusted return statistics were omitted.")
    return _plain({
        "schema_version": 1, "symbol": str(info.get("symbol") or ""),
        "sector": info.get("sector"), "industry": info.get("industry"), "currency": currency,
        "financial_company": financial,
        "score": {"overall": overall, "coverage": coverage, "status": "available" if enough else "insufficient_data",
                  "factors": factors, "interpretation": "Absolute research heuristic; not a trained prediction, peer percentile, probability, or buy/sell recommendation."},
        "technical": technical, "valuation": dcf_valuation(info),
        "warnings": list(dict.fromkeys(warnings)),
        "methodology": {
            "factor_scales": "Continuous linear scales clipped to 0..100; subjective fixed anchors shown for every metric. No fabricated sector averages.",
            "coverage": "Sum of factor weight × available metric weight fraction; available evidence is renormalized. Overall requires >=50% coverage, >=3 factors and >=2 fundamental factors.",
            "risk": "Trailing up to 756 sessions; sample standard deviation, 252-session annualization; downside semideviation uses zero-return target. VaR/CVaR are nonnegative historical one-session losses at 95%, not loss limits. Maximum drawdown is negative.",
            "prices": "Adjusted daily prices provided by the data service; returns include provider dividend/split adjustments. Missing rows are excluded, never filled.",
            "beta": "Sample covariance / sample benchmark variance after joining price dates, then computing common-interval returns.",
            "limitations": "ROE can be distorted by leverage/buybacks; earnings growth by small/negative comparison periods. Different sectors warrant different interpretation. Current fundamentals are not backtested.",
            "risk_free_rate": risk_free_rate, "sources": SOURCES,
        },
    })


def _dcf_case(fcf: float, shares: float, price: float | None, growth: float,
              rate: float, terminal: float, years: int) -> dict:
    if rate <= terminal or rate <= -1 or terminal <= -1 or growth <= -1:
        return {"fair_value_per_share": None, "reason": "Required equity return must exceed terminal growth; all growth/return rates must exceed -100%."}
    try:
        cashflows = [fcf * (1 + growth) ** year for year in range(1, years + 1)]
        pv_cashflows = sum(cashflow / (1 + rate) ** year for year, cashflow in enumerate(cashflows, 1))
        terminal_value = cashflows[-1] * (1 + terminal) / (rate - terminal)
        pv_terminal = terminal_value / (1 + rate) ** years
        equity_value = pv_cashflows + pv_terminal
        per_share = _positive(equity_value / shares)
    except (OverflowError, ZeroDivisionError):
        per_share = None
    if per_share is None:
        return {"fair_value_per_share": None, "reason": "Inputs produced a nonfinite or nonpositive scenario value."}
    return {"fair_value_per_share": per_share, "equity_value": equity_value,
            "present_value_cashflows": pv_cashflows, "present_value_terminal": pv_terminal,
            "terminal_value_share": pv_terminal / equity_value,
            "projected_cashflows": cashflows, "upside": per_share / price - 1 if price else None,
            "reason": None}


def dcf_valuation(info: dict, growth: float = .05, discount_rate: float = .10,
                  terminal_growth: float = .025, years: int = 5) -> dict:
    """Illustrative equity-cash-flow proxy DCF, assuming zero net borrowing.

    Yahoo freeCashflow is CFO minus capex, after interest but before net
    borrowing. Treating it as FCFE requires the explicit zero-net-borrowing
    assumption. Discount at a *required equity return*, not WACC. Do NOT subtract
    debt or add cash; doing so would mix enterprise and equity cash-flow bases.
    Financial firms, nonpositive cashflow and currency mismatches abstain.
    """
    info = info if isinstance(info, dict) else {}
    fcf, shares = _positive(info.get("freeCashflow")), _positive(info.get("sharesOutstanding"))
    rate, g, terminal = _number(discount_rate), _number(growth), _number(terminal_growth)
    valid_years = isinstance(years, (int, np.integer)) and not isinstance(years, bool) and 1 <= years <= 20
    currency, financial_currency = info.get("currency"), info.get("financialCurrency")
    result = {
        "status": "unavailable", "model": "Equity cash-flow proxy scenarios (zero net borrowing)",
        "currency": currency, "fair_value_per_share": None, "scenarios": [], "sensitivity": [], "reasons": [],
        "assumptions": {"growth": g, "discount_rate": rate, "terminal_growth": terminal, "years": int(years) if valid_years else None,
                        "reported_free_cashflow": fcf, "shares_outstanding": shares, "net_borrowing": 0.,
                        "cashflow_basis": "Reported CFO minus capital expenditure, used as an equity-cash-flow proxy assuming zero net borrowing.",
                        "discount_rate_basis": "Required annual equity return; fixed user assumption, not an estimated WACC.",
                        "debt_cash_treatment": "No debt deduction or cash addition: this uses an equity-cash-flow basis.",
                        "reported_total_debt": _number(info.get("totalDebt")), "reported_total_cash": _number(info.get("totalCash"))},
        "warnings": ["Scenario illustration only: reported FCF is not a complete FCFE calculation; net borrowing, dilution and detailed reinvestment are not forecast. Not used in the overall score.",
                     "Constant growth and a perpetual terminal value are sensitive assumptions; terminal value can dominate the result."],
    }
    reasons = result["reasons"]
    if not _equity(info):
        reasons.append("The cash-flow model applies to operating-company equities only.")
    if _financial_company(info):
        reasons.append("Generic cash-flow DCF is inappropriate for financial service firms; capital adequacy and equity/dividend models are needed.")
    if fcf is None:
        reasons.append("A finite positive reported free cash flow is required; loss-making cash flow is not assigned a zero value.")
    if shares is None:
        reasons.append("A finite positive shares-outstanding count is required (cash flow and shares must use full units).")
    if currency and financial_currency and str(currency) != str(financial_currency):
        reasons.append("Quote currency differs from financial-statement currency; convert units before valuation.")
    if not currency or not financial_currency:
        result["warnings"].append("Both quote and financial currencies were not supplied; currency consistency could not be verified.")
    if not valid_years or any(value is None for value in (rate, g, terminal)):
        reasons.append("Use finite rates and an integer projection length from 1 to 20 years.")
    elif rate <= terminal or min(rate, g, terminal) <= -1:
        reasons.append("Required equity return must exceed terminal growth and all rates must exceed -100%.")
    if reasons:
        return _plain(result)
    price = _positive(info.get("currentPrice")) or _positive(info.get("regularMarketPrice"))
    cases = (("Bear", max(-.95, g - .03), rate + .02, max(-.95, terminal - .005)),
             ("Base", g, rate, terminal),
             ("Bull", g + .03, rate - .02, terminal + .005))
    for name, case_growth, case_rate, case_terminal in cases:
        case = {"name": name, "growth": case_growth, "discount_rate": case_rate, "terminal_growth": case_terminal}
        case.update(_dcf_case(fcf, shares, price, case_growth, case_rate, case_terminal, int(years)))
        result["scenarios"].append(case)
    base = result["scenarios"][1]
    result["fair_value_per_share"] = base["fair_value_per_share"]
    for case_rate in (rate - .02, rate, rate + .02):
        for case_terminal in (terminal - .005, terminal, terminal + .005):
            sensitivity = {"discount_rate": case_rate, "terminal_growth": case_terminal}
            value = _dcf_case(fcf, shares, price, g, case_rate, case_terminal, int(years))
            sensitivity.update(fair_value_per_share=value["fair_value_per_share"], reason=value["reason"])
            result["sensitivity"].append(sensitivity)
    result["status"] = "available" if base["fair_value_per_share"] is not None else "unavailable"
    if result["status"] == "unavailable":
        reasons.append(base["reason"])
    return _plain(result)


MODEL_NAMES = ("random_walk", "historical_drift", "ewma_drift", "ridge_ar5")


def _fit_model(log_prices: np.ndarray, name: str) -> dict:
    """Fit on a training prefix only; AR standardization never sees holdout."""
    returns = np.diff(log_prices)
    if name == "random_walk":
        return {"name": name, "drift": 0.}
    if name == "historical_drift":
        return {"name": name, "drift": float(returns.mean())}
    if name == "ewma_drift":
        # Fixed half-life; no tuning on holdout.
        weights = np.exp2(-np.arange(len(returns) - 1, -1, -1) / 20.)
        return {"name": name, "drift": float(np.average(returns, weights=weights))}
    lags = 5
    features = np.array([returns[index-lags:index][::-1] for index in range(lags, len(returns))])
    targets = returns[lags:]
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale < 1e-10] = 1.
    standardized = (features - mean) / scale
    target_mean = float(targets.mean())
    # Ridge penalty fixed in advance. Intercept remains unpenalized by centering y.
    penalty = 10. * np.eye(lags)
    coefficients = np.linalg.solve(standardized.T @ standardized + penalty,
                                   standardized.T @ (targets - target_mean))
    return {"name": name, "mean": mean, "scale": scale, "coefficients": coefficients,
            "intercept": target_mean, "return_limit": max(.02, float(np.std(returns)) * 5.)}


def _next_return(model: dict, previous: list[float] | np.ndarray) -> float:
    if model["name"] != "ridge_ar5":
        return model["drift"]
    features = np.asarray(previous[-5:][::-1])
    prediction = model["intercept"] + ((features - model["mean"]) / model["scale"]) @ model["coefficients"]
    # Fixed training-derived stability guard for recursive AR forecasts.
    return float(np.clip(prediction, -model["return_limit"], model["return_limit"]))


def _forecast_log(log_prices: np.ndarray, name: str, horizon: int) -> np.ndarray:
    model = _fit_model(log_prices, name)
    previous = list(np.diff(log_prices))
    predictions = []
    level = float(log_prices[-1])
    for _ in range(horizon):
        prediction = _next_return(model, previous)
        previous.append(prediction)
        level += prediction
        predictions.append(level)
    return np.asarray(predictions)


def _training_residuals(log_prices: np.ndarray, model: dict) -> np.ndarray:
    returns = np.diff(log_prices)
    if model["name"] != "ridge_ar5":
        residuals = returns - model["drift"]
    else:
        residuals = np.array([returns[index] - _next_return(model, returns[:index])
                              for index in range(5, len(returns))])
    # Bands describe shocks about the selected model, not a hidden extra drift.
    return residuals - residuals.mean()


def forecast_prices(history: pd.DataFrame, horizon: int = 21) -> dict:
    """Compare fixed models on expanding prefixes at the requested session horizon.

    Holdout outcomes are never used to fit their own forecasts or preprocessing.
    Model selection uses this holdout, so selected-model performance is NOT an
    independent final test. Unless a candidate improves RMSE by >=2%, retain the
    random-walk baseline. Bands use 2,000 seeded, 5-session moving-block residual
    bootstrap paths; they are uncalibrated scenarios, not promised probabilities.
    """
    close, warnings = _close_series(history)
    valid_horizon = isinstance(horizon, (int, np.integer)) and not isinstance(horizon, bool) and 1 <= horizon <= 126
    result = {"status": "insufficient_data", "horizon": int(horizon) if valid_horizon else None,
              "observations": len(close), "current_price": float(close.iloc[-1]) if len(close) else None,
              "as_of": close.index[-1].date().isoformat() if len(close) else None,
              "selected_model": None, "forecast": [], "backtest": {"models": []}, "warnings": warnings,
              "methodology": {"target": "Adjusted daily log prices; forecast horizon in observed trading sessions, not calendar days.",
                              "models": "Random walk, full-history mean log-return drift, fixed 20-session-half-life EWMA drift, ridge AR(5) on log returns (alpha=10; training-only standardization and stability bounds).",
                              "intervals": "2,000 seeded 5-session moving-block bootstraps of centered training residuals. 80% and 95% empirical scenario bands, conditional on history/model, uncalibrated; exclude parameter uncertainty and structural breaks.",
                              "validation": "Expanding training prefixes predict the same requested horizon. All models use identical chronological origins; endpoints can overlap. Model selection reuses this comparison and has no independent final test.",
                              "sources": [SOURCES[2]]}}
    if not valid_horizon:
        result["warnings"].append("Horizon must be an integer from 1 to 126 trading sessions.")
        return _plain(result)
    horizon = int(horizon)
    minimum = max(100, 60 + horizon + 4)
    if len(close) < minimum:
        result["warnings"].append(f"At least {minimum} adjusted daily closes are required for this horizon and a usable chronological comparison.")
        return _plain(result)
    if np.median(np.diff(close.index.asi8)) / 8.64e13 > 4:
        result["warnings"].append("Forecasts require daily history; no monthly/weekly-to-daily conversion was made.")
        return _plain(result)
    prices = close.to_numpy(dtype=float)
    log_prices = np.log(prices)
    if np.max(np.abs(np.diff(log_prices))) > math.log(10):
        result["warnings"].append("An observed session changed by more than a factor of ten; verify adjustments/data quality before forecasting.")
        return _plain(result)
    # Never let long horizons consume every holdout origin. Initial fit >=60 closes.
    first = max(60, min(int(len(close) * .70), len(close) - horizon - 4))
    last = len(close) - horizon
    stride = max(1, min(5, horizon, (last - first) // 4))
    origins = list(range(first, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    origins = origins[-60:]
    actual = np.array([prices[origin + horizon - 1] for origin in origins])
    starting = np.array([prices[origin - 1] for origin in origins])
    for name in MODEL_NAMES:
        with np.errstate(over="ignore", under="ignore"):
            predictions = np.exp(np.array([_forecast_log(log_prices[:origin], name, horizon)[-1] for origin in origins]))
        if not np.isfinite(predictions).all() or np.any(predictions <= 0):
            result["warnings"].append("Comparison forecasts overflowed or underflowed; verify price units and horizon.")
            return _plain(result)
        errors = predictions - actual
        directional = (np.abs(predictions / starting - 1) > 1e-10
                       if name != "random_walk" else np.zeros(len(starting), dtype=bool))
        directional_accuracy = (float(np.mean(
            np.sign(predictions[directional] - starting[directional])
            == np.sign(actual[directional] - starting[directional]))) if directional.any() else None)
        error_scale = float(np.max(np.abs(errors)))
        rmse = error_scale * float(np.sqrt(np.mean((errors / error_scale) ** 2))) if error_scale else 0.
        result["backtest"]["models"].append({
            "name": name, "mae": float(np.mean(np.abs(errors))),
            "rmse": rmse,
            "mape": float(np.mean(np.abs(errors) / actual)),
            # A flat random walk makes no directional call; rounding a log/exp
            # roundtrip must not manufacture one or count it as a failed call.
            "directional_accuracy": directional_accuracy,
            "directional_coverage": float(directional.mean()),
            "n_predictions": len(origins), "horizon": horizon,
        })
    baseline = result["backtest"]["models"][0]
    candidate = min(result["backtest"]["models"], key=lambda row: row["rmse"])
    improvement = (1 - candidate["rmse"] / baseline["rmse"]) if baseline["rmse"] > 1e-12 else 0.
    selected = candidate["name"] if improvement >= .02 else "random_walk"
    result["selected_model"] = selected
    result["backtest"].update({
        "horizon": horizon, "n_origins": len(origins), "origin_stride": stride,
        "first_train_observations": origins[0],
        "first_origin": close.index[origins[0]-1].date().isoformat(),
        "last_origin": close.index[origins[-1]-1].date().isoformat(),
        "first_target": close.index[origins[0]+horizon-1].date().isoformat(),
        "last_target": close.index[origins[-1]+horizon-1].date().isoformat(),
        "rmse_improvement_over_baseline": improvement if selected != "random_walk" else 0.,
        "selection_rule": "Choose lowest holdout RMSE only with >=2% improvement over random walk; otherwise retain random walk.",
        "overlapping_targets": stride < horizon,
        "independent_final_test": False,
    })
    model = _fit_model(log_prices, selected)
    residuals = _training_residuals(log_prices, model)
    generator = np.random.default_rng(1729)
    n_paths, block = 2000, min(5, len(residuals))
    blocks_needed = math.ceil(horizon / block)
    starts = generator.integers(0, len(residuals), size=(n_paths, blocks_needed))
    shock_indices = (starts[:, :, None] + np.arange(block)) % len(residuals)
    shocks = residuals[shock_indices].reshape(n_paths, -1)[:, :horizon]
    paths = np.zeros((n_paths, horizon))
    levels = np.full(n_paths, log_prices[-1])
    # Vectorized recursive AR so each simulated path responds to its own shocks.
    previous = np.tile(np.diff(log_prices)[-5:], (n_paths, 1))
    for step in range(horizon):
        if selected == "ridge_ar5":
            features = previous[:, -5:][:, ::-1]
            prediction = model["intercept"] + ((features - model["mean"]) / model["scale"]) @ model["coefficients"]
            prediction = np.clip(prediction, -model["return_limit"], model["return_limit"])
        else:
            prediction = model["drift"]
        next_return = prediction + shocks[:, step]
        levels += next_return
        paths[:, step] = levels
        previous = np.column_stack([previous[:, -4:], next_return])
    with np.errstate(over="ignore", under="ignore"):
        simulated = np.exp(paths)
        central = np.exp(_forecast_log(log_prices, selected, horizon))
    if not np.isfinite(simulated).all() or not np.isfinite(central).all() or np.any(simulated <= 0):
        result["warnings"].append("Scenario simulation overflowed or underflowed; check history and horizon.")
        return _plain(result)
    quantiles = np.quantile(simulated, [.025, .10, .50, .90, .975], axis=0)
    for step in range(horizon):
        result["forecast"].append({"step": step + 1, "point_forecast": float(central[step]),
                                    "median": float(quantiles[2, step]),
                                    "lower_80": float(quantiles[1, step]), "upper_80": float(quantiles[3, step]),
                                    "lower_95": float(quantiles[0, step]), "upper_95": float(quantiles[4, step])})
    result["status"] = "available"
    if len(origins) < 20:
        result["warnings"].append("Fewer than 20 comparison origins; model ranking is especially unstable.")
    if stride < horizon:
        result["warnings"].append("Holdout target windows overlap; comparison errors are correlated and are not independent trials.")
    result["warnings"].append("Past holdout errors do not establish future accuracy. Selected-model results reuse the selection holdout; scenario-band coverage is not calibrated.")
    if len(close) < 504:
        result["warnings"].append("Less than two trading years of history; market-regime coverage is limited.")
    return _plain(result)
