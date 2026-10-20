"""Finsight: a portable, explainable stock research dashboard."""
import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finsight.analytics import analyze_stock, dcf_valuation, forecast_prices
from finsight.data import demo_market_data, fetch_market_data
from finsight.decision import machine_learning_decision
from finsight.ml import predict_buy_signal
from finsight.news import analyze_news, fetch_news, load_news_key, summarize_news
from finsight.report import export_report

st.set_page_config(page_title="Finsight · Stock research", page_icon="◈", layout="wide")
st.markdown("""
<style>
 .stApp {background: #f7f9fc;}
 [data-testid="stMainBlockContainer"] {padding-top: 2rem;}
 [data-testid="stSidebar"] {background: #eef3f8;}
 h1, h2, h3 {color: #132a43; letter-spacing: -.025em;}
 [data-testid="stMetric"] {background: white; border: 1px solid #e2e8f0;
   padding: 18px; border-radius: 12px;}
 [data-testid="stMetricLabel"] {color: #526278;}
 .stTabs [data-baseweb="tab-list"] {gap: 16px;}
 .stTabs [aria-selected="true"] {color: #087f8c;}
</style>
""", unsafe_allow_html=True)


def number(value, digits=2, suffix=""):
    try:
        value = float(value)
        return f"{value:,.{digits}f}{suffix}" if math.isfinite(value) else "Unavailable"
    except (ValueError, TypeError):
        return "Unavailable"


def percent(value, digits=1):
    return number(None if value is None else value * 100, digits, "%")


def compact(value):
    try:
        for magnitude, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(value) >= magnitude:
                return f"{value / magnitude:,.2f}{suffix}"
        return number(value)
    except TypeError:
        return "Unavailable"


def chart_style(fig, title=None, yaxis_title=None):
    fig.update_layout(title=title, template="plotly_white", height=410,
        margin=dict(l=16, r=16, t=48 if title else 20, b=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#34445a"),
        legend=dict(orientation="h", y=1.12), yaxis_title=yaxis_title, hovermode="x unified")
    return fig


def show_decision(decision, *, detailed=False):
    """Render one shared result in Overview, the final tab, and exports."""
    st.subheader("Should you buy this stock?")
    display = {"BUY": st.success, "WAIT": st.warning, "DO_NOT_BUY": st.error,
               "INSUFFICIENT_DATA": st.info}[decision["verdict"]]
    display(f"**{decision['title']}**")
    st.write(decision["summary"])
    for reason in decision["reasons"][:None if detailed else 3]:
        st.write("• " + reason)
    st.caption("The selected ML horizon defines this assessment. It does not assess your portfolio, personal goals or risk tolerance.")
    if decision["is_demo"]:
        st.caption(f"Synthetic snapshot assessed as of {decision['as_of'][:10]}.")
    if not detailed:
        st.caption("Open Should you buy? for the full reasons and decision rules.")
        return
    evidence = decision["evidence"]
    if evidence.get("ml_probability") is not None:
        cols = st.columns(3)
        cols[0].metric("Estimated target probability", percent(evidence["ml_probability"]))
        cols[1].metric("Prediction horizon", f"{evidence.get('ml_horizon_sessions')} sessions")
        cols[2].metric("Qualified test evidence", "Passed" if evidence.get("ml_eligible") else "Not established")
        st.caption(str(evidence.get("ml_target") or ""))
    cols = st.columns(3)
    cols[0].metric(f"Current market quote · {evidence.get('currency') or 'unknown currency'}", number(evidence.get("quote")))
    cols[1].metric("Base scenario value per share", number(evidence.get("base_fair_value")))
    cols[2].metric("Margin of safety", percent(evidence.get("margin_of_safety")))
    st.caption("Cashflow valuation is separate financial context. Margin of safety = 1 − quote / base scenario value; it is not the ML prediction or a promised return.")
    if decision["conditions_to_improve"]:
        st.markdown("**What would change this assessment?**")
        for condition in decision["conditions_to_improve"]:
            st.write("• " + condition)
    with st.expander("Check every decision rule"):
        rows = [{"Rule": check["label"], "Result": "Pass" if check["passed"] else "Not met" if check["passed"] is False else "Unavailable",
                 "Observed": str(check.get("value")), "Requirement": str(check.get("threshold"))} for check in decision["checks"]]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.json(decision["thresholds"])
    with st.expander("Valuation assumptions used for this assessment"):
        st.json(evidence.get("valuation_assumptions", {}))
    with st.expander("Separate fundamentals-only assessment"):
        context = decision.get("fundamental_assessment", {})
        st.write(context.get("title", "Unavailable"))
        for reason in context.get("reasons", []):
            st.write("• " + reason)
        st.caption("This comparison uses the earlier fixed factor/valuation rules. It is not a learned prediction.")
    st.caption("ML performance is measured on held-out history, not guaranteed for the future. Current financial context can veto or moderate a signal. News AI does not decide the result.")


@st.cache_data(ttl=900, show_spinner=False)
def market_data(symbol, benchmark):
    return fetch_market_data(symbol, benchmark_symbol=benchmark, period="max")


@st.cache_data(show_spinner=False, max_entries=24)
def model_forecast(history, horizon):
    return forecast_prices(history, horizon=horizon)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=12)
def learned_buy_models(history, benchmark, horizon, cost_bps):
    return predict_buy_signal(history, benchmark, horizon=horizon, cost_bps=cost_bps)


st.sidebar.title("◈ Finsight")
st.sidebar.caption("A clearer view of a business, its price, and its risks.")
mode = st.sidebar.radio("Data source", ["Demo", "Live market data"], key="source_mode")
st.sidebar.caption("Demo works offline. Live data requires an internet connection.")
if mode == "Demo":
    bundle = demo_market_data()
else:
    with st.sidebar.form("research"):
        symbol = st.text_input("Stock ticker", "MSFT", help="Yahoo symbols, for example AAPL, BRK-B, or TCS.NS.")
        benchmark = st.text_input("Benchmark ticker", "SPY", help="Choose a relevant market and currency for your company.")
        submitted = st.form_submit_button("Analyze company", width="stretch")
    if submitted:
        try:
            with st.spinner("Loading company data and adjusted price history…"):
                loaded = market_data(symbol.strip().upper(), benchmark.strip().upper())
            st.session_state["live_bundle"] = loaded
            st.session_state.pop("news_result", None)
        except ValueError as exc:
            st.sidebar.error(str(exc))
        except Exception:
            st.sidebar.error("Market data is unavailable. Check your connection and try again, or open Demo.")
    bundle = st.session_state.get("live_bundle")
    if st.sidebar.button("Refresh cached market data", width="stretch"):
        market_data.clear()
        st.sidebar.info("Cache cleared. Select Analyze company to fetch new data.")
st.sidebar.divider()
st.sidebar.markdown("**Reading the dashboard**")
st.sidebar.caption("Start with Overview, inspect the factors, then compare the valuation and forecast assumptions.")
st.sidebar.caption("Scores summarize available evidence. They are not probabilities of profit.")

st.title("Company research")
st.caption("FUNDAMENTALS  /  PRICE & RISK  /  SCENARIOS  /  NEWS")
if bundle is None:
    st.info("Enter a ticker and select Analyze company in the sidebar, or select Demo to explore without setup.")
    st.stop()
info, history = bundle["info"], bundle["history"]
benchmark_history = bundle.get("benchmark", pd.DataFrame())
analysis = analyze_stock(info, history, benchmark_history)
score, tech = analysis["score"], analysis["technical"]
currency = bundle.get("currency") or info.get("currency") or "Currency unknown"
symbol = bundle.get("symbol", "DEMO")
name = info.get("longName") or info.get("shortName") or symbol
st.header(f"{name} · {symbol}")
st.caption(f"{info.get('sector') or 'Sector unavailable'} · {info.get('industry') or 'Industry unavailable'} · {currency}")
if bundle.get("is_demo"):
    st.warning("DEMO · Synthetic company and prices for exploring the app. These are not real market observations.")
    st.caption(f"Synthetic history through {bundle.get('price_as_of') or 'unknown'}")
else:
    st.caption(f"History through {bundle.get('price_as_of') or 'unknown'} · Retrieved {bundle.get('fetched_at') or 'unknown'} · Yahoo Finance via yfinance")
for warning in dict.fromkeys(bundle.get("warnings", []) + analysis.get("warnings", [])):
    if not (bundle.get("is_demo") and warning.startswith("DEMO:")):
        st.warning(warning)
columns = st.columns(4)
columns[0].metric(f"Latest adjusted close · {currency}", number(tech.get("latest_close")))
columns[1].metric("Research score", number(score.get("overall"), 1, " / 100"))
columns[2].metric("Model data coverage", percent(score.get("coverage"), 0))
columns[3].metric("1-year adjusted return", percent(tech.get("returns", {}).get("one_year")))
st.caption("Adjusted history accounts for splits and dividends. The latest adjusted close can differ from the current trading quote.")
overview, factors_tab, risk_tab, valuation_tab, forecast_tab, ml_tab, news_tab, decision_tab, method_tab = st.tabs(
    ["Overview", "Factors", "Price & risk", "Valuation", "Forecast lab", "ML buy models", "News", "Should you buy?", "Methodology & export"])
valuation = analysis.get("valuation")
forecast = None
news_result = st.session_state.get("news_result")
news_key = f"{mode}:{symbol}"
if st.session_state.get("news_company") != news_key:
    news_result = None

with overview:
    # Filled after the valuation controls run, so edited assumptions update both
    # copies of the result in the same rerun without showing a stale verdict.
    overview_assessment = st.empty()
    left, right = st.columns([2, 1])
    with left:
        st.subheader("The price, in context")
        if history.empty:
            st.info("No usable history was returned. Available fundamentals still appear in Factors.")
        else:
            chart_range = st.select_slider("Chart history", ["3 months", "6 months", "1 year", "All available"], value="1 year")
            periods = {"3 months": 63, "6 months": 126, "1 year": 252, "All available": len(history)}
            view = history.tail(periods[chart_range])
            fig = go.Figure(go.Scatter(x=view.index, y=view["Close"], name="Adjusted close", line=dict(color="#087f8c", width=2.5)))
            for window, color in ((50, "#e49a32"), (200, "#8293aa")):
                average = history["Close"].rolling(window).mean().reindex(view.index)
                fig.add_trace(go.Scatter(x=view.index, y=average, name=f"{window}-day average", line=dict(color=color, width=1.3)))
            st.plotly_chart(chart_style(fig, yaxis_title=currency), width="stretch")
    with right:
        st.subheader("What the score says")
        st.write(score.get("interpretation") or "There is not enough evidence for an overall assessment.")
        st.caption(f"Assessment status: {str(score.get('status', 'unknown')).replace('_', ' ')}")
        st.progress(min(1., max(0., float(score.get("coverage") or 0.))), text="Available weighted model inputs")
        st.caption("Missing inputs do not receive neutral scores. Coverage measures available inputs, not confidence or predictive accuracy.")
        st.metric(f"Market capitalization · {currency}", compact(info.get("marketCap")))
        st.metric("Trailing P/E", number(info.get("trailingPE"), 1))
        st.metric("Revenue growth", percent(info.get("revenueGrowth")))

with factors_tab:
    st.subheader("Six perspectives on the company")
    st.write("Continuous scores combine valuation, business quality, growth, momentum, financial strength, and historical risk. Higher scores represent more favorable inputs within these rules.")
    factor_values = score.get("factors", {})
    fig = go.Figure(go.Bar(x=[item.get("score") for item in factor_values.values()],
        y=[key.replace("_", " ").title() for key in factor_values], orientation="h", marker_color="#087f8c"))
    fig.update_xaxes(range=[0, 100], title="Rule-based score / 100")
    st.plotly_chart(chart_style(fig), width="stretch")
    for factor, values in factor_values.items():
        with st.expander(f"{factor.replace('_', ' ').title()} · {number(values.get('score'), 1)} / 100 · {percent(values.get('coverage'), 0)} coverage"):
            st.caption(f"Target factor weight: {percent(values.get('weight'), 0)}. Available input weights determine its contribution.")
            rows = [{"Input": metric.get("label", metric.get("key")),
                "Value": percent(metric.get("value")) if metric.get("unit") == "fraction" else number(metric.get("value")),
                "Score / 100": metric.get("score"), "Input weight": metric.get("weight"), "Available": metric.get("available"),
                "Scale low": percent(metric.get("scale_low")) if metric.get("unit") == "fraction" else number(metric.get("scale_low")),
                "Scale high": percent(metric.get("scale_high")) if metric.get("unit") == "fraction" else number(metric.get("scale_high")),
                "Higher input is better": metric.get("higher_is_better"),
                "Explanation": metric.get("reason") or "Linear scale, clipped at 0 and 100."} for metric in values.get("metrics", [])]
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.info("These are absolute research heuristics, not industry percentiles. The app uses the actual company sector and does not invent an industry P/E average.")

with risk_tab:
    st.subheader("What the historical path felt like")
    cols = st.columns(4)
    for col, label, key in zip(cols,
            ["Annualized volatility", "Worst historical drawdown", "Daily 95% VaR", "Daily tail loss (CVaR)"],
            ["annualized_volatility", "max_drawdown", "historical_var95", "historical_cvar95"]):
        col.metric(label, percent(tech.get(key)))
    st.caption("Daily VaR is a historical loss threshold; CVaR is the average loss in that tail. Future losses can exceed both. Drawdown and volatility use up to 756 trailing adjusted sessions.")
    cols = st.columns(3)
    cols[0].metric("Beta vs benchmark", number(tech.get("beta")))
    cols[1].metric("RSI (14 days)", number(tech.get("rsi14"), 1))
    cols[2].metric("Downside volatility", percent(tech.get("downside_volatility")))
    if not history.empty:
        risk_close = history["Close"].tail(756)
        drawdown = risk_close / risk_close.cummax() - 1
        fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown * 100, name="Drawdown", fill="tozeroy", line=dict(color="#c65b55")))
        st.plotly_chart(chart_style(fig, "Drawdown from the previous high · trailing 756 sessions", "%"), width="stretch")
        if not benchmark_history.empty:
            aligned = pd.concat([history["Close"].rename(symbol), benchmark_history["Close"].rename("Benchmark")], axis=1).dropna()
            if len(aligned) > 1:
                normalized = aligned / aligned.iloc[0] * 100
                fig = go.Figure()
                for key, color in ((symbol, "#087f8c"), ("Benchmark", "#8293aa")):
                    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[key], name=key, line=dict(color=color)))
                st.plotly_chart(chart_style(fig, f"Growth of 100 · {symbol} vs {bundle.get('benchmark_symbol', 'benchmark')}", "Indexed value"), width="stretch")
                st.caption("Aligned observation dates; returns are in each instrument's local currency with no FX conversion.")

with valuation_tab:
    st.subheader("What would these cash flows be worth?")
    st.write("Explore a simplified equity cashflow model. The assumptions below are yours to change; they are not forecasts of the company's results.")
    cols = st.columns(3)
    growth = cols[0].number_input("Annual cashflow growth (%)", min_value=-20., max_value=30., value=5., step=1.) / 100
    discount = cols[1].number_input("Required annual equity return (%)", min_value=4., max_value=30., value=10., step=.5) / 100
    terminal = cols[2].number_input("Perpetual cashflow growth (%)", min_value=-2., max_value=5., value=2.5, step=.5) / 100
    valuation = dcf_valuation(info, growth=growth, discount_rate=discount, terminal_growth=terminal)
    st.caption("Five explicit forecast years, then a perpetual-growth terminal value. Reported operating cashflow less capex is an equity cashflow proxy assuming zero net borrowing. No debt/cash adjustment is applied again.")
    for warning in valuation.get("warnings", []):
        st.warning(warning)
    if valuation.get("scenarios"):
        for col, scenario in zip(st.columns(3), valuation["scenarios"]):
            col.metric(f"{str(scenario.get('name', 'Scenario')).title()} · {currency}", number(scenario.get("fair_value_per_share")))
        st.dataframe(pd.DataFrame(valuation["scenarios"]), width="stretch", hide_index=True)
        sensitivity = pd.DataFrame(valuation.get("sensitivity", []))
        if not sensitivity.empty and {"discount_rate", "terminal_growth", "fair_value_per_share"}.issubset(sensitivity.columns):
            st.markdown("**Sensitivity: value per share**")
            table = sensitivity.pivot(index="discount_rate", columns="terminal_growth", values="fair_value_per_share")
            table.index = [percent(v) for v in table.index]
            table.columns = [percent(v) for v in table.columns]
            st.caption("Rows: required equity return. Columns: perpetual growth.")
            st.dataframe(table.round(2), width="stretch")
    else:
        st.info("Valuation unavailable: " + "; ".join(valuation.get("reasons", []) or [str(valuation.get("status", "missing inputs"))]))
    with st.expander("Model assumptions and exclusions"):
        st.json(valuation.get("assumptions", {}))
        st.caption("Cashflow and share counts can refer to different reporting dates. This simplified model omits dilution, changing financing, and detailed financial statements. Financial businesses and funds require other valuation methods.")

with forecast_tab:
    st.subheader("Test a model before trusting its projection")
    st.write("Compare simple forecasting models using time-ordered historical holdouts. A random-walk baseline can outperform a more complex model.")
    horizon = st.select_slider("Forecast horizon (trading days)", [5, 10, 21, 63, 126], value=21)
    if st.button("Run forecast comparison", type="primary"):
        with st.spinner("Evaluating historical holdouts and simulating scenarios…"):
            result = model_forecast(history, horizon)
        st.session_state["forecast_result"] = result
        st.session_state["forecast_company"] = news_key
        st.session_state["forecast_data_stamp"] = bundle.get("fetched_at")
    if (st.session_state.get("forecast_company") == news_key
            and st.session_state.get("forecast_data_stamp") == bundle.get("fetched_at")):
        forecast = st.session_state.get("forecast_result")
    if forecast is not None and forecast.get("horizon") != horizon:
        st.info("The displayed result uses the previous horizon. Run the comparison to update it.")
    if forecast:
        for warning in forecast.get("warnings", []):
            st.warning(warning)
        backtest = forecast.get("backtest", {})
        if backtest.get("models"):
            comparison = pd.DataFrame(backtest["models"])
            for key in ("mape", "directional_accuracy", "directional_coverage"):
                if key in comparison:
                    comparison[key] = comparison[key].map(percent)
            comparison = comparison.rename(columns={"name": "Model", "mae": f"Mean absolute error ({currency})",
                "rmse": f"Root mean square error ({currency})", "mape": "Mean absolute percentage error",
                "directional_accuracy": "Direction accuracy when predicted", "directional_coverage": "Directional predictions",
                "n_predictions": "Historical holdouts", "horizon": "Horizon (sessions)"})
            st.dataframe(comparison, width="stretch", hide_index=True)
            st.caption("Direction accuracy excludes flat predictions. The random walk makes no directional call. Overlapping holdouts are correlated observations.")
        st.caption(f"Selected model: {forecast.get('selected_model', backtest.get('selected_model', 'unavailable'))}. Selection on historical errors is not independent proof of future accuracy.")
        points = pd.DataFrame(forecast.get("forecast", []))
        if not points.empty:
            fig = go.Figure()
            for lo, hi, label, color in (("lower_95", "upper_95", "95% simulated range", "rgba(8,127,140,.10)"), ("lower_80", "upper_80", "80% simulated range", "rgba(8,127,140,.20)")):
                fig.add_trace(go.Scatter(x=points["step"], y=points[hi], mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=points["step"], y=points[lo], mode="lines", fill="tonexty", fillcolor=color, line=dict(width=0), name=label))
            fig.add_trace(go.Scatter(x=points["step"], y=points["median"], name="Simulated median", line=dict(color="#087f8c")))
            fig.update_xaxes(title="Trading days ahead")
            st.plotly_chart(chart_style(fig, yaxis_title=currency), width="stretch")
            st.caption("Model-based simulated ranges are not calibrated guarantees. Future shocks and structural changes can put outcomes outside them. Horizons count observations, not calendar days.")
        else:
            st.info(f"Forecast status: {str(forecast.get('status', 'insufficient data')).replace('_', ' ')}")
        with st.expander("Full validation details"):
            st.json(backtest)
    else:
        st.info("Run the comparison to see historical error metrics and projection ranges. No model download is needed.")

with ml_tab:
    st.subheader("Learn from prices, then test on later history")
    st.write("Logistic regression, random forest and gradient boosting compete with a simple class-prior baseline. Model selection uses validation data; later test data check whether the selected model has earned a usable signal.")
    cols = st.columns(2)
    ml_horizon = cols[0].selectbox("ML prediction horizon (trading sessions)", [21, 63, 126], index=1,
        help="About one, three or six trading months. Longer horizons need more years of history for non-overlapping test outcomes.")
    ml_cost = cols[1].number_input("Assumed round-trip trading cost (basis points)", min_value=0., max_value=200., value=20., step=5.,
        help="20 basis points = 0.20% for the full entry and exit. Taxes, market impact and FX costs are not modeled.")
    with st.spinner("Training and validating the buy models on this price history…"):
        try:
            ml_result = learned_buy_models(history, benchmark_history, ml_horizon, ml_cost)
        except Exception:
            ml_result = {"status": "unavailable", "eligible_for_decision": False, "signal": "NO_RELIABLE_EDGE",
                "probability": None, "horizon": ml_horizon, "cost_bps": ml_cost,
                "reasons": ["The ML evaluation could not complete. Check the installation and price data."],
                "data": {}, "warnings": []}
    st.session_state["ml_result"] = ml_result
    if bundle.get("is_demo"):
        st.warning("Synthetic demonstration: these model scores do not measure performance on real investments.")
    st.write(ml_result.get("target_description", "The target requires sufficient adjusted price and benchmark history."))
    if ml_result.get("probability") is not None:
        cols = st.columns(3)
        cols[0].metric("Estimated target probability", percent(ml_result["probability"]))
        model_label = {"baseline": "Baseline", "logistic_regression": "Logistic", "random_forest": "Random forest",
            "hist_gradient_boosting": "Boosted trees"}.get(ml_result.get("selected_model"), "Unavailable")
        cols[1].metric("Selected model", model_label)
        cols[2].metric("Evidence check", "Passed" if ml_result.get("eligible_for_decision") else "Not qualified")
        st.caption("An estimated probability is displayed for inspection even when the model fails qualification. A failed model does not provide a buy signal.")
    findings = ml_result.get("reasons", [])
    if findings:
        st.info(findings[0])
        if len(findings) > 1:
            with st.expander("More evaluation findings"):
                for reason in findings[1:]:
                    st.write("• " + reason)
    partitions = ml_result.get("data", {}).get("partitions", {})
    if partitions:
        train_end = partitions.get("train", {}).get("feature_end")
        calibration_end = partitions.get("calibration", {}).get("label_end")
        st.caption(f"Frozen estimator trained through {train_end}; probability calibration used outcomes through {calibration_end}. The exact tested fit produces the current estimate; recent labels are not used to refit it after the test.")
        rows = [{"Partition": name.title(), "Feature start": values.get("feature_start"),
                 "Feature end": values.get("feature_end"), "Last outcome end": values.get("label_end"),
                 "Rows after purging": values.get("rows")} for name, values in partitions.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    test_result = ml_result.get("test", {})
    for label, key in (("Validation · used to select a model", "validation"), ("Final test · used only after selection", "test")):
        records = ml_result.get(key, {}).get("models", [])
        if records:
            st.markdown(f"**{label}**")
            rows = []
            for row in records:
                rows.append({"Model": row.get("name", row.get("model")), "Brier loss ↓": row.get("brier"),
                    "Log loss ↓": row.get("log_loss"), "Balanced accuracy": percent(row.get("balanced_accuracy")),
                    "ROC AUC": row.get("roc_auc"), "Target frequency": percent(row.get("positive_rate")),
                    "Daily samples": row.get("observations")})
            st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
    if test_result.get("models"):
        cols = st.columns(3)
        cols[0].metric("Non-overlapping test outcomes", str(test_result.get("independent_observations", 0)))
        cols[1].metric("Non-overlapping positive calls", str(test_result.get("independent_buy_calls", 0)))
        cols[2].metric("Test Brier improvement vs baseline", percent(test_result.get("brier_relative_improvement")))
        st.caption("Daily target windows overlap; their sample count is not a count of independent trades. Non-overlapping outcomes can still be statistically dependent.")
    checks = ml_result.get("eligibility_checks", [])
    if checks:
        with st.expander("Why the model can or cannot influence the buy assessment"):
            st.dataframe(pd.DataFrame([{"Check": item.get("key"), "Passed": item.get("passed"),
                "Value": str(item.get("value")), "Requirement": str(item.get("threshold")),
                "Reason": item.get("reason") or "Met"} for item in checks]), width="stretch", hide_index=True)
    trading = test_result.get("trading", {})
    if trading:
        with st.expander("Cost-aware results on non-overlapping historical trades"):
            st.json({key: value for key, value in trading.items() if key != "trades"})
            if trading.get("trades"):
                st.dataframe(pd.DataFrame(trading["trades"]), width="stretch", hide_index=True)
            st.caption("Historical simulation at the stated next-close entry, horizon and cost. Uninvested slots earn zero; no borrowing, tax or slippage model. This is not a promised investment result.")
    importance = ml_result.get("feature_importance", [])
    if importance:
        with st.expander("Inspect the selected model's features"):
            st.dataframe(pd.DataFrame(importance[:10]), width="stretch", hide_index=True)
            st.caption("Global model associations, not a causal explanation of an individual prediction.")
    with st.expander("ML methodology and limitations"):
        st.write(ml_result.get("methodology", "No model result available."))
        for warning in ml_result.get("warnings", []):
            st.write("• " + warning)
        st.caption("The engine uses at most the latest 20 years of source sessions. Benchmark returns remain in each instrument's local currency; choose a relevant benchmark. Repeated horizon experiments reuse historical test periods.")
        st.markdown("[Time-ordered validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) · [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)")

with news_tab:
    st.subheader("Recent coverage, kept separate from the research score")
    st.write("News adds context; headline sentiment does not establish what the stock will do next.")
    if bundle.get("is_demo"):
        st.info("Switch to Live market data to fetch real company news. Demo does not invent news headlines or sentiment.")
    else:
        key = st.text_input("NewsAPI key (optional)", type="password", help="Or set NEWS_API_KEY in your project .env, environment, or secrets.txt. The key is never included in exports.")
        finbert = st.checkbox("Analyze with FinBERT (optional AI install required)", value=False,
            help="Install requirements-ai.txt first. First use downloads the model from Hugging Face and may take several minutes.")
        if st.button("Load recent news"):
            configured_key = key.strip() or load_news_key()
            if not configured_key:
                st.info("Add a NewsAPI key to fetch headlines. All other research tools work without one.")
            else:
                with st.spinner("Loading recent news…"):
                    fetched = fetch_news(name, symbol, configured_key)
                for warning in fetched.get("warnings", []):
                    st.warning(warning)
                if fetched.get("error"):
                    st.warning(str(fetched["error"]))
                with st.spinner("Analyzing headlines…" if finbert else "Preparing headlines…"):
                    news_result = analyze_news(fetched.get("articles", []), use_finbert=finbert)
                st.session_state["news_result"] = news_result
                st.session_state["news_company"] = news_key
                st.session_state["news_fetched_at"] = fetched.get("fetched_at")
                st.session_state.pop("news_summary", None)
        if news_result:
            st.caption(f"News retrieved: {st.session_state.get('news_fetched_at', 'unknown')}")
            for warning in news_result.get("warnings", []):
                st.warning(warning)
            if news_result.get("score") is not None:
                cols = st.columns(3)
                cols[0].metric("Headline sentiment", number(news_result["score"]))
                cols[1].metric("Label", str(news_result.get("label", "Unavailable")))
                cols[2].metric("Analyzed coverage", percent(news_result.get("coverage"), 0))
                trend = pd.DataFrame(news_result.get("trend", []))
                if not trend.empty:
                    fig = go.Figure(go.Scatter(x=trend["date"], y=trend["sentiment"], mode="lines+markers",
                        name="Daily mean tone", line=dict(color="#087f8c")))
                    fig.update_yaxes(range=[-1, 1])
                    st.plotly_chart(chart_style(fig, "Daily headline tone", "P(positive) − P(negative)"), width="stretch")
                    st.caption("Daily means of analyzed articles; the overall score separately weights publication age.")
            else:
                st.caption("Sentiment has not been analyzed. Headlines are shown without an AI score.")
            for article in news_result.get("articles", []):
                with st.container(border=True):
                    st.text(article.get("title", "Untitled"))
                    source = article.get("source", "")
                    if isinstance(source, dict):
                        source = source.get("name", "")
                    st.caption(f"{source} · {article.get('publishedAt', 'Date unavailable')}")
                    if article.get("description"):
                        st.write(article["description"])
                    if article.get("sentiment"):
                        sentiment = article["sentiment"]
                        st.caption(f"FinBERT: {sentiment.get('label')} · probability of model label {percent(sentiment.get('confidence'))}")
                    url = article.get("url", "")
                    if url.startswith(("https://", "http://")):
                        st.link_button("Read source", url)
            if not news_result.get("articles"):
                st.info("No usable recent articles were returned.")
            else:
                with st.expander("Optional local news summary"):
                    st.caption("Requires a separately installed Ollama service and model. AI summaries can make mistakes; check the linked reporting.")
                    ollama_model = st.text_input("Installed Ollama model", value="mistral")
                    if st.button("Summarize with local Ollama"):
                        with st.spinner("Summarizing snippets with your local model…"):
                            summary_result = summarize_news(news_result["articles"], model=ollama_model)
                        st.session_state["news_summary"] = {"company": news_key, **summary_result}
                    summary_result = st.session_state.get("news_summary", {})
                    if summary_result.get("company") == news_key:
                        if summary_result.get("summary"):
                            st.write(summary_result["summary"])
                        elif summary_result.get("error"):
                            st.info(summary_result["error"])

decision = machine_learning_decision(analysis, valuation, info, ml_result,
    price_as_of=bundle.get("price_as_of"), quote_as_of=bundle.get("quote_as_of"),
    fundamentals_as_of=bundle.get("fundamentals_as_of"), is_demo=bundle.get("is_demo", False),
    now=bundle.get("fetched_at") if bundle.get("is_demo") else None)
st.session_state["investment_decision"] = decision
with overview_assessment.container():
    show_decision(decision)
with decision_tab:
    show_decision(decision, detailed=True)

with method_tab:
    st.subheader("Know what went into the result")
    st.write("Finsight is a research aid. The factor model uses documented rules; forecast performance is measured on historical holdouts; valuation is an editable scenario. None establishes future returns.")
    st.json(analysis.get("methodology", {}))
    with st.expander("Data provenance and availability"):
        st.json({key: bundle.get(key) for key in ("symbol", "benchmark_symbol", "currency", "price_as_of", "quote_as_of", "fundamentals_as_of", "price_basis", "fetched_at", "warnings")})
        st.caption("Retrieval time is not the financial statement date. Fundamentals may be missing, stale, or use different reporting periods. News availability depends on your NewsAPI plan.")
    cols = st.columns(2)
    cols[0].download_button("Download research report (JSON)", export_report(bundle, analysis, valuation, forecast, news_result, decision, ml_result),
        file_name=f"finsight-{symbol}-report.json", mime="application/json", width="stretch")
    cols[1].download_button("Download adjusted prices (CSV)", history.to_csv(index_label="Date"),
        file_name=f"finsight-{symbol}-prices.csv", mime="text/csv", width="stretch")
    st.markdown("Model references: [cashflows and discount rates](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/val.html), [terminal value](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/valquestions/termvalapproaches.htm), [time-ordered validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).")
st.divider()
st.caption("Finsight 3.0 · ML buy signals with held-out evidence and risk checks · Prices may be delayed. No automatic trading.")
