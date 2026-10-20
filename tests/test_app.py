"""Exercise the dashboard's user flows without network, credentials or AI weights."""
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from finsight import data, news, ml

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app(monkeypatch):
    import streamlit as st
    st.cache_data.clear()
    monkeypatch.setattr(data.yf, "Ticker", Mock(side_effect=AssertionError("Unexpected network request")))
    monkeypatch.setattr(news, "load_news_key", lambda: None)
    monkeypatch.setattr(ml, "predict_buy_signal", lambda history, benchmark, **kwargs: {
        "status": "available", "eligible_for_decision": False, "signal": "NO_RELIABLE_EDGE",
        "probability": .70, "horizon": kwargs["horizon"], "cost_bps": kwargs["cost_bps"],
        "selected_model": "logistic_regression", "data": {"latest_feature_date": "2026-10-16"},
        "reasons": ["No out-of-sample edge was demonstrated."], "warnings": []})
    return AppTest.from_file(APP, default_timeout=30)


def click(app, label):
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


def test_demo_starts_without_network_or_keys(app):
    app.run()
    assert not app.exception
    assert len(app.tabs) == 9
    assert "DEMO" in app.warning[0].value
    assert any(metric.label == "Research score" and metric.value != "Unavailable" for metric in app.metric)


def test_forecast_runs_and_persists_through_rerun(app):
    app.run()
    click(app, "Run forecast comparison")
    assert app.session_state["forecast_result"]["status"] == "available"
    assert app.session_state["forecast_result"]["horizon"] == 21
    app.run()
    assert not app.exception
    assert any("Selected model:" in caption.value for caption in app.caption)
    app.select_slider[1].set_value(5).run()
    assert any("previous horizon" in info.value for info in app.info)
    click(app, "Run forecast comparison")
    assert app.session_state["forecast_result"]["horizon"] == 5


def test_invalid_valuation_assumptions_show_reason(app):
    app.run()
    app.number_input[1].set_value(4.).run()
    app.number_input[2].set_value(5.).run()
    assert not app.exception
    assert any("Required equity return must exceed" in info.value for info in app.info)
    assert app.session_state["investment_decision"]["verdict"] == "INSUFFICIENT_DATA"


def test_final_assessment_uses_current_edited_valuation(app):
    app.run()
    first = app.session_state["investment_decision"]
    assert first["verdict"] == "INSUFFICIENT_DATA"
    assert first["is_demo"] and not first["actionable"]
    assert any(tab.label == "Should you buy?" for tab in app.tabs)
    app.number_input[1].set_value(6.).run()
    assert not app.exception
    changed = app.session_state["investment_decision"]
    assert changed["evidence"]["base_fair_value"] > first["evidence"]["base_fair_value"]
    assert changed["evidence"]["valuation_assumptions"]["discount_rate"] == .06
    assert changed["verdict"] == "INSUFFICIENT_DATA"  # Changing DCF cannot manufacture a validated ML signal.
    assert sum(heading.value == "Should you buy this stock?" for heading in app.subheader) == 2


def test_live_mode_does_not_fetch_until_submitted(app, monkeypatch):
    provider = Mock(return_value=data.demo_market_data())
    monkeypatch.setattr(data, "fetch_market_data", provider)
    app.run()
    app.radio[0].set_value("Live market data").run()
    assert not app.exception
    provider.assert_not_called()
    click(app, "Analyze company")
    provider.assert_called_once_with("MSFT", benchmark_symbol="SPY", period="max")


def test_live_missing_data_and_news_key_do_not_crash(app, monkeypatch):
    bundle = data.demo_market_data()
    bundle.update(symbol="EMPTY", is_demo=False, info={}, history=pd.DataFrame(), benchmark=pd.DataFrame())
    monkeypatch.setattr(data, "fetch_market_data", lambda *args, **kwargs: bundle)
    app.run()
    app.radio[0].set_value("Live market data").run()
    click(app, "Analyze company")
    assert next(metric for metric in app.metric if metric.label == "Research score").value == "Unavailable"
    click(app, "Run forecast comparison")
    assert app.session_state["forecast_result"]["status"] == "insufficient_data"
    click(app, "Load recent news")
    assert any("Add a NewsAPI key" in info.value for info in app.info)


def test_invalid_live_symbol_shows_actionable_error(app, monkeypatch):
    def invalid(*args, **kwargs):
        raise ValueError("Enter one ticker using letters and numbers.")
    monkeypatch.setattr(data, "fetch_market_data", invalid)
    app.run()
    app.radio[0].set_value("Live market data").run()
    app.text_input[0].set_value("../../BAD")
    click(app, "Analyze company")
    assert any("Enter one ticker" in error.value for error in app.error)


def test_optional_ai_failure_preserves_headlines(app, monkeypatch):
    from datetime import datetime, timezone
    bundle = data.demo_market_data()
    bundle.update(is_demo=False, symbol="TEST")
    monkeypatch.setattr(data, "fetch_market_data", lambda *args, **kwargs: bundle)
    article = {"title": "Example company reports results", "url": "https://example.com/news",
               "publishedAt": datetime.now(timezone.utc).isoformat(), "source": {"name": "Example"}}
    monkeypatch.setattr(news, "fetch_news", lambda *args, **kwargs: {"articles": [article]})
    monkeypatch.setattr(news, "_load_finbert", Mock(side_effect=ImportError("Missing optional dependency")))
    app.run()
    app.radio[0].set_value("Live market data").run()
    click(app, "Analyze company")
    next(field for field in app.text_input if field.label == "NewsAPI key (optional)").set_value("test-key")
    app.checkbox[0].check()
    click(app, "Load recent news")
    assert app.session_state["news_result"]["status"] == "unavailable"
    assert any("optional ai dependencies" in warning.value.lower() for warning in app.warning)
    assert any("Example company reports results" in text.value for text in app.text)


def test_demo_runs_the_real_ml_pipeline_without_network(monkeypatch):
    import streamlit as st
    st.cache_data.clear()
    monkeypatch.setattr(data.yf, "Ticker", Mock(side_effect=AssertionError("Unexpected network request")))
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    result = app.session_state["ml_result"]
    assert result["status"] == "available"
    assert len(result["validation"]["models"]) == 4
    assert len(result["test"]["models"]) == 4
    assert app.session_state["investment_decision"]["is_demo"]
    assert not app.session_state["investment_decision"]["actionable"]


def test_ml_horizon_changes_update_the_final_target(app):
    app.run()
    next(control for control in app.selectbox if control.label == "ML prediction horizon (trading sessions)").set_value(21).run()
    assert not app.exception
    assert app.session_state["ml_result"]["horizon"] == 21
    assert app.session_state["investment_decision"]["evidence"]["ml_horizon_sessions"] == 21
