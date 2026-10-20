"""Provider-boundary checks; no live finance requests are made."""
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from finsight import data


@pytest.mark.parametrize("symbol, expected", [(" msft ", "MSFT"), ("BRK-B", "BRK-B"),
    ("7203.T", "7203.T"), ("^GSPC", "^GSPC"), ("EURUSD=X", "EURUSD=X"), ("GC=F", "GC=F")])
def test_common_yahoo_symbols(symbol, expected):
    assert data.validate_symbol(symbol) == expected


@pytest.mark.parametrize("symbol", ["", "AAPL MSFT", "<script>", "AAPL;ls", "../secrets", "https://example.com", "A" * 30, None])
def test_invalid_symbols_rejected_before_network(symbol, monkeypatch):
    provider = Mock()
    monkeypatch.setattr(data.yf, "Ticker", provider)
    with pytest.raises(ValueError):
        data.fetch_market_data(symbol)
    provider.assert_not_called()


def test_history_sanitizer_drops_bad_bars_keeps_local_date_and_missing_gaps():
    dates = pd.DatetimeIndex(["2025-01-03", "2025-01-02", "2025-01-02", "2025-01-06", "2025-01-07", "2025-01-08"], tz="Asia/Tokyo")
    frame = pd.DataFrame({"Close": ["12", 10, 11, 0, np.inf, "bad"], "Open": [12, 10, -1, 1, 1, 1],
                          "Volume": [2, 3, -1, 3, 4, 5]}, index=dates)
    result = data.sanitize_history(frame)
    assert list(result.index) == [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]
    assert list(result.Close) == [11, 12]
    assert pd.isna(result.Open.iloc[0]) and pd.isna(result.Volume.iloc[0])
    assert result.index.tz is None
    assert len(frame) == 6  # The caller's original is never mutated.


def test_missing_close_or_malformed_dates_do_not_invent_history():
    assert data.sanitize_history(pd.DataFrame({"Open": [1]}, index=["bad"])).empty
    assert data.sanitize_history(pd.DataFrame({"Close": [1]}, index=[1])).empty
    assert data.sanitize_history(None).empty


def test_fetch_prices_and_benchmark_share_adjusted_basis_and_preserve_quote(monkeypatch):
    history = pd.DataFrame({"Close": [90.0, 100.0]}, index=pd.date_range("2025-01-02", periods=2))
    tickers = {}

    def make(symbol):
        fake = Mock()
        fake.history.return_value = history
        fake.get_info.return_value = {"currency": "GBP", "regularMarketPrice": 103.0,
                                     "regularMarketTime": 1735909200}
        tickers.setdefault(symbol, []).append(fake)
        return fake

    monkeypatch.setattr(data.yf, "Ticker", make)
    result = data.fetch_market_data("abc.l", "^ftse", period="5y")
    assert result["latest_close"] == 100.0
    assert result["info"]["regularMarketPrice"] == 103.0
    assert result["currency"] == "GBP"
    assert result["price_as_of"] == "2025-01-03"
    assert result["quote_as_of"] is not None
    assert result["is_demo"] is False
    assert result["warnings"] == []
    for symbol in ["ABC.L", "^FTSE"]:
        tickers[symbol][0].history.assert_called_once_with(period="5y", interval="1d", auto_adjust=True, actions=False, timeout=10)


def test_partial_provider_failure_keeps_successful_history(monkeypatch):
    history = pd.DataFrame({"Close": [100.0, 102.0]}, index=pd.date_range("2025-01-02", periods=2))

    def make(symbol):
        fake = Mock()
        fake.get_info.side_effect = RuntimeError("provider failure with private payload")
        if symbol == "SPY":
            fake.history.side_effect = TimeoutError("secret provider URL")
        else:
            fake.history.return_value = history
        return fake

    monkeypatch.setattr(data.yf, "Ticker", make)
    result = data.fetch_market_data("AAPL")
    assert len(result["history"]) == 2 and result["benchmark"].empty
    assert result["info"] == {}
    assert any("fundamentals" in warning for warning in result["warnings"])
    assert any("Benchmark" in warning for warning in result["warnings"])
    assert "private payload" not in " ".join(result["warnings"])
    assert result["is_demo"] is False


def test_same_benchmark_reuses_history_without_duplicate_request(monkeypatch):
    fake = Mock()
    fake.history.return_value = pd.DataFrame({"Close": [100]}, index=pd.date_range("2025-01-02", periods=1))
    fake.get_info.return_value = {"currency": "USD"}
    monkeypatch.setattr(data.yf, "Ticker", Mock(return_value=fake))
    result = data.fetch_market_data("SPY", "SPY")
    assert fake.history.call_count == 1
    pd.testing.assert_frame_equal(result["history"], result["benchmark"])
    assert result["history"] is not result["benchmark"]


def test_invalid_period_does_not_call_provider(monkeypatch):
    provider = Mock()
    monkeypatch.setattr(data.yf, "Ticker", provider)
    with pytest.raises(ValueError):
        data.fetch_market_data("AAPL", period="nonsense")
    provider.assert_not_called()


def test_demo_is_reproducible_clearly_labelled_and_offline(monkeypatch):
    provider = Mock(side_effect=AssertionError("Demo must never call the network"))
    monkeypatch.setattr(data.yf, "Ticker", provider)
    first, second = data.demo_market_data(), data.demo_market_data()
    pd.testing.assert_frame_equal(first["history"], second["history"])
    pd.testing.assert_frame_equal(first["benchmark"], second["benchmark"])
    assert first["is_demo"] and first["symbol"] == "DEMO"
    assert first["fetched_at"] == second["fetched_at"]
    assert "fictional" in " ".join(first["warnings"]).lower()
    assert len(first["history"]) > 1200
    assert (first["history"].Close > 0).all()
    assert (first["history"].High >= first["history"].Close).all()
    assert (first["history"].Low <= first["history"].Close).all()


def test_open_session_bar_is_not_treated_as_a_completed_close(monkeypatch):
    from datetime import datetime, timezone
    fake = Mock()
    fake.history.return_value = pd.DataFrame({"Close": [100., 110.]}, index=pd.to_datetime(["2026-10-16", "2026-10-19"]))
    fake.get_info.return_value = {"currency": "USD", "marketState": "REGULAR",
        "exchangeTimezoneName": "America/New_York", "regularMarketPrice": 111.,
        "regularMarketTime": datetime(2026, 10, 19, 15, 0, tzinfo=timezone.utc).timestamp()}
    monkeypatch.setattr(data.yf, "Ticker", lambda symbol: fake)
    result = data.fetch_market_data("MSFT", "MSFT")
    assert result["price_as_of"] == "2026-10-16"
    assert result["latest_close"] == 100.
    assert result["info"]["regularMarketPrice"] == 111.
    assert any("provisional" in warning for warning in result["warnings"])
    pd.testing.assert_frame_equal(result["history"], result["benchmark"])


def test_closed_session_bar_is_retained(monkeypatch):
    fake = Mock()
    fake.history.return_value = pd.DataFrame({"Close": [100., 110.]}, index=pd.to_datetime(["2026-10-16", "2026-10-19"]))
    fake.get_info.return_value = {"currency": "USD", "marketState": "POST", "regularMarketPrice": 111.}
    monkeypatch.setattr(data.yf, "Ticker", lambda symbol: fake)
    result = data.fetch_market_data("MSFT", "MSFT")
    assert result["latest_close"] == 110.
