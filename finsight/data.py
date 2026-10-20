"""Market-data boundary: adjusted daily bars, honest missing data, and offline examples.

The UI owns caching so a user can explicitly refresh a snapshot.  This module never
substitutes synthetic values for a failed live request.
"""
from __future__ import annotations

from datetime import datetime, timezone
from queue import Empty, Queue
import re
from threading import Thread
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
import yfinance as yf

_SYMBOL = re.compile(r"\^?[A-Z0-9][A-Z0-9.=-]{0,23}\Z")
_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
PRICE_BASIS = "Split/dividend-adjusted daily close"


def validate_symbol(symbol: str) -> str:
    """Accept a single Yahoo-style symbol, including international shares/FX/indexes."""
    if not isinstance(symbol, str):
        raise ValueError("Enter one ticker symbol, such as MSFT, BRK-B, or 7203.T.")
    cleaned = symbol.strip().upper()
    if not _SYMBOL.fullmatch(cleaned) or ".." in cleaned or cleaned.endswith((".", "-", "=")):
        raise ValueError("Enter one ticker using letters, numbers, a dot, dash, ^ prefix, or = suffix code.")
    return cleaned


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex([], name="Date"))


def sanitize_history(frame: Any) -> pd.DataFrame:
    """Return sorted unique trading dates and finite positive adjusted closes.

    Daily bars retain their exchange-local calendar date rather than shifting an
    Asian exchange's midnight into the previous UTC day. Missing bars stay missing.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty or "Close" not in frame:
        return _empty_history()
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex) or not result.columns.is_unique:
        return _empty_history()  # Ticker.history must describe exactly one symbol.
    if isinstance(result.index, pd.DatetimeIndex):
        dates = result.index.tz_localize(None).normalize()
    else:
        def trading_date(value: Any) -> pd.Timestamp:
            if isinstance(value, (int, float, np.number)):
                return pd.NaT
            try:
                return pd.Timestamp(value).tz_localize(None).normalize()
            except (ValueError, TypeError, OverflowError):
                return pd.NaT
        dates = pd.DatetimeIndex([trading_date(value) for value in result.index])
    result.index = dates
    result.index.name = "Date"
    result = result.loc[~result.index.isna()]
    for column in ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = result.loc[result["Close"].notna() & (result["Close"] > 0)]
    for column in ["Open", "High", "Low"]:
        if column in result:
            result.loc[result[column] <= 0, column] = np.nan
    if "Volume" in result:
        result.loc[result["Volume"] < 0, "Volume"] = np.nan
    return result.loc[~result.index.duplicated(keep="last")].sort_index()


def _bounded_call(operation: Callable[[], Any], timeout: float) -> Any:
    """Bound the caller's wait, including vendor methods without timeout arguments.

    Vendor socket timeouts still apply. A daemon prevents a stuck provider request
    from holding up dashboard rendering or application exit.
    """
    results: Queue = Queue(maxsize=1)

    def run() -> None:
        try:
            results.put((True, operation()))
        except Exception as exc:
            results.put((False, exc))

    Thread(target=run, daemon=True).start()
    try:
        succeeded, value = results.get(timeout=timeout)
    except Empty:
        raise TimeoutError("The market data provider took too long to respond.") from None
    if not succeeded:
        raise value
    return value


def _fetch_history(symbol: str, period: str) -> pd.DataFrame:
    # Explicit auto_adjust is essential: every return/benchmark calculation uses
    # the same split/dividend-adjusted basis, independently of yfinance defaults.
    return yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True, actions=False, timeout=10)


def _safe_info(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    return {key: (None if isinstance(item, (float, np.floating)) and not np.isfinite(item) else item)
            for key, item in value.items()}


def _timestamp_from_epoch(value: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def fetch_market_data(symbol: str, benchmark_symbol: str = "SPY", period: str = "5y") -> dict:
    """Fetch a partially usable snapshot even when fundamentals/benchmark fail.

    ``info`` prices are provider quotes; ``latest_close`` and ``history`` use
    adjusted daily closes. Do not mix these two price bases in a return series.
    ``fetched_at`` records retrieval, not the date of the financial statements.
    """
    symbol = validate_symbol(symbol)
    benchmark_symbol = validate_symbol(benchmark_symbol)
    if period not in _PERIODS:
        raise ValueError("Choose a supported history period (1mo through 10y, ytd, or max).")
    warnings: list[str] = []
    history = _empty_history()
    benchmark = _empty_history()
    info: dict = {}
    try:
        raw_history = _bounded_call(lambda: _fetch_history(symbol, period), 15)
        history = sanitize_history(raw_history)
        if history.empty:
            warnings.append(f"No usable adjusted price history was returned for {symbol}.")
        elif len(history) < len(raw_history):
            warnings.append(f"Removed {len(raw_history) - len(history)} invalid or duplicate price rows for {symbol}.")
    except Exception:
        warnings.append(f"Price history for {symbol} is unavailable. The provider may be offline, rate limited, or unable to find this ticker.")
    try:
        info = _safe_info(_bounded_call(lambda: yf.Ticker(symbol).get_info(), 15))
        if not info:
            warnings.append("Company fundamentals are unavailable; dependent models will show missing inputs.")
    except Exception:
        warnings.append("Company fundamentals are unavailable; dependent models will show missing inputs.")
    if not history.empty and str(info.get("marketState") or "").upper() == "REGULAR":
        quote_stamp = _timestamp_from_epoch(info.get("regularMarketTime"))
        if quote_stamp:
            try:
                exchange_zone = ZoneInfo(str(info.get("exchangeTimezoneName") or "UTC"))
            except (ZoneInfoNotFoundError, ValueError):
                exchange_zone = timezone.utc
            quote_day = datetime.fromisoformat(quote_stamp).astimezone(exchange_zone).date()
            if history.index[-1].date() == quote_day:
                history = history.iloc[:-1].copy()
                warnings.append("Excluded the open session's provisional daily bar. Models use completed session closes; the current quote remains available separately.")
    if symbol == benchmark_symbol:
        benchmark = history.copy()
    else:
        try:
            benchmark = sanitize_history(_bounded_call(lambda: _fetch_history(benchmark_symbol, period), 15))
            if benchmark.empty:
                warnings.append(f"No usable benchmark history for {benchmark_symbol}; relative risk metrics are unavailable.")
        except Exception:
            warnings.append(f"Benchmark {benchmark_symbol} is unavailable; relative risk metrics are unavailable.")
    currency = info.get("currency") or None
    if not currency:
        warnings.append("Quote currency was not supplied by the provider. Currency-dependent comparisons need care.")
    as_of = history.index[-1].date().isoformat() if not history.empty else None
    return {
        "symbol": symbol, "benchmark_symbol": benchmark_symbol, "info": info,
        "history": history, "benchmark": benchmark,
        "fetched_at": datetime.now(timezone.utc).isoformat(), "warnings": warnings,
        "currency": currency, "price_as_of": as_of,
        "latest_close": float(history["Close"].iloc[-1]) if not history.empty else None,
        "price_basis": PRICE_BASIS, "quote_as_of": _timestamp_from_epoch(info.get("regularMarketTime")),
        "fundamentals_as_of": _timestamp_from_epoch(info.get("mostRecentQuarter")),
        "is_demo": False, "source": "Yahoo Finance via yfinance",
    }


def demo_market_data() -> dict:
    """Deterministic fictional ten-year dataset: never presented as live prices."""
    rng = np.random.default_rng(157)
    dates = pd.bdate_range("2016-10-17", "2026-10-16", name="Date")
    common = rng.normal(0.00030, 0.0090, len(dates))
    benchmark_close = 100 * np.exp(np.cumsum(common))
    stock_returns = 0.00020 + 1.12 * common + rng.normal(0, 0.0085, len(dates))
    # Add a drawdown regime so risk models do not receive an implausibly smooth trend.
    stock_returns[330:365] -= 0.004
    close = 75 * np.exp(np.cumsum(stock_returns))

    def bars(prices: np.ndarray) -> pd.DataFrame:
        spread = rng.uniform(0.001, 0.015, len(prices))
        opened = prices * (1 + rng.normal(0, 0.003, len(prices)))
        return pd.DataFrame({"Open": opened, "High": np.maximum(opened, prices) * (1 + spread),
                             "Low": np.minimum(opened, prices) * (1 - spread), "Close": prices,
                             "Volume": rng.integers(800_000, 9_000_000, len(prices))}, index=dates)
    history, benchmark = bars(close), bars(benchmark_close)
    price = float(close[-1])
    shares = 1_200_000_000
    info = {
        "symbol": "DEMO", "longName": "Finsight Example Technologies (fictional)", "shortName": "Fictional Example Technologies",
        "sector": "Technology", "industry": "Software - Infrastructure", "quoteType": "EQUITY", "currency": "USD",
        "financialCurrency": "USD", "exchange": "DEMO", "country": "United States",
        "longBusinessSummary": "Fictional data for exploring the dashboard offline. No values describe a real security.",
        "regularMarketPrice": price, "currentPrice": price, "sharesOutstanding": shares,
        "marketCap": price * shares, "trailingPE": 23.4, "forwardPE": 21.0,
        "trailingEps": price / 23.4, "forwardEps": price / 21.0, "priceToBook": 5.3,
        "priceToSalesTrailing12Months": 5.2, "enterpriseToEbitda": 17.0,
        "returnOnEquity": 0.226, "returnOnAssets": 0.115, "profitMargins": 0.22,
        "operatingMargins": 0.27, "grossMargins": 0.68, "revenueGrowth": 0.12,
        "earningsGrowth": 0.15, "debtToEquity": 48.0, "currentRatio": 1.85,
        "totalRevenue": 24_000_000_000, "totalCash": 12_000_000_000,
        "totalDebt": 8_000_000_000, "freeCashflow": 5_800_000_000,
        "operatingCashflow": 6_900_000_000, "ebitda": 7_200_000_000,
        "netIncomeToCommon": 5_280_000_000, "bookValue": price / 5.3,
        "payoutRatio": 0.24, "dividendRate": price * 0.012, "dividendYield": 1.2,
        "beta": 1.12, "fiftyTwoWeekHigh": float(max(close[-252:])),
        "fiftyTwoWeekLow": float(min(close[-252:])), "mostRecentQuarter": 1782777600,
    }
    return {
        "symbol": "DEMO", "benchmark_symbol": "DEMO-MARKET", "info": info, "history": history,
        "benchmark": benchmark, "fetched_at": "2026-10-19T00:00:00+00:00",
        "warnings": ["DEMO: All prices and fundamentals are fictional, deterministic examples. They must not guide real trades."],
        "currency": "USD", "price_as_of": dates[-1].date().isoformat(), "latest_close": price,
        "price_basis": PRICE_BASIS, "quote_as_of": "2026-10-16T20:00:00+00:00", "fundamentals_as_of": "2026-06-30T00:00:00+00:00",
        "is_demo": True, "source": "Deterministic synthetic demonstration data",
    }
