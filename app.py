import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import requests
import pandas as pd
from transformers import pipeline

# -----------------------
# Load secrets safely
# -----------------------

def load_secrets(filepath="secrets.txt"):
    secrets = {}
    try:
        with open(filepath) as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    secrets[key.strip()] = value.strip()
    except FileNotFoundError:
        st.warning("secrets.txt not found. Please create it with your API keys.")
    return secrets

secrets = load_secrets()
api_key = secrets.get("NEWS_API_KEY", None)

if api_key is None:
    st.error("API key not found. Please check secrets.txt.")
    st.stop()

# -----------------------
# Finsight App
# -----------------------

st.set_page_config(page_title="Finsight", layout="wide")

st.title("Finsight: Stock Health Dashboard")

# Ticker input
ticker_input = st.text_input("Enter a stock ticker (e.g. MSFT, AAPL, TSLA)")

if ticker_input:
    try:
        ticker = yf.Ticker(ticker_input.upper())

        # Fetch info
        info = ticker.info

        company_name = info.get("longName", ticker_input.upper())
        current_price = info.get("regularMarketPrice")

        # Display company name and current price
        st.header(f"{company_name} ({ticker_input.upper()})")
        if current_price is not None:
            st.subheader(f"Current Price: ${current_price:.2f}")
        else:
            st.subheader("Current Price: Data not available")

        # ------------------------------
        # Plot price history with technicals
        # ------------------------------

        hist = ticker.history(period="6mo")

        if not hist.empty:
            # Calculate moving averages
            hist["MA20"] = hist["Close"].rolling(window=20).mean()
            hist["MA50"] = hist["Close"].rolling(window=50).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                name="Close Price"
            ))
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist["MA20"],
                mode="lines",
                name="20-day MA"
            ))
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist["MA50"],
                mode="lines",
                name="50-day MA"
            ))
            fig.update_layout(title="Price History (with Technical Indicators)")
            st.plotly_chart(fig)
        else:
            st.write("No price data available.")

        # ------------------------------
        # Financial Metrics
        # ------------------------------

        st.subheader("Key Financial Metrics")

        metrics = {
            "Market Cap": info.get("marketCap"),
            "PE Ratio (TTM)": info.get("trailingPE"),
            "EPS (TTM)": info.get("trailingEps"),
            "Dividend Yield": info.get("dividendYield"),
            "52 Week High": info.get("fiftyTwoWeekHigh"),
            "52 Week Low": info.get("fiftyTwoWeekLow"),
            "ROE": info.get("returnOnEquity"),
        }

        for key, value in metrics.items():
            if value is not None:
                st.write(f"**{key}:** {value}")
            else:
                st.write(f"**{key}:** Data not available.")

        # ------------------------------
        # Industry Comparison
        # ------------------------------

        st.subheader("Valuation vs Industry Average")

        # Example industry PE averages
        industry_pe_averages = {
            "Aerospace & Defense": 18.3,
            "Software": 24.7,
            "Automotive": 14.2,
            "Banks": 10.1,
        }

        # For demo, assume RTX is Aerospace
        sector = "Aerospace & Defense"
        industry_avg_pe = industry_pe_averages.get(sector)

        pe_ratio = info.get("trailingPE")

        if pe_ratio and industry_avg_pe:
            st.write(
                f"**Sector:** {sector}\n\n"
                f"**Company PE:** {pe_ratio}\n\n"
                f"**Industry Avg PE:** {industry_avg_pe}\n\n"
            )
            if pe_ratio < industry_avg_pe:
                st.success("This stock may be undervalued vs its industry.")
            else:
                st.warning("This stock appears pricier than its industry average.")
        else:
            st.write("Insufficient data for industry comparison.")

        # ------------------------------
        # Quant Factor Scoring
        # ------------------------------

        st.subheader("Quant Factor Scores")

        # VALUE
        if pe_ratio is not None:
            if pe_ratio < 15:
                value_score = 100
            elif pe_ratio < 25:
                value_score = 50
            else:
                value_score = 0
        else:
            value_score = 50

        st.write(f"**Value Score:** {value_score}")

        # QUALITY
        roe = info.get("returnOnEquity")
        if roe is not None:
            roe_pct = roe * 100
            if roe_pct > 15:
                quality_score = 100
            elif roe_pct > 5:
                quality_score = 50
            else:
                quality_score = 0
        else:
            quality_score = 50

        st.write(f"**Quality Score:** {quality_score}")

        # MOMENTUM
        if not hist.empty:
            start_price = hist["Close"].iloc[0]
            end_price = hist["Close"].iloc[-1]
            pct_change = ((end_price - start_price) / start_price) * 100

            if pct_change > 10:
                momentum_score = 100
            elif pct_change > -5:
                momentum_score = 50
            else:
                momentum_score = 0

            st.write(f"**Momentum Score:** {momentum_score} ({pct_change:.2f}% in 6 months)")
        else:
            momentum_score = 50

        # ------------------------------
        # Sentiment Analysis with FinBERT & Mistral
        # ------------------------------

        st.subheader("News Sentiment Analysis")

        finbert = pipeline(
            "sentiment-analysis",
            model="yiyanghkust/finbert-tone"
        )

        url = (
            f"https://newsapi.org/v2/everything?"
            f"q=\"{company_name}\"&"
            f"sortBy=publishedAt&"
            f"language=en&"
            f"pageSize=20&"
            f"apiKey={api_key}"
        )

        response = requests.get(url)

        sentiment_trend_data = []

        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles")

            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            all_texts = []

            if articles:
                for article in articles:
                    title = article["title"]
                    content = article.get("content") or ""
                    text_to_analyze = f"{title}. {content}"
                    date = article["publishedAt"][:10]

                    if not text_to_analyze.strip():
                        continue

                    results = finbert(text_to_analyze[:512], top_k=None)
                    label_scores = {res["label"].lower(): res["score"] for res in results}
                    label = max(label_scores, key=label_scores.get)

                    sentiment_counts[label] += 1

                    avg_sentiment = label_scores.get("positive", 0) - label_scores.get("negative", 0)
                    sentiment_trend_data.append({"date": date, "sentiment": avg_sentiment})

                    st.write(f"- {title}")
                    st.write(
                        f"    → FinBERT Sentiment breakdown: "
                        f"Positive: {label_scores.get('positive', 0):.2f}, "
                        f"Neutral: {label_scores.get('neutral', 0):.2f}, "
                        f"Negative: {label_scores.get('negative', 0):.2f}"
                    )

                    if label == "neutral" or label_scores[label] < 0.65:
                        def analyze_with_ollama(article_text):
                            prompt = f"""
You are a financial analyst. Summarize the following news article and determine if it is Positive, Negative, or Neutral for the company's stock price. Provide a brief explanation.

Article:
{article_text}
"""

                            payload = {
                                "model": "mistral",
                                "prompt": prompt,
                                "stream": False
                            }

                            try:
                                resp = requests.post(
                                    "http://localhost:11434/api/generate",
                                    json=payload
                                )
                                response_json = resp.json()
                                if "response" in response_json:
                                    return response_json["response"].strip()
                                elif "error" in response_json:
                                    return f"LLM error: {response_json['error']}"
                                else:
                                    return f"Unexpected LLM response: {response_json}"
                            except Exception as e:
                                return f"Error contacting Ollama LLM: {e}"

                        llm_response = analyze_with_ollama(text_to_analyze)
                        st.write(f"    → Mistral LLM Analysis:\n{llm_response}")

                    all_texts.append(text_to_analyze)

                # Plot sentiment trend
                if sentiment_trend_data:
                    trend_df = pd.DataFrame(sentiment_trend_data)
                    trend_df = trend_df.groupby("date").mean().reset_index()

                    st.subheader("Sentiment Trend Over Time")
                    trend_fig = go.Figure()
                    trend_fig.add_trace(go.Scatter(
                        x=trend_df["date"],
                        y=trend_df["sentiment"],
                        mode="lines+markers",
                        name="Average Sentiment"
                    ))
                    trend_fig.update_layout(
                        yaxis_title="Average Sentiment Score (-1 to +1)",
                        xaxis_title="Date",
                    )
                    st.plotly_chart(trend_fig)

                # Summarize all news
                if all_texts:
                    all_news_text = "\n\n".join(all_texts)
                    st.subheader("News Summary (LLM)")
                    summary = analyze_with_ollama(all_news_text)
                    st.write(summary)

                # Compute sentiment verdict
                total = sum(sentiment_counts.values())
                if total > 0:
                    pos_ratio = sentiment_counts["positive"] / total
                    neg_ratio = sentiment_counts["negative"] / total

                    if pos_ratio > 0.5:
                        sentiment_verdict = "Positive"
                    elif neg_ratio > 0.5:
                        sentiment_verdict = "Negative"
                    else:
                        sentiment_verdict = "Neutral"

                    st.success(
                        f"**Overall FinBERT Sentiment:** {sentiment_verdict} "
                        f"(Positive: {sentiment_counts['positive']}, "
                        f"Neutral: {sentiment_counts['neutral']}, "
                        f"Negative: {sentiment_counts['negative']})"
                    )
                else:
                    sentiment_verdict = "Neutral"
                    st.write("No meaningful content found for sentiment analysis.")

            else:
                sentiment_verdict = "Neutral"
                st.write("No recent news articles found.")

        else:
            sentiment_verdict = "Neutral"
            st.write("Failed to fetch news data.")

        # ------------------------------
        # Combine sentiment into final score
        # ------------------------------

        sentiment_factor = {
            "Positive": 1.1,
            "Neutral": 1.0,
            "Negative": 0.9
        }

        overall_score = (
            (value_score + quality_score + momentum_score) / 3
        ) * sentiment_factor[sentiment_verdict]

        final_verdict = "GOOD" if overall_score >= 70 else "NEUTRAL" if overall_score >= 40 else "BAD"

        st.success(f"**FINAL Overall Verdict (Including Sentiment):** {final_verdict} ({overall_score:.1f}/100)")

    except Exception as e:
        st.error(f"Error fetching data: {e}")
