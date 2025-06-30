# Finsight

Finsight is a Streamlit-powered dashboard for analyzing stocks, combining:

✅ Financial fundamentals  
✅ Technical indicators (moving averages)  
✅ Valuation vs. industry averages  
✅ Hybrid news sentiment analysis using FinBERT and Mistral LLM  
✅ News trend visualization and summaries  
✅ Overall stock health scoring

All running locally and for free!

---

## 🎯 Features

- **Price Charts**  
  View historical prices plus 20-day and 50-day moving averages.

- **Financial Metrics**  
  Quickly check market cap, PE ratio, EPS, ROE, dividend yield, and more.

- **Industry Comparison**  
  Compare a stock’s valuation to industry averages.

- **Quant Factor Scoring**  
  Automatic scoring on:
  - Value
  - Quality
  - Momentum

- **Hybrid Sentiment Analysis**  
  - FinBERT scans news headlines for fast sentiment classification.
  - Mistral LLM provides nuanced analysis and news summaries.

- **News Sentiment Trend**  
  Visualizes how sentiment changes over time.

- **Overall Verdict**  
  Combines fundamentals and sentiment into a final GOOD / NEUTRAL / BAD score.

## 🛠 How to Run Locally

Clone the repo:

```bash
git clone https://github.com/abhayasho/Finsight.git
cd Finsight
pip install -r requirements.txt
ollama pull mistral
ollama serve

Run the Streamlit app:

streamlit run app.py


