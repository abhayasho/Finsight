# Data and privacy

Finsight serves its desktop interface on `127.0.0.1` by default. A local interface does not make upstream data downloads offline: the following connections can occur when you request those features.

| Feature | External service and information sent |
| --- | --- |
| First setup or package refresh | Python package indexes receive package installation requests and ordinary connection metadata. |
| Live stock analysis | Yahoo Finance, accessed through `yfinance`, receives requested stock and benchmark symbols and connection metadata. |
| ML classifiers | Features, training, probability calibration and inference run locally on CPU; no model API receives the stock history. |
| Optional NewsAPI headlines | NewsAPI receives the news query and your NewsAPI credential. |
| Explicit FinBERT use | Hugging Face receives model download requests and connection metadata. After loading, headline inference runs in the local Python process. |

Streamlit usage statistics are disabled in the included configuration and launcher. Your browser and operating system have their own network and privacy behavior.

No NewsAPI key is required to start the app. To add your own, copy `.env.example` to `.env`, fill in the key, and restart. Local `.env`, `secrets.txt`, and `.streamlit/secrets.toml` files are excluded from Git and Docker builds. Git ignore rules do not prevent accidental disclosure if you manually share an entire working folder. Share a clean project download without local credentials or `.venv`.

Market data may be delayed, adjusted, restated, missing, or incorrect. Financial ratios can use different reporting dates and definitions. Check the original filings and data dates before acting. Yahoo Finance, NewsAPI, and model providers impose their own licenses, terms, and access limits; Finsight does not grant permission to redistribute their data.

The optional local sentiment model classifies wording. It does not verify a headline, infer investment suitability, or predict returns. The synthetic demonstration is labeled and does not represent historical company performance. Valuation scenarios and forecast intervals depend on assumptions and are not calibrated guarantees of future results.

References: [yfinance project and data usage notice](https://github.com/ranaroussi/yfinance), [NewsAPI documentation](https://newsapi.org/docs), [Streamlit configuration](https://docs.streamlit.io/develop/api-reference/configuration/config.toml), and [Hugging Face Transformers installation](https://huggingface.co/docs/transformers/installation).
