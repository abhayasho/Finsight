# Finsight

Finsight is a local stock research dashboard for Windows, macOS, and Linux. Explore company fundamentals, price trends, risk, valuation assumptions, forecasts, machine learning buy evidence, and recent news in one place. Start with the included demonstration data without an API key, then choose live data when you are ready.

The expanded models expose their assumptions and data coverage. Their output is research support: more calculations do not establish predictive accuracy or guarantee an investment outcome. See the **Methodology** tab for the calculations and limitations.

## Start here

1. Install **Python 3.12** from [python.org](https://www.python.org/downloads/). Python 3.11 and 3.13 are also supported. On Windows, enable **Add Python to PATH** if the installer offers it.
2. Download and **extract the entire Finsight ZIP** into a folder you can write to, such as Documents. Do not run it from inside the ZIP.
3. Open the extracted folder and double-click the launcher for your computer:

   | Computer | Open this file |
   | --- | --- |
   | Windows | `Start Finsight.bat` |
   | macOS | `Start Finsight.command` |
   | Linux | Run `python3 launch.py` in a terminal |

The first launch installs the core packages into a private `.venv` folder and needs an internet connection. Later launches reuse that environment. A browser opens at [http://127.0.0.1:8501](http://127.0.0.1:8501). Keep the launcher window open while using Finsight; press **Ctrl+C** there to stop it. No paid service, NewsAPI key, Ollama installation, or AI model download is required for the demonstration.

If macOS opens the launcher as text or prevents it from opening, use Terminal in the extracted folder and run `python3 launch.py`. For setup help, see [the installation guide](docs/INSTALL.md).

## What you can explore

| Area | Purpose |
| --- | --- |
| Overview and factors | Compare valuation, profitability, growth, financial health, and momentum with visible coverage |
| Risk | Examine volatility, drawdowns, benchmark sensitivity, and return statistics |
| Valuation | Change discounted cash flow assumptions and inspect scenario sensitivity |
| Forecast | Compare simple forecasting models with time ordered validation and uncertainty |
| ML buy models | Train logistic regression, random forest and gradient boosting; inspect calibrated probabilities and held-out performance |
| News | Read source linked headlines and optional sentiment estimates |
| Should you buy? | Get a conditional Buy, Wait, Do not buy, or Insufficient data assessment with reasons and every rule disclosed |
| Exports | Download analysis JSON and price history CSV for further review |

The six-factor model combines 18 inputs for ordinary companies with continuous scores and explicit missing-data coverage. The forecast lab compares a random walk, historical drift, EWMA drift, and ridge autoregression at the selected horizon. Valuation adds bear/base/bull equity cashflow scenarios and a sensitivity table. See [the model guide](docs/MODELS.md) for formulas, assumptions, and exclusions.

The final **Should you buy?** assessment is now driven primarily by supervised machine learning. It learns a defined positive-net-return and benchmark-outperformance target over 21, 63 or 126 sessions. Three classifiers compete with a class-prior baseline using separate, purged chronological training, calibration, validation and final-test periods. Only a qualified model can support a Buy, Wait or Do not buy result; failed tests produce an explicit **No reliable ML buy signal** assessment. Current risk and financial evidence can veto or moderate a positive signal. The earlier factor/valuation rules remain visible as separate context, not learned predictions.

Training runs automatically when you analyze a company and is cached on later dashboard interactions. No GPU or paid AI service is needed. Live analysis requests available history; the ML engine uses at most the latest 20 years. Current fundamentals and headlines do not enter historical training because their point-in-time history is unavailable. See [the ML guide](docs/ML.md) for the exact target, costs, performance gates, frozen-model dates and limitations.

Live market information depends on Yahoo Finance through `yfinance`, can be delayed or incomplete, and may be unavailable because of rate limits. The demonstration uses synthetic data and is clearly labeled. It is useful for learning the interface, not researching actual stock performance.

## Optional news and AI

For NewsAPI headlines, copy `.env.example` to `.env`, enter your own `NEWS_API_KEY`, and restart. Keep that file private. Existing `secrets.txt` installations are also supported. News availability and use are subject to your provider's plan and terms.

The standard installation shows headlines without assigning sentiment. For optional local FinBERT inference, run:

```text
python launch.py --install-ai
```

On Windows, use `py -3.12 launch.py --install-ai` if `python` is unavailable. This installs PyTorch and Transformers into Finsight's environment and then exits. Start Finsight again and explicitly select FinBERT in the app. The first selected use downloads the model from Hugging Face; it can require substantial disk space and time. The app works without these optional packages. See [privacy and data](docs/DATA_AND_PRIVACY.md).

Existing Ollama users can also request a local snippet summary in the News tab. Install an Ollama model such as `mistral` separately and start its local service. Summaries are optional and do not change scores or forecast outputs.

## Other ways to run

```text
python launch.py --check
python launch.py --port 8502
python launch.py --no-browser
```

`--check` validates your Python version and project files without installing packages or starting a server. The launcher binds to `127.0.0.1` so the default desktop session is accessible from your computer.

For a manually managed environment:

```text
python -m venv .venv
```

Then use `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux:

```text
<environment-python> -m pip install -r requirements.txt
<environment-python> -m streamlit run app.py --server.address=127.0.0.1
```

No environment activation is necessary. See [the Docker and developer guide](docs/DEVELOPMENT.md) for container use and tests.

## Compatibility and validation

The project targets 64 bit Python 3.11–3.13. CI is configured to install dependencies, run tests, and start the server on Windows, macOS, and Ubuntu for each supported Python version. A configured workflow is not evidence that every platform has passed: check the repository's **Actions** results for an actual run. Native Windows behavior must be confirmed there or on a Windows machine.

Core dependencies have version ranges; they are not a locked, reproducible package snapshot. A first install resolves compatible releases. The launcher reinstalls requirements after their contents change; `python launch.py --reinstall` requests a fresh dependency resolution when needed.

## License

[MIT](LICENSE). Third party software, data, news, and optional model weights have their own licenses and terms.
