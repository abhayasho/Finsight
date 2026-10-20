# Validation for the 3.0 ML update

Checked locally on macOS with Python 3.12 and scikit-learn 1.9.1 on October 19, 2026:

- **270 automated tests passed.** Coverage includes actual logistic, random-forest and gradient-boosting fits; exact next-close labels and costs; causal features; all chronological purge boundaries; held-out outcome changes that cannot alter model selection or fitted predictions; non-overlapping trade calculations; unsupported high-probability calls; data and currency checks; UI interactions and strict JSON exports.
- The full dashboard trains the ML pipeline on synthetic demo data without network or API keys. The real local server and ML interface were also checked in a browser. Demo metrics are explicitly fictional and never establish investment performance.
- A live MSFT/SPY experiment fetched 10,229 stock sessions and 8,488 benchmark sessions. The ML engine limited its input to the latest 5,040 stock sessions. At the default 63-session horizon and 20 bps round-trip cost, validation selected logistic regression. The untouched final test contained 1,434 daily labels and 23 non-overlapping outcomes. Selected-model Brier loss was 0.3358 versus 0.2454 for the class-prior baseline, so the system correctly withheld a buy/no-buy signal. This single experiment does not establish general model accuracy or profitability.
- The live experiment's frozen base classifier trained on feature dates through 2014-11-18; calibration outcomes ended 2017-12-26; final outcomes extended through 2026-10-19. This age limitation is displayed prominently. The exact tested fitted objects generate current estimates; the system does not silently refit them after inspecting the final test.
- Open-session provisional daily bars are excluded when provider market-state and quote metadata identify them. Current quotes remain separate from completed adjusted closes. Both open- and closed-session cases are tested.
- Launcher diagnostics, syntax and Git whitespace checks passed. Setup uses local environments, handles paths with spaces and includes Windows/macOS entrypoints.

The test run emits one harmless joblib physical-core detection warning in the sandbox and falls back to logical-core detection. ML work itself is constrained to one CPU thread.

The Windows/macOS/Ubuntu × Python 3.11/3.12/3.13 workflow is included. Native Windows execution and the matrix have not yet run for this update because GitHub publishing still awaits approval. Docker and real FinBERT/Ollama inference have not been run; optional AI behavior is covered with mocked outputs and failures.

The classifier experiment is not a survivorship-free universe backtest, a personal suitability model or a live track record. Its fixed qualification rules are not formal proof of future profitability. See [ML.md](ML.md) for target semantics, costs, sample limitations, serial dependence and frozen-model caveats.

Repeat automated checks with `python -m pytest -q` after installing `requirements-dev.txt`. See [the developer guide](DEVELOPMENT.md) for the platform workflow.
