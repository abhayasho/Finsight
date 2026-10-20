# Machine learning buy evidence

Finsight's ML lab trains supervised classifiers on the selected stock and benchmark's adjusted price histories. It estimates the probability of a specific future outcome over the chosen 21, 63 or 126 trading sessions. That probability is evidence for the final buy assessment, not a forecast of every aspect of the company or a promise of profit. More complex models are retained only when the held-out evidence supports them.

Live data requests the available price history; ML fitting uses at most the latest 5,040 stock sessions, approximately 20 trading years. Less history remains usable when the stated sample requirements can be met. Longer horizons leave fewer non-overlapping outcomes; a short listing history may prevent qualification even if many daily rows are available. Demo results use synthetic data and cannot establish real investment performance.

## What the model predicts

Features use prices available through the signal date's close. A simulated purchase occurs at the **next observed close**, with an exit one selected horizon later. This separates information time from execution time; it does not assume an order could execute at the close that produced its signal.

Let `stock_return = stock_exit / stock_entry - 1`, and compute `benchmark_return` over the same entry and exit dates. The positive label requires both:

- A positive stock return after the configured round-trip transaction cost.
- A stock return after that cost greater than the benchmark return.

The default round-trip cost is 20 basis points, or 0.20% of the invested amount. It is a simple subtraction from the holding-period stock return; it is not an execution simulation. Change it to reflect the intended trading conditions. Taxes, market impact, trading restrictions and financing costs are not modeled. The benchmark return is a comparison hurdle with no deducted trading cost.

The target is **positive net performance and benchmark outperformance**, not simply “the price will rise.” A stock can rise and still receive a negative target if it underperforms its benchmark or the assumed costs absorb the gain. A low estimated positive-target probability does not establish a profitable short trade. If no usable benchmark is available, the experiment explicitly switches to the positive-net-return target alone and cannot qualify to drive the final buy assessment.

## Inputs and candidates

The model uses 13 trailing features when a benchmark is present:

| Features | Definition |
| --- | --- |
| Returns over 5, 21, 63 and 126 sessions | Adjusted close divided by its earlier value, minus one. |
| Distances from 20, 50 and 200-session averages | Current adjusted close divided by the trailing simple average, minus one. |
| Volatility over 21 and 63 sessions | Sample standard deviation of log returns, annualized with `sqrt(252)`. |
| Drawdown over 63 sessions | Distance below the trailing maximum adjusted close. |
| 14-session RSI | RSI from simple trailing average gains and losses, scaled to 0–1; this differs from the dashboard's Wilder-smoothed RSI. |
| Relative returns over 21 and 63 sessions | Stock return minus benchmark return over the same trailing dates. |

It never places the target's future returns among the inputs. Current fundamentals, present-day analyst opinions and current news are excluded from historical model training: their historical availability cannot be reconstructed from today's provider response. A future fundamental or news model requires dated, point-in-time inputs before it can be tested honestly.

The candidates are:

| Candidate | Purpose |
| --- | --- |
| Training-prior baseline | Predicts the positive-label frequency observed in training; a learned model must improve on this simple reference. |
| Standardized logistic regression | A regularized linear probability classifier with `C=1` and at most 1,000 iterations; the scaler is fitted on training rows only. |
| Random forest | 100 trees with maximum depth 5 and minimum leaf size 20, running on one worker. |
| Histogram gradient boosting | At most 80 iterations, maximum depth 3 and minimum leaf size 20, with internal early stopping disabled. |

There is no large hyperparameter search or claim that a particular architecture is universally best. The app reports the evaluated candidates and their measured results for the selected history and settings.

## Validation and leakage controls

Rows stay in chronological order in four successive partitions, initially assigned **40% training, 15% calibration, 15% validation and 30% final test**. Purging reduces the earlier partitions' retained row counts. Training fits the feature preprocessing and classifier. A later, separate calibration period fits a sigmoid mapping from classifier output to probability. Validation selects the candidate with the lowest Brier score, using log loss to break a tie; the final test is reserved for assessing that selected candidate. The training-prior baseline uses only training label frequency.

At each boundary, rows whose forward outcome ends on or after the next partition's first feature date are purged. A label that needs a future exit price cannot be treated as known at an earlier partition's end. There is no random shuffle or random train/test split. Finsight also excludes the newest observations whose outcomes are not yet known from supervised training and evaluation; those can still supply current prediction features. At least 250 training rows and 50 rows in each later partition must remain. Training and calibration must each contain both outcome classes.

The classifier and calibrator used for current inference are the same frozen fitted objects evaluated on the final test. They are not refitted on the test outcomes after evaluation. Their training and calibration cutoff dates are disclosed: the classifier can be several years old, and it does not learn from more recent labels. That is a material limitation in a changing market. Rerunning with a different ticker, benchmark, horizon, cost or longer history creates another experiment. Repeatedly choosing settings after looking at their test results weakens the final-test claim.

These design choices follow scikit-learn's guidance on [avoiding data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) and fitting probability calibration on [data separate from classifier training](https://scikit-learn.org/stable/modules/calibration.html#calibrating-a-classifier). Sigmoid calibration can improve probability estimates; it does not guarantee they remain calibrated in a different market regime.

## Reading the evidence

Finsight compares probability predictions with the training-prior baseline using Brier score and log loss. Lower values are better. These evaluate overall probability quality; a lower Brier score alone does not establish superior calibration. Accuracy or a high percentage of rising prices can be misleading when one outcome dominates. The report includes sample counts and event rates so the baseline is visible. See scikit-learn's discussion of [probability scores and calibration](https://scikit-learn.org/stable/modules/calibration.html).

Daily historical labels overlap whenever the horizon exceeds one session: adjacent forecasts share most of the same holding period. Daily metrics are useful descriptive diagnostics, but their row counts must not be interpreted as independent trades. The lab separately selects observations with disjoint holding windows, using dates rather than predictions or label values. Eight such final-test observations are still a small sample, and these observations can remain correlated through market conditions.

Qualification requires a validation improvement over the baseline and all of the following final-test checks:

- At least 2% relative Brier-score improvement over the training-prior baseline on daily test rows.
- ROC AUC greater than 0.5 on daily test rows.
- At least eight non-overlapping outcomes, including at least two of each target class.
- At least five non-overlapping positive calls at the fixed 60% probability threshold.
- A 95% Wilson lower bound for positive-call precision above the training positive-label frequency.
- Positive mean stock return after costs minus gross benchmark return among those non-overlapping positive calls.

These are fixed conservative research gates, not a proof of future profitability or formal independence. The Wilson calculation is a descriptive uncertainty check whose assumptions are weakened by serial dependence. The 60% positive and 40% negative call thresholds are fixed; the final test does not tune them.

The economic diagnostics compare the model's positive calls with buying the same stock in every non-overlapping evaluation slot. They report trade count, average stock return after costs, average excess return over the gross benchmark, win rate and target-outcome precision. A skipped slot stays in zero-return cash. Average return per slot includes those skipped slots; average return per selected trade does not. These are unannualized descriptive averages, not a compounded portfolio backtest. They omit cash interest and realistic account-level execution constraints.

A model that fails the final-test requirements does not get promoted into a confident buy recommendation. The app can report no demonstrated ML edge, insufficient history or another abstention reason. That means the available experiment did not support the required conclusion; it does not prove there can never be an edge. The final assessment combines qualified ML evidence with the application's other purchase checks and explains the result.

## How ML affects “Should you buy?”

The primary final assessment uses the qualified classifier for the selected horizon. The previous factor-and-valuation assessment remains visible as separate fundamental context; its fixed score thresholds are not presented as learned evidence.

| Final result | Conditions |
| --- | --- |
| Buy | Qualified model probability at least 60%, with required current data and risk evidence, no severe-risk veto and no financial conflict listed below. |
| Wait | Qualified model probability strictly between 40% and 60%, or positive ML support accompanied by a financial conflict. |
| Do not buy | Qualified model probability at most 40%, or a historical risk score below 25/100. This assesses a new purchase and does not instruct an existing holder to sell. |
| Insufficient data | Model qualification failed, required data is missing or stale, or another required check fails. The title is “No reliable ML buy signal.” |

A substantive Buy, Wait or Do not buy result requires a positive current quote, known quote currency, explicit equity instrument metadata and quote and history dates within seven calendar days. Model features must match the latest supplied price session, and historical risk evidence must be available. Missing requirements take precedence over the other verdicts.

A positive ML signal becomes Wait if supplied trailing earnings, reported net income or book equity per share is nonpositive, or if the current quote is at least 20% above an available base cashflow scenario. These are current financial-context checks; they were not used as historical training features. An unavailable cashflow valuation alone does not block a price-model result. Banks and other businesses requiring specialized valuation still need that additional financial review.

The model does not know the user's portfolio, liquidity needs, tax situation or tolerance for loss. Its probability concerns the specified target and execution assumptions; it is not a personal suitability score.

Historical test results are limited to the selected surviving stock, benchmark, price provider, date range and execution assumptions. They are not a cross-sectional stock-selection backtest, a survivorship-bias-free research universe or a live track record. Adjusted data may be revised; actual fills, liquidity, spreads and slippage can differ. A model that performed well historically can fail on the next prediction. This feature does not use a language model to invent a trading signal.

## Local setup and compatibility

The core requirements include scikit-learn. Training and prediction run locally on CPU; no GPU, paid model endpoint or AI API key is required. The ordinary launcher installs dependencies in the project's virtual environment, including on Windows. Supported Finsight Python versions are 3.11–3.13; pip selects compatible dependencies for the interpreter and platform. Use a 64-bit Python installation. Prebuilt package availability depends on the interpreter and platform; see the official [scikit-learn installation instructions](https://scikit-learn.org/stable/install.html).

Do not copy a virtual environment between Windows, macOS or Linux machines. Extract the project and use its launcher on each machine so that native dependencies are installed for that system. Current model results are cached for responsiveness; changing the experiment settings invalidates the corresponding cached analysis.
