# Models and interpretation

Finsight combines transparent research rules with price-model experiments. Increasing the number of models does not prove higher forecasting accuracy. Factor scores are not fitted to returns; their fixed anchors are subjective. Inspect the evidence and coverage before interpreting any number.

## Six-factor research model

| Factor | Target weight | Evidence |
| --- | --- | --- |
| Value | 20% | Earnings yield, reported free cashflow yield, EV/EBITDA; price/book replaces enterprise cashflow metrics for financial firms |
| Quality | 20% | ROE, ROA, operating margin and cashflow margin; financial firms use net margin |
| Growth | 15% | Provider revenue growth and earnings growth |
| Momentum | 15% | Adjusted returns over 63 and 126 sessions and distance from the 200-session moving average |
| Financial strength | 15% | Current ratio, debt/EBITDA, net debt/market cap; accounting equity/assets for financial firms when supplied |
| Historical risk | 15% | Realized volatility, drawdown magnitude and historical tail loss |

Each input is linearly mapped between disclosed anchors and clipped to 0–100. A low-is-better input reverses that mapping. Factor scores average only available input weights. Overall contributions use `factor target weight × factor coverage`, then renormalize. Missing data receives no neutral score. Overall scoring abstains below 50% weighted coverage, fewer than three available factors, or fewer than two fundamental factors. The Factors tab displays each input, units, anchors, weight and missing-data reason; JSON exports contain the complete evidence.

Non-equity instruments have no company-fundamental score. Banks and insurers exclude generic cashflow, enterprise-value and debt/liquidity measures; their accounting equity/assets measure is not regulatory capital. Regulatory capital, reserves, loan losses and asset quality require a specialist model. Nonpositive reported book equity excludes ROE; nonpositive current earnings excludes percentage earnings growth. Quote/financial currency mismatches exclude monetary cross-currency ratios rather than making an implicit FX conversion. Other sector differences and negative comparison-year earnings remain limitations.

## Prices and historical risk

Yahoo prices are explicitly requested with split/dividend adjustments. Invalid prices and duplicate sessions are removed; missing rows are never forward filled. Quote prices remain separate from adjusted daily closes. One, three, six and twelve month returns use 21, 63, 126 and 252 observed sessions. Moving averages use 20, 50 and 200 sessions; RSI uses Wilder's 14-session smoothing.

Risk uses at most the trailing 756 price observations and requires at least 62 returns for distribution estimates. Volatility is the sample daily return standard deviation multiplied by `sqrt(252)`. Downside volatility is the root mean square of negative returns, annualized by `sqrt(252)`. Maximum drawdown is the minimum of `price / previous running high − 1`; it is displayed as a negative return. Historical 95% VaR is the nonnegative loss corresponding to the 5th return percentile; CVaR averages the returns at or below that percentile and reports nonnegative loss. These are one-session historical estimates, not limits on future losses.

Beta aligns price dates before computing common-interval returns, then divides sample covariance by benchmark sample variance; it requires 60 aligned returns. Benchmark charts index both histories to 100 on their first common date. They do not convert currencies or account for investor-specific taxes, costs or reinvestment restrictions. A relevant benchmark is a user input.

## Equity cashflow scenarios

The provider's `freeCashflow` is an operating-cashflow-minus-capex proxy. Finsight treats it as cashflow to equity under an explicit **zero net borrowing** assumption. It is discounted at the user's **required equity return**, with no additional debt subtraction or cash addition. This avoids mixing enterprise and equity valuation bases. It is a simplified illustration, not a detailed FCFE forecast.

Projected annual cashflows are `FCF0 × (1 + growth)^year`. Terminal value after the explicit period is `FCF_final × (1 + terminal_growth) / (required_equity_return − terminal_growth)`. Discount both the projected cashflows and terminal value to today, then divide by shares outstanding in full units. Bear/base/bull scenarios vary growth and discount rates; the sensitivity table exposes terminal-rate effects. Invalid scenario cells remain unavailable rather than being repaired to plausible numbers.

The required return must exceed perpetual growth. Positive cashflow and shares are required; financial businesses, funds and currency mismatches abstain. Missing currency metadata is explicitly disclosed. The model omits changing financing, detailed reinvestment, dilution and statement reconciliation. Periods in provider fields may differ. Valuation does not alter the research score.

## Forecast lab

Four fixed candidates model adjusted log prices: a random walk, full-history mean log-return drift, fixed 20-session half-life EWMA drift, and ridge AR(5) on returns. Ridge uses five prior returns, training-only standardization, an unpenalized centered intercept and fixed penalty 10. Its recursive return guard derives from training data.

For each historical origin, every model fits only that origin's preceding history and predicts the **requested horizon**. There is no random train/test split or future-fitted scaling. Up to 60 expanding-window origins report endpoint MAE, RMSE, percentage error and directional accuracy when a nonflat direction was predicted. The random walk makes no directional call. Origins can overlap, so errors are correlated. Model selection reuses these comparisons: these are not independent final-test results. The lowest-RMSE candidate is retained only if it improves over the random walk by at least 2%; otherwise the baseline remains selected. This rule is not a statistical significance test.

The future scenario uses 2,000 seeded moving-block bootstraps of centered residuals, in five-session blocks. AR paths respond recursively to their own shocks. The 80% and 95% bands are empirical simulation quantiles, **not calibrated predictive guarantees**; they omit parameter uncertainty and structural breaks. Horizons count observed trading sessions, not exact calendar dates. At least `max(100, 60 + horizon + 4)` observations are required, with additional warnings for small validation samples. Neither current fundamental data nor current news enters historical forecast validation.

## Optional news models

NewsAPI provides dated snippets; removed, duplicate, invalid, stale and future articles are excluded. Without the optional AI install, sentiment is explicitly unscored. FinBERT computes all three label probabilities and uses tokenizer-aware 512-token truncation. Aggregate tone is `P(positive) − P(negative)`, weighted by publication age with a seven-day half-life. Confidence is an uncalibrated class probability, not future-return accuracy. Daily trend points average the available article tones for that day.

Optional Ollama summaries use source snippets as delimited input and request source citations; they may still hallucinate. Neither sentiment nor LLM summaries changes the research score or numeric forecast. Missing news, credentials, models or network access never becomes an artificial neutral opinion.

## Final: should you buy?

The primary final assessment now uses the supervised classifiers and final-test qualification described in [the ML guide](ML.md). A prediction must earn decision eligibility; a high probability alone cannot authorize a purchase. Current financial evidence and historical risk provide additional checks. The same ML result and current valuation context appear in Overview, Should you buy? and JSON exports.

## Separate fundamentals-only comparison

The earlier fundamentals-only comparison uses the current quote, factor evidence and selected cashflow assumptions. It is now shown separately inside the final assessment. Its fixed thresholds below are not machine learning predictions. Both approaches assess a new purchase; a Do not buy result is not an instruction to sell an existing holding.

| Result | Rule |
| --- | --- |
| Buy | Overall score at least 70, quality at least 65, financial strength at least 60, risk and value each at least 50, and margin of safety at least 20%; no reported nonpositive earnings or equity |
| Wait | Sufficient supported evidence, but the Buy conditions are not met and no Do not buy rule is triggered |
| Do not buy | Sufficient supported evidence with overall below 40, quality below 35, financial strength below 30, risk below 25, or quote at least 20% above the base scenario value |
| Insufficient data | Missing, stale, invalid or unsupported evidence; this is not a negative judgment on an unmodeled investment |

All results first require explicit equity metadata, a non-financial operating company, matching known quote/financial/valuation currency units, positive cashflow and current quote, available positive bear/base/bull valuations, all six valid factor scores and at least 85% overall coverage. Minimum factor coverage is 80% for value, 75% for quality, two thirds for financial strength and 100% for risk. This prevents a complete price-history factor from disguising missing business evidence. Quote and daily-history dates must be within seven calendar days; the reported-quarter end must be within 180 days. Missing and future dates abstain. A retrieval timestamp is never substituted for a quote or statement date.

Margin of safety is `1 − quote / base scenario value`. At 20%, the quote must be no more than 80% of base value; that is 25% modeled upside, not a promised return. The displayed buy-price condition is conditional on all other evidence and assumptions continuing to hold. The simplified valuation can be wrong, so the rule does not establish actual intrinsic value. Banks, ETFs, negative-cashflow companies and missing valuations need other assessment methods; this model abstains.

Thresholds are subjective research rules and have not been validated as a trading strategy. The price Forecast lab, AI headlines and language-model summaries do not vote on this comparison or the primary ML assessment. The separately validated ML buy classifier provides the primary signal. Demo conclusions explicitly say synthetic and are not actionable. The model does not know a user's portfolio, goals, time horizon or risk tolerance; those affect personal suitability ([Investor.gov](https://www.investor.gov/introduction-investing/investing-basics/investment-products)).

## References

- [Damodaran: cashflows, discount rates and valuation](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/val.html).
- [Damodaran: estimating terminal value](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/valquestions/termvalapproaches.htm).
- [Time-ordered validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- [FinBERT model card and label meanings](https://huggingface.co/yiyanghkust/finbert-tone).
- [yfinance and data usage](https://github.com/ranaroussi/yfinance), [NewsAPI Everything endpoint](https://newsapi.org/docs/endpoints/everything).
