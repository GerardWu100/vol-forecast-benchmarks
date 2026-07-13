# Adaptive outline: volatility forecasting when the benchmark cannot start

## Archetype decision

- Problem: compare realised-variance forecasts while preserving information timing and chronological evaluation.
- Options considered: `strategy-backtest`, `risk-model`, and `mixed`.
- Choice: `mixed`, combining the risk-model and empirical benchmark blueprints.
- Why: the repository defines variance estimators and losses, but its main analytical task is an out-of-sample comparison of forecasting models rather than a trading strategy.
- Verify: the draft must connect every claimed result to the unchanged default run or to the explicitly labelled one-year sensitivity run.

## Section blueprint

1. **A benchmark can fail before a model does**
   - State the forecasting question and the default-run finding: 537 usable rows, zero out-of-sample forecasts.
2. **What is being forecast**
   - Define intraday close-to-close realised variance and the one-day/five-day targets.
   - Explain daily, weekly, and monthly Heterogeneous Autoregressive Realised Variance (HAR-RV) features.
3. **Information timing is part of the model**
   - Trace the one-business-day lag on option and VIX features.
   - Describe the expanding-window walk-forward schedule.
4. **Why the default leaderboard is empty**
   - Compare the 537-row feature span with the configured five-calendar-year initial training period.
   - Include a coverage/burn-in graph and explain why an empty score table is correct behavior.
5. **A one-year sensitivity run**
   - Change only `initial_train_years` in a copied in-memory configuration.
   - Compare HAR-RV, GARCH(1,1), Ridge, Lasso, and XGBoost using QLIKE.
   - Include a model-comparison graph and numerical table from frozen derived data.
6. **What the losses do and do not say**
   - Interpret QLIKE differences, floor-hit diagnostics, and Diebold-Mariano comparisons.
   - Avoid presenting the sensitivity run as the repository's configured result.
7. **What I would change before treating this as research evidence**
   - Reconcile data history and burn-in, widen the asset/time sample, test retraining choices, and examine stability across volatility regimes.

## Planned equations

1. Intraday log return: $r_{t,i}=\log(P_{t,i}/P_{t,i-1})$, where $P_{t,i}$ is minute $i$'s closing price on day $t$.
2. Daily realised variance: $RV_t=\sum_{i=1}^{M_t}r_{t,i}^2$, where $M_t$ is the number of intraday returns.
3. Five-day target: $y_{t,5}=\frac{1}{5}\sum_{j=1}^{5}RV_{t+j}$; the one-day target is $y_{t,1}=RV_{t+1}$.
4. HAR-RV regression: $\widehat y_{t,h}=\beta_0+\beta_d RV_t^{(d)}+\beta_w RV_t^{(w)}+\beta_m RV_t^{(m)}$, with all terms defined before use.
5. QLIKE per observation: $L_t=\log(\widehat y_t)+y_t/\widehat y_t$, where lower average loss is better for the same target series.
6. Diebold-Mariano loss difference: $d_t=L_{a,t}-L_{b,t}$, with a positive mean indicating higher loss for model $a$.

## Planned code excerpts

- The shifted rolling-window construction in `add_har_lags` to show that rolling summaries exclude day $t$.
- The business-day shift used for option-chain and VIX alignment.
- A short sensitivity-run block showing that the configuration is copied before setting the one-year burn-in.

## Planned graphs

1. `images/01_realised_volatility.png`: annualised SPY realised volatility and its 22-day trailing mean. Takeaway: volatility is clustered, so chronological evaluation and regime coverage matter.
2. `images/02_qlike_vs_har.png`: QLIKE difference from HAR-RV for each model and horizon in the one-year sensitivity run. Takeaway: compare relative loss within a horizon, not raw MSE magnitudes across horizons.
3. `images/cover.png`: a generated editorial cover depicting layered volatility horizons, forecast paths, and an honest chronological boundary without text or logos.

## Known gaps and assumptions

- The default stage-4 outputs are empty because the data cover about two years while the initial window is five years.
- The one-year run is diagnostic sensitivity analysis, not the configured benchmark.
- The packaged sample covers SPY only, with VIX as market context.
- No transaction-cost or trading-performance claim is appropriate because this is a forecast benchmark, not a strategy backtest.
- The option data and underlying raw cache are accepted as supplied; the post does not independently validate vendor construction.

## Deployment note

The canonical workspace is `vol-forecast-benchmarks/blog/`. The ordinary publish bundle would be `~/projects/website/content/post/volatility-forecast-benchmarks/`, containing only the two Markdown files and referenced images. The user explicitly deferred publication, so this task will not create that bundle, run Hugo, commit website files, or push the website repository.

## Outline review

The outline was checked against the four coverage requirements: context, methodology, evidence, and practical limitations are all present. The central finding is not buried beneath the sensitivity analysis. Equations are limited to quantities implemented in the repository, and both planned charts can be regenerated from checked-in raw data with a blog-local script.
