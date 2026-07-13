---
title: "When a Volatility Benchmark Has No Forecasts"
description: "An audit of a leakage-aware SPY realised-variance benchmark, why its default walk-forward run is empty, and what a one-year sensitivity run can still teach us."
date: 2026-07-13
image: images/cover.png
categories: ["Quantitative Research", "Risk Management"]
---

A model leaderboard is comforting. It is also the wrong place to begin.

I ran this repository's offline pipeline unchanged, from minute bars through feature construction and walk-forward evaluation. The first two computational stages worked: 564 daily realised-variance rows became 537 model-ready rows, covering 2022-11-02 through 2024-12-20. Stage 4 then produced zero forecasts and zero scores.

That empty result is correct. The configuration asks for five calendar years of initial training data, but the feature matrix contains a little over two years. There is no valid first forecast date. Silently shortening the training window would make a prettier demonstration and a less honest benchmark.

This post follows the data all the way to that boundary. I then run a separate one-year sensitivity analysis, without changing the repository configuration, to examine the model comparison that the default sample cannot support.

## What is being forecast?

Let $P_{t,i}$ be the closing price of minute $i$ on trading day $t$. The intraday log return is

$$
r_{t,i}=\log\left(\frac{P_{t,i}}{P_{t,i-1}}\right).
$$

If day $t$ contains $M_t$ intraday returns, close-to-close realised variance is

$$
RV_t=\sum_{i=1}^{M_t}r_{t,i}^2.
$$

Here, $RV_t$ is a daily variance estimate in squared decimal-return units. The pipeline keeps a day only when it has at least 300 minute bars. It also computes the Parkinson range estimator and the Garman-Klass open-high-low-close estimator, although close-to-close realised variance is the forecast target.

At a feature date $t$, the one-day target is

$$
y_{t,1}=RV_{t+1},
$$

and the five-day target is the mean of the next five observations,

$$
y_{t,5}=\frac{1}{5}\sum_{j=1}^{5}RV_{t+j}.
$$

The graph below converts daily variance to annualised volatility as $100\sqrt{252RV_t}$, where 252 is the assumed number of trading days per year. The pale line is noisy by construction. Its 22-day trailing mean makes the persistence easier to see.

![Annualised SPY realised volatility and its 22-day trailing mean](images/01_realised_volatility.png)

Volatility clusters. Quiet and turbulent observations arrive in blocks rather than as independent draws. A random train-test split would mix those blocks. The project instead trains on the past and forecasts the next chronological segment.

## Three horizons in one regression

The Heterogeneous Autoregressive Realised Variance model, usually shortened to HAR-RV, represents persistence with daily, weekly, and monthly summaries. Define the three inputs available on feature date $t$ as

$$
x_{d,t}=RV_{t-1},
$$

$$
x_{w,t}=\frac{1}{5}\sum_{j=1}^{5}RV_{t-j},
$$

and

$$
x_{m,t}=\frac{1}{22}\sum_{j=1}^{22}RV_{t-j}.
$$

For horizon $h$, measured in trading days, the regression is

$$
\widehat y_{t,h}=\beta_0+\beta_d x_{d,t}+\beta_w x_{w,t}+\beta_m x_{m,t},
$$

where $\widehat y_{t,h}$ is the predicted realised variance and $\beta_0$, $\beta_d$, $\beta_w$, and $\beta_m$ are fitted coefficients.

The lag is visible in the implementation. The rolling window is built only after shifting the variance series by one row:

```python
shifted_series = lagged_df[column].shift(1)
rolling_mean = shifted_series.rolling(window=lag_window, min_periods=lag_window).mean()
lagged_df[lag_column_name] = rolling_mean
```

The benchmark adds four alternatives. Generalized Autoregressive Conditional Heteroskedasticity with one return and one variance lag, written GARCH(1,1), models conditional variance from returns. Ridge and Lasso regressions are fitted to log variance. The final model is an XGBoost tree ensemble. Ridge and Lasso use all 14 features, including realised-variance histories, option-implied volatility, option skew, the implied-minus-realised spread, and VIX.

## Information timing is part of the model

An accurate formula can still produce a dishonest backtest if its timestamps are wrong. This pipeline treats an option-chain observation or VIX close from day $t-1$ as available on feature date $t$. It enforces that convention by moving each observation forward one business day before merging:

```python
vix_lagged["date"] = pd.to_datetime(vix_lagged["date"]) + pd.offsets.BDay(1)
vix_lagged = vix_lagged.sort_values("date").reset_index(drop=True)
```

The same shift is applied to option features. Short gaps can be forward-filled for at most three days. Rows missing any required feature or target are then removed. Those rules turn 564 realised-variance rows into 537 complete feature rows.

Evaluation uses an expanding window. After the initial training period, each fitted model forecasts the next 21 trading days. The training sample then expands, the model is refitted, and the process repeats. Nothing from the test block enters the preceding fit.

## Why the configured leaderboard is empty

The feature sample begins on 2022-11-02. The evaluator adds the configured five-year training period to that date, putting the first possible cutoff in November 2027. The last feature row is dated 2024-12-20.

| Default-run audit | Value |
|---|---:|
| Daily realised-variance rows | 564 |
| Complete feature rows | 537 |
| Feature range | 2022-11-02 to 2024-12-20 |
| Configured initial training period | 5 calendar years |
| Forecast rows | 0 |
| Score rows | 0 |

The evaluator checks whether the cutoff exists. When it does not, it returns an empty forecast table with the correct schema. That behavior is safer than training on too little history without disclosure.

There is still a configuration problem. A portable benchmark should ship with data that can execute its default settings, or with default settings that fit the shipped data. As committed, this repository reproduces feature construction, not the advertised model comparison.

## A one-year sensitivity run

To inspect the evaluation machinery, I copied the loaded configuration in memory and changed one field in the copy:

```python
sensitivity_config = deepcopy(config)
sensitivity_config["forecast"]["initial_train_years"] = 1
```

The model set, 21-day retraining cadence, features, targets, and scoring code stayed unchanged. This produced 286 out-of-sample forecasts for every model at each horizon.

The primary score is Quasi-Likelihood loss, or QLIKE. For realised variance $y_t>0$ and forecast variance $\widehat y_t>0$, the per-observation loss is

$$
L_t=\log(\widehat y_t)+\frac{y_t}{\widehat y_t}.
$$

Lower average loss is better when models forecast the same target series. The absolute values below are negative because variance is recorded in small decimal units. Their ordering and differences are what matter.

| Horizon | Model | QLIKE | QLIKE vs. HAR-RV | MSE | Observations |
|---:|---|---:|---:|---:|---:|
| 1 day | Lasso | -8.8272 | -0.0301 | 5.9584e-09 | 286 |
| 1 day | Ridge | -8.8268 | -0.0296 | 6.2964e-09 | 286 |
| 1 day | HAR-RV | -8.7972 | 0.0000 | 2.8105e-09 | 286 |
| 1 day | GARCH(1,1) | -8.7519 | 0.0453 | 3.1568e-09 | 286 |
| 1 day | XGBoost | -8.6919 | 0.1052 | 3.9683e-09 | 286 |
| 5 days | Lasso | -8.8008 | -0.0309 | 2.3718e-09 | 286 |
| 5 days | Ridge | -8.7967 | -0.0268 | 2.2992e-09 | 286 |
| 5 days | HAR-RV | -8.7699 | 0.0000 | 1.6964e-09 | 286 |
| 5 days | GARCH(1,1) | -8.7471 | 0.0228 | 1.7151e-09 | 286 |
| 5 days | XGBoost | -8.6966 | 0.0733 | 2.4243e-09 | 286 |

![QLIKE differences from HAR-RV in the one-year sensitivity run](images/02_qlike_vs_har.png)

Lasso has the lowest QLIKE at both horizons, narrowly ahead of Ridge. HAR-RV has the lowest mean squared error (MSE) at both horizons. This is not a contradiction. MSE weights squared absolute errors, while QLIKE penalises the ratio between realised and predicted variance. The choice of loss determines what kind of miss matters most.

## Statistical comparison and forecast health

The project also computes pairwise Diebold-Mariano tests. For models $a$ and $b$, define the loss difference

$$
d_t=L_{a,t}-L_{b,t}.
$$

The null hypothesis is $E[d_t]=0$. A positive test statistic means model $a$ has higher average loss than model $b$. In the sensitivity run, the HAR-RV versus Lasso statistic is 2.0812 with a two-sided $p$-value of 0.0374 at one day. At five days, the statistic is 2.1698 and the $p$-value is 0.0300. Under this implementation's large-sample approximation, both comparisons reject equal QLIKE accuracy at the 5% level.

That statement needs restraint. Forecast horizons overlap at five days, the sample is one asset, and the result comes from an unplanned shorter burn-in. A $p$-value does not repair a weak research design.

The forecast diagnostics add one reassuring detail: none of the 2,860 sensitivity forecasts hit the positive variance floor. The linear models' log-target transformation avoided the clipping problem that can otherwise make variance regressions look stable for the wrong reason.

## What I would change next

First, I would reconcile the default configuration with the portable data. Keeping a five-year initial window means packaging substantially more history. Keeping the current history means selecting and documenting a shorter window before looking at model rankings.

Next, I would repeat the benchmark across more liquid underlyings and across several volatility regimes. SPY plus VIX is a useful demonstration, not evidence that one model family wins broadly. I would also vary the retraining cadence and initial window as predeclared sensitivity checks.

Finally, I would inspect the option features by timestamp and source coverage, not only after they have entered a regression matrix. A one-business-day lag prevents direct lookahead, but stale quotes, sparse expiries, and bounded forward-fills can still change what the variables mean.

The main result here is not that Lasso wins. It is that a sound evaluator refused to manufacture a test set. The sensitivity run tells us the pipeline can compare models once the window is feasible. The empty default run tells us the research specification still needs to catch up with its data.
