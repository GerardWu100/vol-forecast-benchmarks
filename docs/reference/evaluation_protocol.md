# Evaluation Protocol

## Purpose

Stage 4 compares variance forecasts on dates that occur strictly after each
model's training data. The portable benchmark uses one calendar year of initial
training because the checked-in feature matrix spans 2022-11-02 through
2024-12-20. This produces 286 out-of-sample dates for each model and horizon.

## Expanding-Window Schedule

Let $t_0$ be the first feature date and let $Y$ be the configured number of
initial calendar years. The requested first evaluation date is

$$
t_{\mathrm{cutoff}} = t_0 + Y\text{ calendar years}.
$$

The evaluator selects the first observed feature date on or after
$t_{\mathrm{cutoff}}$, subject to at least 25 training rows. It fits each model
on every row before that date, forecasts the next 21 trading-day block, expands
the training sample, and repeats.

If no observed feature date satisfies the cutoff, the evaluator raises
`InsufficientTrainingHistoryError`. It reports the symbol, horizon, row count,
available date range, and requested cutoff. Empty forecast and score tables are
not valid benchmark evidence.

## Targets

Let $RV_t$ denote close-to-close realised variance on trading day $t$. The
one-day target is

$$
y_{t,1}=RV_{t+1}.
$$

The five-day target is the arithmetic mean of the next five daily variances:

$$
y_{t,5}=\frac{1}{5}\sum_{j=1}^{5}RV_{t+j}.
$$

The five-day targets overlap across adjacent forecast origins. This serial
dependence matters when interpreting standard errors and Diebold-Mariano tests.

GARCH(1,1) forecasts use the same target convention. The one-day conditional
variance is updated from the return observed at the forecast origin. For a
multi-day horizon, later expected variances follow the GARCH recursion with
persistence $\alpha+\beta$, and the model returns their arithmetic mean rather
than reusing a one-day forecast.

## Loss Functions

For $n$ out-of-sample observations, actual variance $y_t>0$, and forecast
$\widehat y_t>0$, mean squared error is

$$
\operatorname{MSE}=\frac{1}{n}\sum_{t=1}^{n}(y_t-\widehat y_t)^2.
$$

The quasi-likelihood loss used by the project is

$$
\operatorname{QLIKE}=\frac{1}{n}\sum_{t=1}^{n}
\left[\log(\widehat y_t)+\frac{y_t}{\widehat y_t}\right].
$$

Lower values are better for both scores. QLIKE and MSE may rank the same
forecasts differently because QLIKE is sensitive to the actual-to-forecast
variance ratio, whereas MSE squares absolute errors in variance units.

## Portable and Extended Research Settings

- Portable default: one initial calendar year, 21-trading-day retraining, SPY,
  and the checked-in 2022-2024 cache.
- Extended research: a longer initial period is allowed only with additional
  earlier raw data. The evaluator applies the same feasibility check.

The project has one configuration surface rather than named production and
research profiles. The available data, not a label, determine whether a setting
can run.
