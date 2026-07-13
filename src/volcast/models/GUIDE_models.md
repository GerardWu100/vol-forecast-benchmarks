# Part 1: Conceptual Explanation

The `src/volcast/models/` folder contains benchmark forecasters, not a modeling
framework. Every model implements a minimal `fit`/`predict` interface so stage 4
can run the same walk-forward loop across model families.

The model groups are:

- **Academic baseline**: HAR-RV.
- **Conditional-variance baseline**: GARCH(1,1).
- **Feature-driven regressors**: Ridge, Lasso, and XGBoost.

For GARCH(1,1), let $r_t$ be the return observed at forecast origin $t$ and let
$\sigma_t^2$ be conditional variance. The one-day update is

$$
\sigma_{t+1}^2=\omega+\alpha r_t^2+\beta\sigma_t^2.
$$

For a horizon longer than one day, future expected variances recurse with
persistence $\alpha+\beta$. The model averages the next $h$ conditional
variances so its output matches the project's $h$-day mean-variance target.

This folder is intentionally narrow. It does not build features, define targets,
or score forecasts. It only maps prepared inputs to positive variance
predictions.

Linear models fit in log-variance space: if variance target is $\sigma_t^2 > 0$,
the model fits $\log(\sigma_t^2)$ and maps back with exponential transform. This
reduces negative raw predictions and hard-floor clipping artifacts.

# Part 2: Code Reference

- `src/volcast/models/base.py`
  Defines the `ForecastModel` abstract interface.

- `src/volcast/models/har.py`
  HAR-RV benchmark using `rv_cc_d`, `rv_cc_w`, and `rv_cc_m`.

- `src/volcast/models/garch.py`
  Horizon-aware GARCH(1,1) wrapper built on `arch`, fitted to daily returns and
  returning the mean expected variance over the target horizon.

- `src/volcast/models/linear.py`
  Ridge and Lasso with feature scaling, time-series CV, and log-target fitting.

- `src/volcast/models/xgboost_model.py`
  Nonlinear tree benchmark using `xgboost.XGBRegressor`.

Where to start in code:

1. `src/volcast/models/base.py`
2. `src/volcast/models/har.py`
3. `src/volcast/models/linear.py`
4. `src/volcast/models/garch.py`
5. `src/volcast/models/xgboost_model.py`

# Part 3: Short Journal

- 2026-04-19: Updated imports to absolute `volcast.models.*` paths so model modules
  stay stable after the broader `src/` reorganization.
- 2026-07-13: GARCH forecasts now carry the fitted variance state out of sample
  and average horizon-specific expected variances instead of reusing one-day
  forecasts for the five-day target.
