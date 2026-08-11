"""GARCH(1,1) model wrapper for conditional variance forecasting.

This model estimates a GARCH (Generalized Autoregressive Conditional
Heteroskedasticity) process on daily returns and returns conditional variance
estimates as volatility forecasts.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from volcast.models.base import ForecastModel

PERCENT_SCALE = 100.0
PERCENT_SQUARED_TO_DECIMAL = 10000.0
MIN_POSITIVE_VARIANCE = 1e-10


def _as_float_array(values: object) -> np.ndarray:
    """Convert arch model outputs to a plain float numpy array."""
    if hasattr(values, "to_numpy"):
        return values.to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


class GARCHModel(ForecastModel):
    """GARCH(1,1) conditional variance model."""

    name = "GARCH(1,1)"

    def __init__(self, forecast_horizon: int = 1) -> None:
        """Initialize a horizon-aware GARCH(1,1) forecaster.

        Parameters
        ----------
        forecast_horizon : int, default=1
            Number of trading-day conditional variances to average for each
            forecast, matching the project's average-variance target.

        Raises
        ------
        ValueError
            If ``forecast_horizon`` is not positive.
        """
        if forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")

        self._forecast_horizon = forecast_horizon
        self._omega = 0.0
        self._alpha = 0.0
        self._beta = 0.0
        self._in_sample_variance: np.ndarray | None = None
        self._last_conditional_variance: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit GARCH parameters using the ``daily_returns`` series from X."""
        returns = X["daily_returns"].to_numpy(dtype=float)
        scaled_returns = returns * PERCENT_SCALE

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(scaled_returns, vol="Garch", p=1, q=1, mean="Zero")
            result = model.fit(disp="off")

        self._omega = float(result.params.get("omega", 0.0))
        self._alpha = float(result.params.get("alpha[1]", 0.0))
        self._beta = float(result.params.get("beta[1]", 0.0))

        # arch reports percent-scaled volatility; convert back to decimal variance.
        conditional_volatility = _as_float_array(result.conditional_volatility)
        self._in_sample_variance = (conditional_volatility**2) / PERCENT_SQUARED_TO_DECIMAL
        self._last_conditional_variance = float(self._in_sample_variance[-1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict conditional variances for rows in X."""
        row_count = len(X)
        returns = X["daily_returns"].to_numpy(dtype=float)
        average_variance_forecasts = np.zeros(row_count, dtype=float)

        omega_decimal = self._omega / PERCENT_SQUARED_TO_DECIMAL
        persistence = self._alpha + self._beta

        if row_count == 0:
            return average_variance_forecasts

        # The last fitted conditional variance carries the state into the first
        # out-of-sample date. Fall back to the unconditional variance only when
        # predict() is called before a fitted state is available.
        if self._last_conditional_variance is not None:
            previous_variance = self._last_conditional_variance
        elif persistence < 1.0:
            previous_variance = omega_decimal / (1.0 - persistence)
        else:
            previous_variance = omega_decimal

        for row_index, observed_return in enumerate(returns):
            # At feature date t, return r_t is observed. It updates the one-step
            # conditional variance forecast for target day t+1.
            one_step_variance = (
                omega_decimal + self._alpha * observed_return**2 + self._beta * previous_variance
            )

            # For k>1, E_t[r_{t+k-1}^2] equals its conditional variance, so the
            # expected variance recursion uses persistence = alpha + beta.
            horizon_variance = one_step_variance
            variance_sum = one_step_variance
            for _ in range(1, self._forecast_horizon):
                horizon_variance = omega_decimal + persistence * horizon_variance
                variance_sum += horizon_variance

            average_variance_forecasts[row_index] = variance_sum / self._forecast_horizon
            previous_variance = one_step_variance

        return np.maximum(average_variance_forecasts, MIN_POSITIVE_VARIANCE)
