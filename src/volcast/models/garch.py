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

    def __init__(self) -> None:
        self._omega = 0.0
        self._alpha = 0.0
        self._beta = 0.0
        self._in_sample_variance: np.ndarray | None = None

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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict conditional variances for rows in X."""
        row_count = len(X)

        if self._in_sample_variance is not None and row_count == len(self._in_sample_variance):
            return np.maximum(self._in_sample_variance.copy(), MIN_POSITIVE_VARIANCE)

        returns = X["daily_returns"].to_numpy(dtype=float)
        variance_path = np.zeros(row_count, dtype=float)

        omega_decimal = self._omega / PERCENT_SQUARED_TO_DECIMAL
        persistence = self._alpha + self._beta

        if row_count == 0:
            return variance_path

        # Seed the variance path from the unconditional variance when stationary.
        if persistence < 1.0:
            variance_path[0] = omega_decimal / (1.0 - persistence)
        else:
            variance_path[0] = omega_decimal + self._alpha * (returns[0] ** 2)

        # Roll the GARCH(1,1) recursion forward one day at a time.
        for idx in range(1, row_count):
            prev_return_sq = returns[idx - 1] ** 2
            prev_variance = variance_path[idx - 1]
            variance_path[idx] = (
                omega_decimal + self._alpha * prev_return_sq + self._beta * prev_variance
            )

        return np.maximum(variance_path, MIN_POSITIVE_VARIANCE)
