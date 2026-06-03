"""Regularised linear models (Ridge and Lasso) for variance forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from volcast.models.base import ForecastModel

RIDGE_ALPHAS = np.logspace(-3, 3, 40)
LASSO_ALPHAS = np.logspace(-4, 0, 40)
CV_SPLITS = 5
MIN_CV_SPLITS = 2
LASSO_MAX_ITER = 100_000
LASSO_TOLERANCE = 1e-4
MIN_POSITIVE_VARIANCE = 1e-10
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_LASSO_ALPHA = 0.01


def _log_positive_target(y: pd.Series) -> np.ndarray:
    """Map strictly positive variance targets to log space for stable fitting."""
    positive_target = np.maximum(y.to_numpy(dtype=float), MIN_POSITIVE_VARIANCE)
    return np.log(positive_target)


def _inverse_log_target(log_predictions: np.ndarray) -> np.ndarray:
    """Map log-space predictions back to positive variance forecasts."""
    positive_predictions = np.exp(log_predictions)
    return np.maximum(positive_predictions, MIN_POSITIVE_VARIANCE)


def _time_series_split_or_none(sample_count: int) -> TimeSeriesSplit | None:
    """Create the largest valid time-series CV splitter for the sample size."""
    split_count = min(CV_SPLITS, sample_count - 1)
    if split_count < MIN_CV_SPLITS:
        return None
    return TimeSeriesSplit(n_splits=split_count)


class RidgeModel(ForecastModel):
    """Ridge regression with time-series aware cross-validation."""

    name = "Ridge"

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._model: RidgeCV | Ridge = Ridge(alpha=DEFAULT_RIDGE_ALPHA)
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Scale features and fit ridge regression."""
        self._feature_names = list(X.columns)
        scaled_matrix = self._scaler.fit_transform(X)
        log_target = _log_positive_target(y)

        splitter = _time_series_split_or_none(len(X))
        if splitter is None:
            self._model = Ridge(alpha=DEFAULT_RIDGE_ALPHA)
        else:
            self._model = RidgeCV(alphas=RIDGE_ALPHAS, cv=splitter)

        self._model.fit(scaled_matrix, log_target)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict and floor outputs to a small positive variance."""
        scaled_matrix = self._scaler.transform(X)
        log_predictions = self._model.predict(scaled_matrix)
        return _inverse_log_target(log_predictions)

    def get_coefficients(self) -> dict[str, float]:
        """Return coefficient map keyed by feature name."""
        return dict(zip(self._feature_names, self._model.coef_))


class LassoModel(ForecastModel):
    """Lasso regression with time-series aware cross-validation."""

    name = "Lasso"

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._model: LassoCV | Lasso = Lasso(alpha=DEFAULT_LASSO_ALPHA)
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Scale features and fit lasso regression."""
        self._feature_names = list(X.columns)
        scaled_matrix = self._scaler.fit_transform(X)
        log_target = _log_positive_target(y)

        splitter = _time_series_split_or_none(len(X))
        if splitter is None:
            self._model = Lasso(
                alpha=DEFAULT_LASSO_ALPHA, max_iter=LASSO_MAX_ITER, tol=LASSO_TOLERANCE
            )
        else:
            self._model = LassoCV(
                alphas=LASSO_ALPHAS,
                cv=splitter,
                max_iter=LASSO_MAX_ITER,
                tol=LASSO_TOLERANCE,
            )

        self._model.fit(scaled_matrix, log_target)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict and floor outputs to a small positive variance."""
        scaled_matrix = self._scaler.transform(X)
        log_predictions = self._model.predict(scaled_matrix)
        return _inverse_log_target(log_predictions)

    def get_coefficients(self) -> dict[str, float]:
        """Return coefficient map keyed by feature name."""
        return dict(zip(self._feature_names, self._model.coef_))
