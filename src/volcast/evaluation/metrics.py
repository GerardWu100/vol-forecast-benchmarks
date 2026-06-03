"""Forecast evaluation metrics and statistical comparison tests.

This module provides:
- QLIKE (Quasi-Likelihood) loss
- MSE (Mean Squared Error) loss
- Diebold-Mariano (DM) test with a Newey-West style HAC variance estimate

All functions operate on realised variance forecasts (not volatility levels).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean QLIKE loss.

    Parameters
    ----------
    y_true : np.ndarray
        Realised variance values.
    y_pred : np.ndarray
        Forecast variance values. Must be strictly positive.

    Returns
    -------
    float
        Average QLIKE loss.
    """
    loss_values = np.log(y_pred) + y_true / y_pred
    mean_loss = np.mean(loss_values)
    return float(mean_loss)


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error loss.

    Parameters
    ----------
    y_true : np.ndarray
        Realised variance values.
    y_pred : np.ndarray
        Forecast variance values.

    Returns
    -------
    float
        Average squared forecast error.
    """
    squared_errors = (y_true - y_pred) ** 2
    mean_squared_error = np.mean(squared_errors)
    return float(mean_squared_error)


def diebold_mariano_test(losses_a: np.ndarray, losses_b: np.ndarray) -> tuple[float, float]:
    """Run a two-sided Diebold-Mariano test for equal predictive accuracy.

    The null hypothesis is E[d_t] = 0, where d_t = losses_a[t] - losses_b[t].
    A positive t-statistic means model A has higher average loss than model B.

    Parameters
    ----------
    losses_a : np.ndarray
        Per-observation loss sequence for model A.
    losses_b : np.ndarray
        Per-observation loss sequence for model B.

    Returns
    -------
    tuple[float, float]
        (t_statistic, p_value) under a large-sample normal approximation.
    """
    loss_diff = losses_a - losses_b
    sample_size = len(loss_diff)
    diff_mean = np.mean(loss_diff)

    # Standard data-driven lag choice for HAC bandwidth.
    max_lag = max(1, int(sample_size ** (1.0 / 3.0)))

    centered_diff = loss_diff - diff_mean
    gamma_0 = np.mean(centered_diff * centered_diff)
    hac_variance = gamma_0

    # Newey-West style Bartlett kernel weighted autocovariances.
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cov_lag = np.mean(centered_diff[lag:] * centered_diff[:-lag])
        hac_variance = hac_variance + 2.0 * weight * cov_lag

    standard_error = np.sqrt(hac_variance / sample_size)
    if standard_error < 1e-15:
        return (0.0, 1.0)

    t_statistic = diff_mean / standard_error
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(t_statistic)))
    return (float(t_statistic), float(p_value))
