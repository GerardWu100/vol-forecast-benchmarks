"""HAR-RV (Heterogeneous Autoregressive Realised Variance) model.

HAR-RV is a standard volatility benchmark that regresses next-period realised
variance on lagged daily, weekly, and monthly realised variance summaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from volcast.models.base import ForecastModel

# HAR core explanatory variables (daily, weekly, monthly variance components).
HAR_FEATURES = ["rv_cc_d", "rv_cc_w", "rv_cc_m"]

# Variance forecasts should not be negative.
MIN_POSITIVE_VARIANCE = 1e-10


class HARModel(ForecastModel):
    """OLS-based HAR-RV model."""

    name = "HAR-RV"

    def __init__(self) -> None:
        self._linear_regression = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit OLS regression on HAR feature subset."""
        har_matrix = X[HAR_FEATURES]
        self._linear_regression.fit(har_matrix, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict variance and clip to a small positive floor."""
        har_matrix = X[HAR_FEATURES]
        raw_predictions = self._linear_regression.predict(har_matrix)
        safe_predictions = np.maximum(raw_predictions, MIN_POSITIVE_VARIANCE)
        return safe_predictions
