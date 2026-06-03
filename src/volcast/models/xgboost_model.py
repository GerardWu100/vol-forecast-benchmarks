"""XGBoost model wrapper for nonlinear variance forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from volcast.models.base import ForecastModel

MIN_POSITIVE_VARIANCE = 1e-10


class XGBoostModel(ForecastModel):
    """Gradient-boosted tree regressor with conservative defaults."""

    name = "XGBoost"

    def __init__(self) -> None:
        self._feature_names: list[str] = []
        self._model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit XGBoost regressor on all provided features."""
        self._feature_names = list(X.columns)
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict and floor outputs to a small positive variance."""
        raw_predictions = self._model.predict(X)
        return np.maximum(raw_predictions, MIN_POSITIVE_VARIANCE)

    def get_feature_importance(self) -> dict[str, float]:
        """Return gain-like feature importance indexed by feature name."""
        importance_values = self._model.feature_importances_
        return dict(zip(self._feature_names, importance_values))
