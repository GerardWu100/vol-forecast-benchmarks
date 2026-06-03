"""Abstract model interface for volatility forecast models.

All forecasting models in this benchmark expose the same interface so the
walk-forward evaluation engine can train and score them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    """Abstract base class for benchmark forecast models."""

    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit model parameters on training features and target."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate strictly positive variance forecasts for input features."""
