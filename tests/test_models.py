"""Tests for model interface and concrete model implementations."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from volcast.models.garch import GARCHModel
from volcast.models.har import HARModel
from volcast.models.linear import LassoModel, RidgeModel
from volcast.models.xgboost_model import XGBoostModel


class TestHARModel:
    """HAR model behavior tests."""

    def _make_data(self, n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(42)
        rv_d = rng.uniform(0.0001, 0.001, n)
        rv_w = rng.uniform(0.0001, 0.001, n)
        rv_m = rng.uniform(0.0001, 0.001, n)

        X = pd.DataFrame({"rv_cc_d": rv_d, "rv_cc_w": rv_w, "rv_cc_m": rv_m})
        y = pd.Series(0.3 * rv_d + 0.4 * rv_w + 0.2 * rv_m + rng.normal(0, 0.00001, n))
        return X, y

    def test_fit_predict_interface(self) -> None:
        X, y = self._make_data()
        model = HARModel()

        model.fit(X, y)
        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_predictions_are_positive(self) -> None:
        X, y = self._make_data()
        model = HARModel()
        model.fit(X, y)
        predictions = model.predict(X)

        assert (predictions > 0).all()

    def test_name_attribute(self) -> None:
        model = HARModel()
        assert model.name == "HAR-RV"

    def test_reasonable_fit(self) -> None:
        X, y = self._make_data(n=5000)
        model = HARModel()
        model.fit(X, y)
        predictions = model.predict(X)

        residual_sum_squares = np.sum((y.to_numpy() - predictions) ** 2)
        total_sum_squares = np.sum((y.to_numpy() - y.mean()) ** 2)
        r_squared = 1.0 - residual_sum_squares / total_sum_squares

        assert r_squared > 0.9


class TestGARCHModel:
    """GARCH model behavior tests."""

    def _make_data(self, n: int = 500) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, n)
        returns[200:300] = returns[200:300] * 3.0

        X = pd.DataFrame(
            {
                "rv_cc_d": np.abs(returns),
                "rv_cc_w": np.abs(returns),
                "rv_cc_m": np.abs(returns),
                "daily_returns": returns,
            }
        )
        y = pd.Series(returns**2)
        return X, y

    def test_fit_predict_interface(self) -> None:
        X, y = self._make_data()
        model = GARCHModel()

        model.fit(X, y)
        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_predictions_are_positive(self) -> None:
        X, y = self._make_data()
        model = GARCHModel()
        model.fit(X, y)
        predictions = model.predict(X)

        assert (predictions > 0).all()

    def test_name_attribute(self) -> None:
        model = GARCHModel()
        assert model.name == "GARCH(1,1)"

    def test_captures_vol_clustering(self) -> None:
        X, y = self._make_data(n=500)
        model = GARCHModel()

        model.fit(X, y)
        predictions = model.predict(X)

        high_vol_mean = float(np.mean(predictions[200:300]))
        calm_mean = float(np.mean(predictions[0:100]))
        assert high_vol_mean > calm_mean


class TestRidgeModel:
    """Ridge model behavior tests."""

    def _make_data(self, n: int = 300) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            {
                "rv_cc_d": rng.uniform(0.0001, 0.001, n),
                "rv_cc_w": rng.uniform(0.0001, 0.001, n),
                "rv_cc_m": rng.uniform(0.0001, 0.001, n),
                "atm_iv": rng.uniform(0.10, 0.30, n),
                "iv_skew": rng.uniform(-0.05, 0.05, n),
            }
        )
        y = pd.Series(0.5 * X["rv_cc_d"] + 0.001 * X["atm_iv"] + rng.normal(0, 0.00001, n))
        return X, y

    def test_fit_predict_interface(self) -> None:
        X, y = self._make_data()
        model = RidgeModel()
        model.fit(X, y)
        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_predictions_are_positive(self) -> None:
        X, y = self._make_data()
        model = RidgeModel()
        model.fit(X, y)
        predictions = model.predict(X)

        assert (predictions > 0).all()

    def test_handles_multiplicative_positive_target_without_floor_collapse(self) -> None:
        """Ridge should keep positive-target forecasts away from the hard floor."""
        rng = np.random.default_rng(7)

        feature = np.linspace(-2.0, 2.0, 240)
        X = pd.DataFrame(
            {
                "rv_cc_d": feature,
                "rv_cc_w": feature**2,
                "rv_cc_m": np.sin(feature),
                "atm_iv": np.cos(feature),
                "iv_skew": feature * 0.1,
            }
        )
        log_target = -8.0 + 0.8 * feature - 0.3 * feature**2
        y = pd.Series(np.exp(log_target + rng.normal(0.0, 0.03, len(feature))))

        model = RidgeModel()
        model.fit(X.iloc[:180], y.iloc[:180])
        predictions = model.predict(X.iloc[180:])

        assert float(np.min(predictions)) > 1e-8


class TestLassoModel:
    """Lasso model behavior tests."""

    def _make_data(self, n: int = 300) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            {
                "rv_cc_d": rng.uniform(0.0001, 0.001, n),
                "rv_cc_w": rng.uniform(0.0001, 0.001, n),
                "rv_cc_m": rng.uniform(0.0001, 0.001, n),
                "atm_iv": rng.uniform(0.10, 0.30, n),
                "noise": rng.normal(0, 1, n),
            }
        )
        y = pd.Series(0.5 * X["rv_cc_d"] + 0.3 * X["rv_cc_w"] + rng.normal(0, 0.00001, n))
        return X, y

    def test_fit_predict_interface(self) -> None:
        X, y = self._make_data()
        model = LassoModel()
        model.fit(X, y)
        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_feature_selection(self) -> None:
        X, y = self._make_data(n=1000)
        model = LassoModel()
        model.fit(X, y)

        coefficients = model.get_coefficients()
        assert abs(coefficients.get("noise", 0.0)) < 0.001

    def test_fit_avoids_convergence_warning_on_positive_target(self) -> None:
        """Lasso should fit the positive-target benchmark data without warnings."""
        rng = np.random.default_rng(11)
        feature = np.linspace(-3.0, 3.0, 300)

        X = pd.DataFrame(
            {
                "rv_cc_d": feature,
                "rv_cc_w": feature**2,
                "rv_cc_m": np.sin(feature),
                "atm_iv": np.cos(feature),
                "iv_skew": feature * 0.2,
            }
        )
        y = pd.Series(
            np.exp(-10.0 + 1.5 * feature + 0.9 * feature**2 + rng.normal(0.0, 0.05, len(feature)))
        )

        model = LassoModel()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            model.fit(X, y)

        convergence_warnings = [
            warning
            for warning in caught_warnings
            if issubclass(warning.category, ConvergenceWarning)
        ]
        assert not convergence_warnings


class TestXGBoostModel:
    """XGBoost model behavior tests."""

    def _make_data(self, n: int = 500) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(42)
        rv_d = rng.uniform(0.0001, 0.001, n)
        vix = rng.uniform(10.0, 40.0, n)

        X = pd.DataFrame(
            {
                "rv_cc_d": rv_d,
                "rv_cc_w": rng.uniform(0.0001, 0.001, n),
                "rv_cc_m": rng.uniform(0.0001, 0.001, n),
                "atm_iv": rng.uniform(0.10, 0.30, n),
                "vix": vix,
            }
        )

        high_vix_regime = (vix > 25).astype(float)
        y = pd.Series(rv_d * (1.0 + 0.1 * high_vix_regime) + rng.normal(0, 0.00001, n))
        return X, y

    def test_fit_predict_interface(self) -> None:
        X, y = self._make_data()
        model = XGBoostModel()
        model.fit(X, y)
        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_predictions_are_positive(self) -> None:
        X, y = self._make_data()
        model = XGBoostModel()
        model.fit(X, y)
        predictions = model.predict(X)

        assert (predictions > 0).all()

    def test_name_attribute(self) -> None:
        model = XGBoostModel()
        assert model.name == "XGBoost"

    def test_feature_importance_available(self) -> None:
        X, y = self._make_data()
        model = XGBoostModel()
        model.fit(X, y)

        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == len(X.columns)
