"""Tests for forecast evaluation metrics and DM significance test."""

from __future__ import annotations

import numpy as np
import pytest

from volcast.evaluation.metrics import diebold_mariano_test, mse_loss, qlike_loss


class TestQLIKE:
    """QLIKE metric unit tests."""

    def test_known_values(self) -> None:
        y_true = np.array([2.0])
        y_pred = np.array([1.0])
        assert qlike_loss(y_true, y_pred) == pytest.approx(2.0)

    def test_perfect_forecast(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        expected = np.mean(np.log(y_pred) + y_true / y_pred)
        assert qlike_loss(y_true, y_pred) == pytest.approx(expected)

    def test_multiple_observations(self) -> None:
        y_true = np.array([1.0, 4.0])
        y_pred = np.array([2.0, 2.0])
        expected = np.mean(np.log(y_pred) + y_true / y_pred)
        assert qlike_loss(y_true, y_pred) == pytest.approx(expected)


class TestMSE:
    """MSE metric unit tests."""

    def test_known_values(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        expected = np.mean((y_true - y_pred) ** 2)
        assert mse_loss(y_true, y_pred) == pytest.approx(expected)

    def test_perfect_forecast_is_zero(self) -> None:
        y_values = np.array([1.0, 2.0, 3.0])
        assert mse_loss(y_values, y_values) == pytest.approx(0.0)


class TestDieboldMariano:
    """Diebold-Mariano test behavior checks."""

    def test_identical_losses_not_significant(self) -> None:
        losses_a = np.random.default_rng(42).normal(0, 1, size=500)
        losses_b = losses_a.copy()

        t_stat, p_value = diebold_mariano_test(losses_a, losses_b)

        assert abs(t_stat) < 0.01
        assert p_value > 0.9

    def test_clearly_different_losses(self) -> None:
        rng = np.random.default_rng(42)
        losses_a = rng.normal(5, 0.5, size=500)
        losses_b = rng.normal(0, 0.5, size=500)

        t_stat, p_value = diebold_mariano_test(losses_a, losses_b)

        assert t_stat > 2.0
        assert p_value < 0.05

    def test_returns_two_floats(self) -> None:
        losses_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        losses_b = np.array([1.1, 2.1, 3.1, 4.1, 5.1] * 20)

        t_stat, p_value = diebold_mariano_test(losses_a, losses_b)

        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0
