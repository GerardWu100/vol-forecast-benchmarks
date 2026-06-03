"""Tests for realised variance estimator utilities.

These tests validate formula correctness on known inputs and basic DataFrame
behavior for daily aggregation and HAR lag construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volcast.features.rv_estimators import (
    add_har_lags,
    close_to_close_rv,
    compute_daily_rv,
    garman_klass_rv,
    parkinson_rv,
)


class TestCloseToCloseRV:
    """Unit tests for close-to-close realised variance."""

    def test_known_values(self) -> None:
        closes = np.array([100.0, 101.0, 100.5, 102.0, 101.5])
        log_returns = np.diff(np.log(closes))
        expected_rv = np.sum(log_returns**2)

        result_rv = close_to_close_rv(closes)

        assert result_rv == pytest.approx(expected_rv, rel=1e-10)

    def test_constant_price_is_zero(self) -> None:
        closes = np.array([100.0, 100.0, 100.0, 100.0])
        assert close_to_close_rv(closes) == 0.0

    def test_two_prices_single_return(self) -> None:
        closes = np.array([100.0, 105.0])
        expected_rv = np.log(105.0 / 100.0) ** 2
        assert close_to_close_rv(closes) == pytest.approx(expected_rv, rel=1e-10)


class TestParkinsonRV:
    """Unit tests for Parkinson realised variance."""

    def test_known_values(self) -> None:
        high_price = 105.0
        low_price = 95.0
        expected_rv = (1.0 / (4.0 * np.log(2.0))) * (np.log(high_price / low_price) ** 2)

        result_rv = parkinson_rv(high_price, low_price)
        assert result_rv == pytest.approx(expected_rv, rel=1e-10)

    def test_no_range_is_zero(self) -> None:
        assert parkinson_rv(100.0, 100.0) == 0.0


class TestGarmanKlassRV:
    """Unit tests for Garman-Klass realised variance."""

    def test_known_values(self) -> None:
        open_price = 100.0
        high_price = 105.0
        low_price = 95.0
        close_price = 102.0
        expected_rv = 0.5 * (np.log(high_price / low_price) ** 2)
        expected_rv -= (2.0 * np.log(2.0) - 1.0) * (np.log(close_price / open_price) ** 2)

        result_rv = garman_klass_rv(open_price, high_price, low_price, close_price)
        assert result_rv == pytest.approx(expected_rv, rel=1e-10)

    def test_flat_day_is_zero(self) -> None:
        assert garman_klass_rv(100.0, 100.0, 100.0, 100.0) == 0.0


class TestComputeDailyRV:
    """Integration-style tests for daily aggregation."""

    def _make_minutes(self, n_days: int = 5, bars_per_day: int = 390) -> pd.DataFrame:
        """Create synthetic minute bars for deterministic shape and sign tests."""
        dates = pd.bdate_range("2023-01-02", periods=n_days, freq="B")
        rng = np.random.default_rng(42)

        rows: list[dict[str, object]] = []
        for day in dates:
            market_open = pd.Timestamp(day).replace(hour=9, minute=30)
            for minute_idx in range(bars_per_day):
                timestamp = market_open + pd.Timedelta(minutes=minute_idx)
                base_price = 100.0 + 0.02 * minute_idx
                noise = rng.normal(0.0, 0.05)
                close_price = base_price + noise
                rows.append(
                    {
                        "ts": timestamp,
                        "open": close_price - 0.01,
                        "high": close_price + 0.05,
                        "low": close_price - 0.05,
                        "close": close_price,
                        "volume": 1000.0,
                    }
                )

        return pd.DataFrame(rows)

    def test_output_shape(self) -> None:
        minute_df = self._make_minutes(n_days=5, bars_per_day=390)
        daily_rv_df = compute_daily_rv(minute_df, min_minutes=300)

        assert len(daily_rv_df) == 5
        expected_columns = {"date", "rv_cc", "rv_parkinson", "rv_gk", "n_minutes"}
        assert expected_columns.issubset(set(daily_rv_df.columns))

    def test_filters_short_days(self) -> None:
        minute_df = self._make_minutes(n_days=3, bars_per_day=390)
        first_date = minute_df["ts"].dt.date.iloc[0]
        first_day_mask = minute_df["ts"].dt.date == first_date
        rows_to_drop = minute_df[first_day_mask].index[100:]
        truncated_df = minute_df.drop(rows_to_drop)

        daily_rv_df = compute_daily_rv(truncated_df, min_minutes=300)
        assert len(daily_rv_df) == 2

    def test_rv_values_are_positive(self) -> None:
        minute_df = self._make_minutes(n_days=5, bars_per_day=390)
        daily_rv_df = compute_daily_rv(minute_df, min_minutes=300)

        assert (daily_rv_df["rv_cc"] >= 0).all()
        assert (daily_rv_df["rv_parkinson"] >= 0).all()
        assert (daily_rv_df["rv_gk"] >= 0).all()


class TestAddHarLags:
    """Unit tests for HAR lag feature builder."""

    def test_har_lag_values(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=30, freq="B")
        rv_df = pd.DataFrame({"date": dates, "rv_cc": np.arange(1.0, 31.0)})

        lagged_df = add_har_lags(rv_df, column="rv_cc", lags=[1, 5, 22])
        target_row = lagged_df[lagged_df["date"] == dates[22]].iloc[0]

        assert target_row["rv_cc_d"] == pytest.approx(22.0)
        assert target_row["rv_cc_w"] == pytest.approx(np.mean([18, 19, 20, 21, 22]))
        assert target_row["rv_cc_m"] == pytest.approx(np.mean(np.arange(1, 23)))

    def test_early_rows_are_nan(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=10, freq="B")
        rv_df = pd.DataFrame({"date": dates, "rv_cc": np.arange(1.0, 11.0)})

        lagged_df = add_har_lags(rv_df, column="rv_cc", lags=[1, 5, 22])

        assert lagged_df["rv_cc_m"].isna().all()
        assert pd.isna(lagged_df["rv_cc_d"].iloc[0])
        assert not pd.isna(lagged_df["rv_cc_d"].iloc[1])
