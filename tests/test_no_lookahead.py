"""Lookahead-bias checks for the stage-3 feature matrix.

This test module validates temporal alignment assumptions:
- HAR lag features use past realised variance only.
- 1-day-ahead targets align with next-day realised variance.
- Final feature columns contain no missing values after preprocessing.

If required parquet artifacts are unavailable, tests are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from volcast.features.build_features import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


@pytest.fixture
def spy_features() -> pd.DataFrame:
    """Load SPY feature matrix from stage-3 output."""
    path = PROCESSED_DATA_DIR / "SPY_features.parquet"
    if not path.exists():
        pytest.skip("SPY_features.parquet missing; run pipeline stages 1-3 first")

    dataframe = pd.read_parquet(path)
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    return dataframe


@pytest.fixture
def spy_rv() -> pd.DataFrame:
    """Load SPY realised variance matrix from stage-2 output."""
    path = PROCESSED_DATA_DIR / "SPY_rv.parquet"
    if not path.exists():
        pytest.skip("SPY_rv.parquet missing; run pipeline stages 1-3 first")

    dataframe = pd.read_parquet(path)
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    return dataframe


class TestNoLookahead:
    """Temporal consistency tests preventing future-data leakage."""

    def test_har_daily_lag_matches_previous_day_rv(
        self, spy_features: pd.DataFrame, spy_rv: pd.DataFrame
    ) -> None:
        """Assert rv_cc_d(t) equals rv_cc(t-1)."""
        rv_sorted = spy_rv.sort_values("date").reset_index(drop=True)
        rv_sorted["rv_cc_shifted"] = rv_sorted["rv_cc"].shift(1)

        merged = spy_features.merge(rv_sorted[["date", "rv_cc_shifted"]], on="date", how="left")
        diff = (merged["rv_cc_d"] - merged["rv_cc_shifted"]).abs()

        assert float(diff.max()) < 1e-10

    def test_one_day_target_matches_next_day_rv(
        self, spy_features: pd.DataFrame, spy_rv: pd.DataFrame
    ) -> None:
        """Assert rv_1d_ahead(t) equals rv_cc(t+1)."""
        rv_sorted = spy_rv.sort_values("date").reset_index(drop=True)
        rv_sorted["rv_cc_next"] = rv_sorted["rv_cc"].shift(-1)

        merged = spy_features.merge(rv_sorted[["date", "rv_cc_next"]], on="date", how="left")
        diff = (merged["rv_1d_ahead"] - merged["rv_cc_next"]).abs()

        assert float(diff.max()) < 1e-10

    def test_feature_columns_have_no_missing_values(self, spy_features: pd.DataFrame) -> None:
        """Assert stage-3 output has no NaN in core model feature columns."""
        for feature_name in FEATURE_COLS:
            missing_count = int(spy_features[feature_name].isna().sum())
            assert missing_count == 0
