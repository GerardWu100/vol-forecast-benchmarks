"""Tests for option-implied volatility feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volcast.features.option_features import (
    build_option_features,
    compute_atm_iv,
    compute_iv_skew,
    compute_iv_term_slope,
)


def _make_chain(trade_date: str, expiry_date: str) -> pd.DataFrame:
    """Create a synthetic option chain for one trade date and one expiry."""
    rows: list[dict[str, object]] = []

    for delta in np.arange(0.1, 1.0, 0.1):
        call_iv = 0.20 + 0.02 * abs(delta - 0.5)
        rows.append(
            {
                "trade_date": pd.Timestamp(trade_date),
                "expiry_date": pd.Timestamp(expiry_date),
                "option_type": "c",
                "delta": round(float(delta), 2),
                "bid_iv": call_iv - 0.005,
                "ask_iv": call_iv + 0.005,
                "strike_price": 400 + (0.5 - delta) * 50,
                "bid": 5.0,
                "ask": 5.5,
                "open_interest": 1000,
                "volume": 500,
            }
        )

    for delta in np.arange(-0.1, -1.0, -0.1):
        put_iv = 0.20 + 0.05 * abs(abs(delta) - 0.5) + 0.03
        rows.append(
            {
                "trade_date": pd.Timestamp(trade_date),
                "expiry_date": pd.Timestamp(expiry_date),
                "option_type": "p",
                "delta": round(float(delta), 2),
                "bid_iv": put_iv - 0.005,
                "ask_iv": put_iv + 0.005,
                "strike_price": 400 + (0.5 - abs(delta)) * 50,
                "bid": 5.0,
                "ask": 5.5,
                "open_interest": 1000,
                "volume": 500,
            }
        )

    return pd.DataFrame(rows)


class TestATMIV:
    """ATM implied volatility selection checks."""

    def test_selects_atm_call(self) -> None:
        chain = _make_chain("2023-06-01", "2023-06-20")

        atm_iv = compute_atm_iv(chain, target_delta=0.50, min_dte=7, max_dte=45)

        assert atm_iv == pytest.approx(0.20, abs=0.01)

    def test_returns_nan_if_no_valid_expiry(self) -> None:
        chain = _make_chain("2023-06-01", "2023-06-05")
        atm_iv = compute_atm_iv(chain, target_delta=0.50, min_dte=7, max_dte=45)

        assert np.isnan(atm_iv)


class TestIVSkew:
    """Skew feature behavior checks."""

    def test_skew_is_positive(self) -> None:
        chain = _make_chain("2023-06-01", "2023-06-20")

        skew = compute_iv_skew(chain, skew_delta=0.25, min_dte=7, max_dte=45)

        assert skew > 0


class TestIVTermSlope:
    """Term structure slope checks."""

    def test_term_slope_with_two_expiries(self) -> None:
        chain_front = _make_chain("2023-06-01", "2023-06-20")
        chain_second = _make_chain("2023-06-01", "2023-07-20")
        chain_second["bid_iv"] = chain_second["bid_iv"] + 0.02
        chain_second["ask_iv"] = chain_second["ask_iv"] + 0.02

        combined_chain = pd.concat([chain_front, chain_second], ignore_index=True)

        slope = compute_iv_term_slope(
            combined_chain,
            target_delta=0.50,
            min_dte=7,
            max_dte=60,
        )

        assert slope == pytest.approx(0.02, abs=0.01)


class TestBuildOptionFeatures:
    """Daily feature DataFrame construction checks."""

    def test_output_columns(self) -> None:
        chain_front = _make_chain("2023-06-01", "2023-06-20")
        chain_second = _make_chain("2023-06-01", "2023-07-20")
        combined_chain = pd.concat([chain_front, chain_second], ignore_index=True)

        config = {
            "atm_delta": 0.50,
            "skew_delta": 0.25,
            "min_dte": 7,
            "max_dte": 60,
        }

        feature_df = build_option_features(combined_chain, config)

        assert len(feature_df) == 1
        expected_columns = {"trade_date", "atm_iv", "iv_skew", "iv_term_slope"}
        assert expected_columns.issubset(set(feature_df.columns))
