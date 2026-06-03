"""Tests for walk-forward evaluation diagnostics and lightweight pipeline flow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import volcast.evaluation.train_evaluate as train_evaluate
import volcast.features.build_features as build_features
import volcast.features.compute_rv as compute_rv


def test_compute_model_diagnostics_counts_floor_hits() -> None:
    """Diagnostics should expose floor-hit rates and forecast ranges by model."""
    forecasts_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]),
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "horizon": [1, 1, 1, 1],
            "model": ["Ridge", "Ridge", "HAR-RV", "HAR-RV"],
            "y_true": [0.0004, 0.0005, 0.0004, 0.0005],
            "y_pred": [1e-10, 0.0003, 0.0004, 0.0006],
        }
    )

    diagnostics_df = train_evaluate.compute_model_diagnostics(forecasts_df)

    ridge_row = diagnostics_df.loc[diagnostics_df["model"] == "Ridge"].iloc[0]
    har_row = diagnostics_df.loc[diagnostics_df["model"] == "HAR-RV"].iloc[0]

    assert ridge_row["floor_hit_count"] == 1
    assert ridge_row["floor_hit_rate"] == 0.5
    assert ridge_row["min_prediction"] == 1e-10
    assert har_row["floor_hit_count"] == 0
    assert har_row["median_prediction"] == 0.0005


def test_stage_2_to_4_pipeline_smoke_writes_outputs(tmp_path: Path) -> None:
    """A tiny local dataset should run through stages 2-4 and emit outputs."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "outputs"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    business_days = pd.bdate_range("2020-01-02", periods=55)

    minute_rows: list[dict[str, object]] = []
    for day_index, day in enumerate(business_days):
        market_open = pd.Timestamp(day).replace(hour=9, minute=30)
        base_price = 100.0 + day_index * 0.2

        for minute_index in range(300):
            timestamp = market_open + pd.Timedelta(minutes=minute_index)
            close_price = base_price + minute_index * 0.01
            minute_rows.append(
                {
                    "symbol": "SPY",
                    "ts": timestamp,
                    "open": close_price - 0.01,
                    "high": close_price + 0.02,
                    "low": close_price - 0.02,
                    "close": close_price,
                    "volume": 1000 + minute_index,
                }
            )

    minutes_df = pd.DataFrame(minute_rows)
    minutes_df.to_parquet(raw_dir / "SPY_minutes.parquet", index=False)

    option_rows: list[dict[str, object]] = []
    for day_index, day in enumerate(business_days):
        front_expiry = pd.Timestamp(day) + pd.Timedelta(days=20)
        second_expiry = pd.Timestamp(day) + pd.Timedelta(days=45)
        atm_iv_level = 0.18 + day_index * 0.0004

        for expiry_date, expiry_bump in [(front_expiry, 0.0), (second_expiry, 0.015)]:
            for option_type, delta, iv_bump in [
                ("c", 0.50, 0.0),
                ("c", 0.25, 0.01),
                ("p", -0.25, 0.03),
            ]:
                midpoint_iv = atm_iv_level + expiry_bump + iv_bump
                option_rows.append(
                    {
                        "symbol": "SPY",
                        "trade_date": pd.Timestamp(day),
                        "strike_price": 100.0,
                        "expiry_date": expiry_date,
                        "option_type": option_type,
                        "bid": 1.0,
                        "ask": 1.1,
                        "bid_iv": midpoint_iv - 0.005,
                        "ask_iv": midpoint_iv + 0.005,
                        "delta": delta,
                        "open_interest": 1000,
                        "volume": 100,
                    }
                )

    options_df = pd.DataFrame(option_rows)
    options_dir = raw_dir / "SPY_options.parquet"
    options_dir.mkdir(parents=True, exist_ok=True)
    options_df.to_parquet(options_dir / "part-00000.parquet", index=False)

    vix_df = pd.DataFrame(
        {
            "date": business_days,
            "vix": np.linspace(15.0, 22.0, len(business_days)),
        }
    )
    vix_df.to_parquet(raw_dir / "VIX_daily.parquet", index=False)

    config = {
        "data": {
            "symbols": ["SPY"],
            "robustness_symbols": [],
            "start_date": "2020-01-01",
            "end_date": "2020-04-30",
        },
        "cache": {
            "raw_cache_dir": str(raw_dir),
            "metadata_suffix": ".metadata.json",
            "version": 1,
            "options_shard_rows": 1000,
        },
        "rv": {
            "min_minutes_per_day": 300,
            "har_lags": [1, 5, 22],
        },
        "options": {
            "atm_delta": 0.50,
            "skew_delta": 0.25,
            "min_dte": 7,
            "max_dte": 45,
            "max_ffill_days": 3,
        },
        "forecast": {
            "horizons": [1],
            "initial_train_years": 0,
            "retrain_every_days": 5,
        },
        "models": {
            "enabled": ["har", "ridge", "lasso"],
        },
    }

    original_compute_processed = compute_rv.PROCESSED_DATA_DIR
    original_build_processed = build_features.PROCESSED_DATA_DIR
    original_train_processed = train_evaluate.PROCESSED_DATA_DIR
    original_train_output = train_evaluate.OUTPUT_DIR
    original_compute_load_config = compute_rv.load_config
    original_build_load_config = build_features.load_config
    original_train_load_config = train_evaluate.load_config

    try:
        compute_rv.PROCESSED_DATA_DIR = processed_dir
        build_features.PROCESSED_DATA_DIR = processed_dir
        train_evaluate.PROCESSED_DATA_DIR = processed_dir
        train_evaluate.OUTPUT_DIR = output_dir

        compute_rv.load_config = lambda: config
        build_features.load_config = lambda: config
        train_evaluate.load_config = lambda: config

        compute_rv.main()
        build_features.main()
        train_evaluate.main()
    finally:
        compute_rv.PROCESSED_DATA_DIR = original_compute_processed
        build_features.PROCESSED_DATA_DIR = original_build_processed
        train_evaluate.PROCESSED_DATA_DIR = original_train_processed
        train_evaluate.OUTPUT_DIR = original_train_output
        compute_rv.load_config = original_compute_load_config
        build_features.load_config = original_build_load_config
        train_evaluate.load_config = original_train_load_config

    assert (processed_dir / "SPY_rv.parquet").exists()
    assert (processed_dir / "SPY_features.parquet").exists()
    assert (output_dir / "forecasts.parquet").exists()
    assert (output_dir / "scores.parquet").exists()
    assert (output_dir / "dm_tests.parquet").exists()
    assert (output_dir / "model_diagnostics.parquet").exists()
