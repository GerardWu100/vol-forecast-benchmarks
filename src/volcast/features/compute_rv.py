"""Stage 2: Compute daily realised variance features from minute bars.

This stage reads minute-level OHLCV parquet files from ``data/raw`` and
computes daily realised variance estimators plus HAR lag features.

Outputs are written to ``data/processed/{SYMBOL}_rv.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from volcast.features.rv_estimators import add_har_lags, compute_daily_rv
from volcast.shared.config import configured_symbols, load_config, resolve_raw_cache_dir
from volcast.shared.logging import configure_logging
from volcast.shared.paths import DEFAULT_PROCESSED_DATA_DIR

PROCESSED_DATA_DIR = DEFAULT_PROCESSED_DATA_DIR

LOGGER = logging.getLogger(__name__)


def _add_forecast_targets(daily_rv_df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-day and 5-day ahead realised variance targets."""
    result_df = daily_rv_df.copy()

    # One-day-ahead target: RV_{t+1}.
    result_df["rv_1d_ahead"] = result_df["rv_cc"].shift(-1)

    # Five-day-ahead target: mean(RV_{t+1}, ..., RV_{t+5}).
    forward_shifted = result_df["rv_cc"].shift(-1)
    forward_five_day_mean = forward_shifted.rolling(window=5, min_periods=5).mean()
    result_df["rv_5d_ahead"] = forward_five_day_mean.shift(-4)

    return result_df


def process_symbol(symbol: str, config: dict, raw_cache_dir: Path) -> pd.DataFrame:
    """Compute daily RV estimators, HAR lags, and targets for one symbol."""
    input_path = raw_cache_dir / f"{symbol}_minutes.parquet"
    minute_df = pd.read_parquet(input_path)
    LOGGER.info("%s minute rows loaded: %s", symbol, f"{len(minute_df):,}")

    min_minutes = config["rv"]["min_minutes_per_day"]
    lag_windows = config["rv"]["har_lags"]

    daily_rv_df = compute_daily_rv(minute_df, min_minutes=min_minutes)
    LOGGER.info("%s daily rows after quality filter: %s", symbol, len(daily_rv_df))

    # Daily log return for GARCH model input.
    daily_rv_df["daily_returns"] = np.log(
        daily_rv_df["daily_close"] / daily_rv_df["daily_close"].shift(1)
    )

    # HAR lags for close-to-close RV.
    daily_rv_df = add_har_lags(daily_rv_df, column="rv_cc", lags=lag_windows)

    # HAR lags for Parkinson RV, then rename to compact prefixes.
    daily_rv_df = add_har_lags(daily_rv_df, column="rv_parkinson", lags=lag_windows)
    daily_rv_df = daily_rv_df.rename(
        columns={
            "rv_parkinson_d": "rv_pk_d",
            "rv_parkinson_w": "rv_pk_w",
            "rv_parkinson_m": "rv_pk_m",
        }
    )

    # HAR lags for Garman-Klass RV.
    daily_rv_df = add_har_lags(daily_rv_df, column="rv_gk", lags=lag_windows)

    return _add_forecast_targets(daily_rv_df)


def main() -> None:
    """Run stage 2 for all configured symbols."""
    configure_logging()
    config = load_config()
    raw_cache_dir = resolve_raw_cache_dir(config)

    symbols = configured_symbols(config)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        symbol_df = process_symbol(symbol, config, raw_cache_dir)
        output_path = PROCESSED_DATA_DIR / f"{symbol}_rv.parquet"
        symbol_df.to_parquet(output_path, index=False)
        LOGGER.info("Saved %s RV data: %s (%s rows)", symbol, output_path, len(symbol_df))

    LOGGER.info("Stage 2 complete")


if __name__ == "__main__":
    main()
