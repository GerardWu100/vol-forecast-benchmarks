"""Stage 3: Build model-ready feature matrix from RV, options, and VIX.

This stage merges:
- Realised variance lag features from stage 2.
- Option-implied volatility features from option chains.
- Lagged VIX feature.

The resulting matrix is written to ``data/processed/{SYMBOL}_features.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from volcast.features.option_features import build_option_features
from volcast.shared.config import configured_symbols, load_config, resolve_raw_cache_dir
from volcast.shared.logging import configure_logging
from volcast.shared.paths import DEFAULT_PROCESSED_DATA_DIR

PROCESSED_DATA_DIR = DEFAULT_PROCESSED_DATA_DIR

LOGGER = logging.getLogger(__name__)

# Core model inputs (X).
FEATURE_COLS = [
    "rv_cc_d",
    "rv_cc_w",
    "rv_cc_m",
    "rv_pk_d",
    "rv_pk_w",
    "rv_pk_m",
    "rv_gk_d",
    "rv_gk_w",
    "rv_gk_m",
    "atm_iv",
    "iv_skew",
    "iv_term_slope",
    "iv_rv_spread",
    "vix",
]

# Forecast targets (y).
TARGET_COLS = ["rv_1d_ahead", "rv_5d_ahead"]

# Metadata columns retained for downstream processing.
META_COLS = ["date", "daily_returns"]

TRADING_DAYS_PER_YEAR = 252.0


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    """Convert datetime-like series to timezone-naive pandas timestamps."""
    converted = pd.to_datetime(series)
    if getattr(converted.dt, "tz", None) is not None:
        converted = converted.dt.tz_localize(None)
    return converted


def _lag_trade_date_feature_frame(feature_df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Shift feature date one business day forward to enforce t-1 feature usage."""
    lagged_df = feature_df.copy()
    lagged_df["date"] = pd.to_datetime(lagged_df[date_col]) + pd.offsets.BDay(1)
    lagged_df = lagged_df.drop(columns=[date_col])
    lagged_df = lagged_df.sort_values("date").reset_index(drop=True)
    return lagged_df


def _merge_lagged_vix(rv_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Merge lagged VIX feature into RV frame using one-business-day lag."""
    vix_lagged = vix_df.copy()
    vix_lagged["date"] = pd.to_datetime(vix_lagged["date"]) + pd.offsets.BDay(1)
    vix_lagged = vix_lagged.sort_values("date").reset_index(drop=True)

    merged = rv_df.merge(vix_lagged[["date", "vix"]], on="date", how="left")
    return merged


def _load_options_cache(options_path: Path) -> pd.DataFrame:
    """Load options cache from sharded directory or legacy single parquet file."""
    if options_path.is_file():
        return pd.read_parquet(options_path)

    shard_paths = sorted(options_path.glob("*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No parquet shards found in options cache directory: {options_path}"
        )

    shard_frames = [pd.read_parquet(shard_path) for shard_path in shard_paths]
    return pd.concat(shard_frames, ignore_index=True)


def process_symbol(symbol: str, config: dict, raw_cache_dir: Path) -> pd.DataFrame:
    """Build final feature frame for one symbol."""
    rv_path = PROCESSED_DATA_DIR / f"{symbol}_rv.parquet"
    options_path = raw_cache_dir / f"{symbol}_options.parquet"
    vix_path = raw_cache_dir / "VIX_daily.parquet"

    rv_df = pd.read_parquet(rv_path)
    rv_df["date"] = _to_naive_datetime(rv_df["date"])

    options_df = _load_options_cache(options_path)
    options_df["trade_date"] = _to_naive_datetime(options_df["trade_date"])
    options_df["expiry_date"] = _to_naive_datetime(options_df["expiry_date"])

    option_config = config["options"]
    option_features = build_option_features(options_df, option_config)

    if option_features.empty:
        option_lagged = pd.DataFrame(columns=["date", "atm_iv", "iv_skew", "iv_term_slope"])
    else:
        option_lagged = _lag_trade_date_feature_frame(option_features, date_col="trade_date")

    merged_df = rv_df.merge(option_lagged, on="date", how="left")

    vix_df = pd.read_parquet(vix_path)
    vix_df["date"] = _to_naive_datetime(vix_df["date"])
    merged_df = _merge_lagged_vix(merged_df, vix_df)

    # IV-RV spread uses annualised weekly RV converted to volatility units.
    merged_df["iv_rv_spread"] = merged_df["atm_iv"] - np.sqrt(
        TRADING_DAYS_PER_YEAR * merged_df["rv_cc_w"]
    )

    # Forward-fill option-like features for short market closures.
    fill_limit = option_config["max_ffill_days"]
    option_like_cols = ["atm_iv", "iv_skew", "iv_term_slope", "vix", "iv_rv_spread"]
    merged_df[option_like_cols] = merged_df[option_like_cols].ffill(limit=fill_limit)

    selected_columns = META_COLS + FEATURE_COLS + TARGET_COLS
    selected_df = merged_df[selected_columns].copy()

    required_columns = FEATURE_COLS + TARGET_COLS
    row_count_before = len(selected_df)
    selected_df = selected_df.dropna(subset=required_columns)
    row_count_after = len(selected_df)
    dropped_rows = row_count_before - row_count_after
    LOGGER.info("%s: dropped %s rows with NaN core fields", symbol, dropped_rows)

    return selected_df.sort_values("date").reset_index(drop=True)


def main() -> None:
    """Run stage 3 for all configured symbols."""
    configure_logging()
    config = load_config()
    raw_cache_dir = resolve_raw_cache_dir(config)

    symbols = configured_symbols(config)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        feature_df = process_symbol(symbol, config, raw_cache_dir)
        output_path = PROCESSED_DATA_DIR / f"{symbol}_features.parquet"
        feature_df.to_parquet(output_path, index=False)
        LOGGER.info("Saved %s feature frame: %s (%s rows)", symbol, output_path, len(feature_df))

    LOGGER.info("Stage 3 complete")


if __name__ == "__main__":
    main()
