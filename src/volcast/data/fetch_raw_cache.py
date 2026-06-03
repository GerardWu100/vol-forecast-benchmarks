"""Stage 1: Populate or validate the raw parquet cache.

This stage enforces an offline-first cache policy:
- If required raw parquet files and metadata sidecars are valid, no database call.
- If cache is missing/invalid and ClickHouse is reachable, refresh only broken datasets.
- If cache is missing/invalid and ClickHouse is unreachable, fail with actionable errors.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import clickhouse_connect
import pandas as pd
from clickhouse_connect.driver import Client
from dotenv import dotenv_values

from volcast.data.cache_validation import (
    CacheSpec,
    build_cache_specs,
    cache_parquet_path,
    cache_sidecar_path,
    cache_spec_for_symbol,
    cache_spec_for_vix,
    normalize_to_date_string,
    validate_required_cache,
    write_sidecar_metadata,
)
from volcast.shared.config import (
    load_config,
    resolve_cache_version,
    resolve_metadata_suffix,
    resolve_options_shard_rows,
    resolve_raw_cache_dir,
)
from volcast.shared.logging import configure_logging
from volcast.shared.paths import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)


def get_clickhouse_client() -> Client:
    """Create ClickHouse client using values from project ``.env`` file."""
    env_path = PROJECT_ROOT / ".env"
    env_values = dotenv_values(env_path)

    host = env_values.get("CLICKHOUSE_HOST", "127.0.0.1")
    port = int(env_values.get("CLICKHOUSE_PORT", "8123"))
    username = env_values.get("CLICKHOUSE_USER", "default")
    password = env_values.get("CLICKHOUSE_PASSWORD", "")
    secure = env_values.get("CLICKHOUSE_SECURE", "false").lower() == "true"
    verify = env_values.get("CLICKHOUSE_VERIFY", "false").lower() == "true"

    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        secure=secure,
        verify=verify,
    )


def _save_parquet(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Persist one DataFrame as parquet, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)


def _remove_existing_path(path: Path) -> None:
    """Delete existing file or directory path before rewriting cache."""
    if not path.exists():
        return

    if path.is_file():
        path.unlink()
        return

    shutil.rmtree(path)


def _save_parquet_shards(dataframe: pd.DataFrame, output_path: Path, rows_per_shard: int) -> None:
    """Persist DataFrame into sharded parquet directory for large options tables."""
    if rows_per_shard <= 0:
        raise ValueError(f"rows_per_shard must be positive, got {rows_per_shard}")

    _remove_existing_path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    for start_row in range(0, len(dataframe), rows_per_shard):
        end_row = start_row + rows_per_shard
        shard_df = dataframe.iloc[start_row:end_row]
        shard_path = output_path / f"part-{shard_index:05d}.parquet"
        shard_df.to_parquet(shard_path, index=False)
        shard_index += 1

    if shard_index == 0:
        shard_path = output_path / "part-00000.parquet"
        dataframe.to_parquet(shard_path, index=False)


def _date_summary(dataframe: pd.DataFrame, column: str) -> tuple[str, str]:
    """Return min/max date string summary for logging."""
    if dataframe.empty:
        return ("n/a", "n/a")

    date_strings = normalize_to_date_string(dataframe[column])
    return (str(date_strings.min()), str(date_strings.max()))


def fetch_minutes(
    client: Client,
    symbol: str,
    start_date: str,
    end_date: str,
    raw_cache_dir: Path,
    metadata_suffix: str,
    cache_version: int,
    force: bool,
) -> Path:
    """Fetch minute bars for one symbol from ClickHouse ``firstrate.etfs``."""
    output_path = raw_cache_dir / f"{symbol}_minutes.parquet"
    if output_path.exists() and not force:
        LOGGER.info("SKIP %s minutes (exists): %s", symbol, output_path)
        return output_path

    query = f"""
        SELECT symbol, ts, open, high, low, close, volume
        FROM firstrate.etfs
        WHERE symbol = '{symbol}'
          AND ts >= '{start_date}'
          AND ts <= '{end_date}'
        ORDER BY ts
    """
    dataframe = client.query_df(query)
    start_ts, end_ts = _date_summary(dataframe, "ts")
    LOGGER.info("%s minutes: %s rows, %s to %s", symbol, f"{len(dataframe):,}", start_ts, end_ts)

    _save_parquet(dataframe, output_path)
    cache_spec = cache_spec_for_symbol(symbol, "minutes")
    write_sidecar_metadata(
        dataframe=dataframe,
        cache_spec=cache_spec,
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
        metadata_suffix=metadata_suffix,
        cache_version=cache_version,
    )
    return output_path


def fetch_options(
    client: Client,
    symbol: str,
    start_date: str,
    end_date: str,
    raw_cache_dir: Path,
    metadata_suffix: str,
    cache_version: int,
    options_shard_rows: int,
    force: bool,
) -> Path:
    """Fetch option chains for one symbol from ClickHouse ``firstrate.options``."""
    output_path = raw_cache_dir / f"{symbol}_options.parquet"
    if output_path.exists() and not force:
        LOGGER.info("SKIP %s options (exists): %s", symbol, output_path)
        return output_path

    query = f"""
        SELECT symbol, trade_date, strike_price, expiry_date, option_type,
               bid, ask, bid_iv, ask_iv, delta, open_interest, volume
        FROM firstrate.options
        WHERE symbol = '{symbol}'
          AND trade_date >= '{start_date}'
          AND trade_date <= '{end_date}'
        ORDER BY trade_date, expiry_date, strike_price
    """
    dataframe = client.query_df(query)
    start_ts, end_ts = _date_summary(dataframe, "trade_date")
    LOGGER.info("%s options: %s rows, %s to %s", symbol, f"{len(dataframe):,}", start_ts, end_ts)

    _save_parquet_shards(dataframe, output_path, rows_per_shard=options_shard_rows)
    cache_spec = cache_spec_for_symbol(symbol, "options")
    write_sidecar_metadata(
        dataframe=dataframe,
        cache_spec=cache_spec,
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
        metadata_suffix=metadata_suffix,
        cache_version=cache_version,
    )
    return output_path


def fetch_vix(
    client: Client,
    start_date: str,
    end_date: str,
    raw_cache_dir: Path,
    metadata_suffix: str,
    cache_version: int,
    force: bool,
) -> Path:
    """Fetch VIX minute closes and aggregate to daily closes."""
    output_path = raw_cache_dir / "VIX_daily.parquet"
    if output_path.exists() and not force:
        LOGGER.info("SKIP VIX daily (exists): %s", output_path)
        return output_path

    query = f"""
        SELECT ts, close
        FROM firstrate.indices
        WHERE symbol = 'VIX'
          AND ts >= '{start_date}'
          AND ts <= '{end_date}'
        ORDER BY ts
    """
    minute_df = client.query_df(query)
    LOGGER.info("VIX minute rows: %s", f"{len(minute_df):,}")

    if minute_df.empty:
        daily_df = pd.DataFrame(columns=["date", "vix"])
    else:
        minute_df["date"] = pd.to_datetime(minute_df["ts"]).dt.normalize()
        daily_df = minute_df.groupby("date", as_index=False).agg(vix=("close", "last"))

    start_ts, end_ts = _date_summary(daily_df, "date")
    LOGGER.info("VIX daily rows: %s, %s to %s", f"{len(daily_df):,}", start_ts, end_ts)

    _save_parquet(daily_df, output_path)
    cache_spec = cache_spec_for_vix()
    write_sidecar_metadata(
        dataframe=daily_df,
        cache_spec=cache_spec,
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
        metadata_suffix=metadata_suffix,
        cache_version=cache_version,
    )
    return output_path


def build_cache_unavailable_error(
    raw_cache_dir: Path,
    cache_specs: list[CacheSpec],
    metadata_suffix: str,
    validation_issues: list[str],
    db_error: Exception,
) -> str:
    """Build actionable offline failure message for missing/invalid cache."""
    message_lines = [
        "ClickHouse is unavailable and required raw cache files are missing or invalid.",
        f"Required cache directory: {raw_cache_dir}",
        "Required files (parquet + metadata sidecar):",
    ]

    for cache_spec in cache_specs:
        parquet_path = cache_parquet_path(raw_cache_dir, cache_spec)
        sidecar_path = cache_sidecar_path(parquet_path, metadata_suffix)
        message_lines.append(f"- {parquet_path.name}")
        message_lines.append(f"- {sidecar_path.name}")

    if validation_issues:
        message_lines.append("Cache validation issues:")
        for issue in validation_issues:
            message_lines.append(f"- {issue}")

    message_lines.append(f"ClickHouse connection error: {db_error}")
    message_lines.append(
        "Action: restore the required cache files in the directory above or "
        "bring ClickHouse online and rerun `uv run python -m volcast.data.fetch_raw_cache --force`."
    )
    return "\n".join(message_lines)


def refresh_invalid_cache_specs(
    client: Client,
    invalid_specs: list[CacheSpec],
    start_date: str,
    end_date: str,
    raw_cache_dir: Path,
    metadata_suffix: str,
    cache_version: int,
    options_shard_rows: int,
) -> None:
    """Refresh only invalid cache datasets from ClickHouse."""
    shared_fetch_kwargs = {
        "client": client,
        "start_date": start_date,
        "end_date": end_date,
        "raw_cache_dir": raw_cache_dir,
        "metadata_suffix": metadata_suffix,
        "cache_version": cache_version,
        "force": True,
    }

    for cache_spec in invalid_specs:
        match cache_spec.dataset_kind:
            case "minutes":
                if cache_spec.symbol is None:
                    raise ValueError(f"Missing symbol for cache spec {cache_spec.dataset_name}")

                fetch_minutes(symbol=cache_spec.symbol, **shared_fetch_kwargs)
            case "options":
                if cache_spec.symbol is None:
                    raise ValueError(f"Missing symbol for cache spec {cache_spec.dataset_name}")

                fetch_options(
                    symbol=cache_spec.symbol,
                    options_shard_rows=options_shard_rows,
                    **shared_fetch_kwargs,
                )
            case "vix":
                fetch_vix(**shared_fetch_kwargs)
            case _:
                raise ValueError(f"Unknown dataset kind: {cache_spec.dataset_kind}")


def main(force: bool = False) -> None:
    """Run stage 1 raw-cache validation/fetch for configured symbols."""
    configure_logging()
    config = load_config()

    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    raw_cache_dir = resolve_raw_cache_dir(config)
    metadata_suffix = resolve_metadata_suffix(config)
    cache_version = resolve_cache_version(config)
    options_shard_rows = resolve_options_shard_rows(config)
    cache_specs = build_cache_specs(config)

    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    if force:
        invalid_specs = cache_specs
        validation_issues: list[str] = []
    else:
        invalid_specs, validation_issues = validate_required_cache(
            cache_specs=cache_specs,
            raw_cache_dir=raw_cache_dir,
            start_date=start_date,
            end_date=end_date,
            metadata_suffix=metadata_suffix,
            cache_version=cache_version,
        )
        if not invalid_specs:
            LOGGER.info(
                "Stage 1 cache hit: all required Parquet + sidecar files are valid in %s",
                raw_cache_dir,
            )
            return

    LOGGER.info(
        "Stage 1 cache refresh required for %s/%s datasets",
        len(invalid_specs),
        len(cache_specs),
    )
    for issue in validation_issues:
        LOGGER.warning("Cache issue: %s", issue)

    try:
        client = get_clickhouse_client()
    except Exception as error:
        message = build_cache_unavailable_error(
            raw_cache_dir=raw_cache_dir,
            cache_specs=cache_specs,
            metadata_suffix=metadata_suffix,
            validation_issues=validation_issues,
            db_error=error,
        )
        raise RuntimeError(message) from error

    refresh_invalid_cache_specs(
        client=client,
        invalid_specs=invalid_specs,
        start_date=start_date,
        end_date=end_date,
        raw_cache_dir=raw_cache_dir,
        metadata_suffix=metadata_suffix,
        cache_version=cache_version,
        options_shard_rows=options_shard_rows,
    )
    LOGGER.info("Stage 1 complete: raw data cache written to %s", raw_cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Fetch benchmark data from ClickHouse")
    parser.add_argument(
        "--force", action="store_true", help="Re-fetch and overwrite existing files"
    )
    parsed = parser.parse_args()
    main(force=parsed.force)
