"""Raw-cache validation helpers for stage-1 data inputs.

This module defines the raw parquet contract used by the offline pipeline.
Validation is strict so the project can fail loudly on stale or incomplete
cache payloads instead of silently mixing inconsistent datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from volcast.shared.config import configured_symbols

MINUTES_SCHEMA_COLUMNS = ["symbol", "ts", "open", "high", "low", "close", "volume"]
OPTIONS_SCHEMA_COLUMNS = [
    "symbol",
    "trade_date",
    "strike_price",
    "expiry_date",
    "option_type",
    "bid",
    "ask",
    "bid_iv",
    "ask_iv",
    "delta",
    "open_interest",
    "volume",
]
VIX_SCHEMA_COLUMNS = ["date", "vix"]


@dataclass(frozen=True)
class CacheSpec:
    """Metadata contract for one required raw cache dataset."""

    dataset_name: str
    dataset_kind: str
    schema_columns: list[str]
    date_column: str
    symbol: str | None = None


def cache_spec_for_symbol(symbol: str, dataset_kind: str) -> CacheSpec:
    """Build one cache spec for a symbol-level dataset kind.

    Parameters
    ----------
    symbol : str
        ETF symbol, for example ``SPY``.
    dataset_kind : str
        One of ``minutes`` or ``options``.
    """
    if dataset_kind == "minutes":
        return CacheSpec(
            dataset_name=f"{symbol}_minutes",
            dataset_kind="minutes",
            schema_columns=MINUTES_SCHEMA_COLUMNS,
            date_column="ts",
            symbol=symbol,
        )

    if dataset_kind == "options":
        return CacheSpec(
            dataset_name=f"{symbol}_options",
            dataset_kind="options",
            schema_columns=OPTIONS_SCHEMA_COLUMNS,
            date_column="trade_date",
            symbol=symbol,
        )

    raise ValueError(f"Unsupported symbol dataset kind: {dataset_kind}")


def cache_spec_for_vix() -> CacheSpec:
    """Build the cache spec for the shared VIX daily dataset."""
    return CacheSpec(
        dataset_name="VIX_daily",
        dataset_kind="vix",
        schema_columns=VIX_SCHEMA_COLUMNS,
        date_column="date",
    )


def build_cache_specs(config: dict[str, Any]) -> list[CacheSpec]:
    """Build required raw-cache dataset specs from configured symbols."""
    symbols = configured_symbols(config)

    specs: list[CacheSpec] = []
    for symbol in symbols:
        specs.append(cache_spec_for_symbol(symbol, "minutes"))
        specs.append(cache_spec_for_symbol(symbol, "options"))

    specs.append(cache_spec_for_vix())
    return specs


def cache_parquet_path(raw_cache_dir: Path, cache_spec: CacheSpec) -> Path:
    """Build expected parquet path for one cache spec."""
    if cache_spec.dataset_kind == "vix":
        return raw_cache_dir / "VIX_daily.parquet"

    if cache_spec.symbol is None:
        raise ValueError(f"Cache spec for {cache_spec.dataset_name} is missing a symbol")

    return raw_cache_dir / f"{cache_spec.symbol}_{cache_spec.dataset_kind}.parquet"


def cache_sidecar_path(parquet_path: Path, metadata_suffix: str) -> Path:
    """Build sidecar metadata path for one parquet file."""
    return Path(f"{parquet_path}{metadata_suffix}")


def normalize_to_date_string(series: pd.Series) -> pd.Series:
    """Convert datetime-like series to ``YYYY-MM-DD`` strings."""
    datetimes = pd.to_datetime(series)
    if getattr(datetimes.dt, "tz", None) is not None:
        datetimes = datetimes.dt.tz_localize(None)

    return datetimes.dt.normalize().dt.date.astype(str)


def _validation_failure(message: str) -> tuple[bool, str]:
    """Return one failed validation result with a human-readable issue."""
    return (False, message)


def _validate_sidecar_payload(
    payload: dict[str, Any],
    cache_spec: CacheSpec,
    parquet_path: Path,
    sidecar_path: Path,
    start_date: str,
    end_date: str,
    cache_version: int,
) -> tuple[bool, str] | None:
    """Validate sidecar JSON contract. Return a failure tuple or None when valid."""
    expected_keys = {
        "cache_version",
        "dataset_name",
        "parquet_file",
        "schema_columns",
        "date_column",
        "row_count",
        "min_date",
        "max_date",
        "request_start_date",
        "request_end_date",
    }
    if set(payload) != expected_keys:
        return _validation_failure(
            f"sidecar keys mismatch for {sidecar_path.name}: "
            f"expected {sorted(expected_keys)}, got {sorted(payload)}"
        )

    # Compare each sidecar field against the expected cache contract.
    field_checks = [
        (
            payload["cache_version"] == cache_version,
            f"cache version mismatch for {sidecar_path.name}: "
            f"expected {cache_version}, got {payload['cache_version']}",
        ),
        (
            payload["dataset_name"] == cache_spec.dataset_name,
            f"dataset_name mismatch for {sidecar_path.name}: "
            f"expected {cache_spec.dataset_name}, got {payload['dataset_name']}",
        ),
        (
            payload["parquet_file"] == parquet_path.name,
            f"parquet_file mismatch for {sidecar_path.name}: "
            f"expected {parquet_path.name}, got {payload['parquet_file']}",
        ),
        (
            payload["schema_columns"] == cache_spec.schema_columns,
            f"schema_columns mismatch for {sidecar_path.name}: "
            f"expected {cache_spec.schema_columns}, got {payload['schema_columns']}",
        ),
        (
            payload["date_column"] == cache_spec.date_column,
            f"date_column mismatch for {sidecar_path.name}: "
            f"expected {cache_spec.date_column}, got {payload['date_column']}",
        ),
        (
            payload["request_start_date"] == start_date
            and payload["request_end_date"] == end_date,
            f"request date range mismatch for {sidecar_path.name}: "
            f"expected {start_date}..{end_date}, "
            f"got {payload['request_start_date']}..{payload['request_end_date']}",
        ),
    ]
    for is_valid, message in field_checks:
        if not is_valid:
            return _validation_failure(message)

    return None


def build_sidecar_payload(
    dataframe: pd.DataFrame,
    cache_spec: CacheSpec,
    output_path: Path,
    start_date: str,
    end_date: str,
    cache_version: int,
) -> dict[str, Any]:
    """Build JSON payload for one raw parquet dataset sidecar."""
    if cache_spec.date_column not in dataframe.columns:
        raise ValueError(
            f"Cannot build sidecar for {cache_spec.dataset_name}: "
            f"missing date column {cache_spec.date_column!r}"
        )

    date_strings = normalize_to_date_string(dataframe[cache_spec.date_column])
    min_date = str(date_strings.min())
    max_date = str(date_strings.max())

    return {
        "cache_version": cache_version,
        "dataset_name": cache_spec.dataset_name,
        "parquet_file": output_path.name,
        "schema_columns": cache_spec.schema_columns,
        "date_column": cache_spec.date_column,
        "row_count": len(dataframe),
        "min_date": min_date,
        "max_date": max_date,
        "request_start_date": start_date,
        "request_end_date": end_date,
    }


def write_sidecar_metadata(
    dataframe: pd.DataFrame,
    cache_spec: CacheSpec,
    output_path: Path,
    start_date: str,
    end_date: str,
    metadata_suffix: str,
    cache_version: int,
) -> Path:
    """Write sidecar metadata JSON for one parquet cache dataset."""
    payload = build_sidecar_payload(
        dataframe=dataframe,
        cache_spec=cache_spec,
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
        cache_version=cache_version,
    )

    sidecar_path = cache_sidecar_path(output_path, metadata_suffix)
    sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sidecar_path


def validate_cache_spec(
    cache_spec: CacheSpec,
    raw_cache_dir: Path,
    start_date: str,
    end_date: str,
    metadata_suffix: str,
    cache_version: int,
) -> tuple[bool, str]:
    """Validate parquet and sidecar metadata for one required cache dataset."""
    parquet_path = cache_parquet_path(raw_cache_dir, cache_spec)
    sidecar_path = cache_sidecar_path(parquet_path, metadata_suffix)

    if not parquet_path.exists():
        return _validation_failure(f"missing parquet file: {parquet_path.name}")

    if cache_spec.dataset_kind == "options" and parquet_path.is_file():
        return _validation_failure(
            f"options cache must be sharded directory, found single file: {parquet_path.name}"
        )

    if cache_spec.dataset_kind == "options" and parquet_path.is_dir():
        shard_paths = sorted(parquet_path.glob("*.parquet"))
        if not shard_paths:
            return _validation_failure(
                f"missing options shard files in directory: {parquet_path.name}"
            )

    if not sidecar_path.exists():
        return _validation_failure(f"missing metadata sidecar: {sidecar_path.name}")

    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as error:  # pragma: no cover - defensive parse path
        return _validation_failure(f"invalid sidecar JSON for {sidecar_path.name}: {error}")

    sidecar_issue = _validate_sidecar_payload(
        payload=payload,
        cache_spec=cache_spec,
        parquet_path=parquet_path,
        sidecar_path=sidecar_path,
        start_date=start_date,
        end_date=end_date,
        cache_version=cache_version,
    )
    if sidecar_issue is not None:
        return sidecar_issue

    try:
        dataframe = pd.read_parquet(parquet_path)
    except Exception as error:  # pragma: no cover - defensive parse path
        return _validation_failure(f"failed to read parquet file {parquet_path.name}: {error}")

    if dataframe.columns.tolist() != cache_spec.schema_columns:
        return _validation_failure(
            f"parquet schema mismatch for {parquet_path.name}: "
            f"expected {cache_spec.schema_columns}, got {dataframe.columns.tolist()}"
        )

    if len(dataframe) != int(payload["row_count"]):
        return _validation_failure(
            f"row_count mismatch for {parquet_path.name}: "
            f"expected {payload['row_count']}, got {len(dataframe)}"
        )

    if dataframe.empty:
        return _validation_failure(
            f"empty parquet file is not valid cache input: {parquet_path.name}"
        )

    date_strings = normalize_to_date_string(dataframe[cache_spec.date_column])
    min_date = str(date_strings.min())
    max_date = str(date_strings.max())

    if payload["min_date"] != min_date or payload["max_date"] != max_date:
        return _validation_failure(
            f"date summary mismatch for {sidecar_path.name}: "
            f"expected {payload['min_date']}..{payload['max_date']}, got {min_date}..{max_date}"
        )

    return (True, "")


def validate_required_cache(
    cache_specs: list[CacheSpec],
    raw_cache_dir: Path,
    start_date: str,
    end_date: str,
    metadata_suffix: str,
    cache_version: int,
) -> tuple[list[CacheSpec], list[str]]:
    """Return invalid cache specs and human-readable validation issues."""
    invalid_specs: list[CacheSpec] = []
    issues: list[str] = []

    for cache_spec in cache_specs:
        is_valid, issue = validate_cache_spec(
            cache_spec=cache_spec,
            raw_cache_dir=raw_cache_dir,
            start_date=start_date,
            end_date=end_date,
            metadata_suffix=metadata_suffix,
            cache_version=cache_version,
        )
        if not is_valid:
            invalid_specs.append(cache_spec)
            issues.append(issue)

    return (invalid_specs, issues)
