"""Offline-first cache policy tests for stage-1 data fetching.

This module validates the cache contract for raw ClickHouse inputs:
- Valid Parquet plus sidecar metadata must bypass database access.
- Missing or invalid cache files must trigger database refresh when available.
- Missing or invalid cache files must fail clearly when the database is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import volcast.data.fetch_raw_cache as fetch_data


def _test_config(raw_cache_dir: Path) -> dict:
    """Build a minimal config fixture for one-symbol cache tests."""
    return {
        "data": {
            "symbols": ["SPY"],
            "robustness_symbols": [],
            "start_date": "2020-01-01",
            "end_date": "2020-01-03",
        },
        "cache": {
            "raw_cache_dir": str(raw_cache_dir),
            "metadata_suffix": ".metadata.json",
            "version": 1,
            "options_shard_rows": 2,
        },
    }


def _sidecar_path(parquet_path: Path, metadata_suffix: str) -> Path:
    """Return the configured sidecar path for one parquet file."""
    return Path(f"{parquet_path}{metadata_suffix}")


def _write_sidecar(
    *,
    parquet_path: Path,
    metadata_suffix: str,
    dataset_name: str,
    schema_columns: list[str],
    date_column: str,
    row_count: int,
    min_date: str,
    max_date: str,
    start_date: str,
    end_date: str,
    version: int,
) -> None:
    """Persist sidecar metadata that should pass cache validation."""
    sidecar_path = _sidecar_path(parquet_path, metadata_suffix)
    payload = {
        "cache_version": version,
        "dataset_name": dataset_name,
        "parquet_file": parquet_path.name,
        "schema_columns": schema_columns,
        "date_column": date_column,
        "row_count": row_count,
        "min_date": min_date,
        "max_date": max_date,
        "request_start_date": start_date,
        "request_end_date": end_date,
    }
    sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_valid_cache_files(raw_cache_dir: Path, config: dict) -> None:
    """Create valid parquet and sidecar files for all required cache datasets."""
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    minute_df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY"],
            "ts": pd.to_datetime(
                ["2020-01-01 09:30:00", "2020-01-02 09:30:00", "2020-01-03 09:30:00"]
            ),
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [100, 110, 120],
        }
    )
    minutes_path = raw_cache_dir / "SPY_minutes.parquet"
    minute_df.to_parquet(minutes_path, index=False)

    options_df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY"],
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "strike_price": [300.0, 301.0, 302.0],
            "expiry_date": pd.to_datetime(["2020-02-01", "2020-02-01", "2020-02-01"]),
            "option_type": ["C", "C", "P"],
            "bid": [1.0, 1.1, 1.2],
            "ask": [1.1, 1.2, 1.3],
            "bid_iv": [0.20, 0.21, 0.22],
            "ask_iv": [0.21, 0.22, 0.23],
            "delta": [0.50, 0.49, -0.25],
            "open_interest": [1000, 1200, 900],
            "volume": [100, 150, 80],
        }
    )
    options_path = raw_cache_dir / "SPY_options.parquet"
    options_path.mkdir(parents=True, exist_ok=True)
    options_shard_path = options_path / "part-00000.parquet"
    options_df.to_parquet(options_shard_path, index=False)

    vix_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "vix": [14.0, 15.0, 16.0],
        }
    )
    vix_path = raw_cache_dir / "VIX_daily.parquet"
    vix_df.to_parquet(vix_path, index=False)

    metadata_suffix = config["cache"]["metadata_suffix"]
    version = config["cache"]["version"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]

    _write_sidecar(
        parquet_path=minutes_path,
        metadata_suffix=metadata_suffix,
        dataset_name="SPY_minutes",
        schema_columns=["symbol", "ts", "open", "high", "low", "close", "volume"],
        date_column="ts",
        row_count=len(minute_df),
        min_date="2020-01-01",
        max_date="2020-01-03",
        start_date=start_date,
        end_date=end_date,
        version=version,
    )
    _write_sidecar(
        parquet_path=options_path,
        metadata_suffix=metadata_suffix,
        dataset_name="SPY_options",
        schema_columns=[
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
        ],
        date_column="trade_date",
        row_count=len(options_df),
        min_date="2020-01-01",
        max_date="2020-01-03",
        start_date=start_date,
        end_date=end_date,
        version=version,
    )
    _write_sidecar(
        parquet_path=vix_path,
        metadata_suffix=metadata_suffix,
        dataset_name="VIX_daily",
        schema_columns=["date", "vix"],
        date_column="date",
        row_count=len(vix_df),
        min_date="2020-01-01",
        max_date="2020-01-03",
        start_date=start_date,
        end_date=end_date,
        version=version,
    )


class _FakeClient:
    """Fake ClickHouse client returning deterministic test frames."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def query_df(self, query: str) -> pd.DataFrame:
        """Return table-specific fixture data for query text."""
        self.queries.append(query)

        if "firstrate.etfs" in query:
            return pd.DataFrame(
                {
                    "symbol": ["SPY", "SPY", "SPY"],
                    "ts": pd.to_datetime(
                        ["2020-01-01 09:30:00", "2020-01-02 09:30:00", "2020-01-03 09:30:00"]
                    ),
                    "open": [10.0, 10.1, 10.2],
                    "high": [10.2, 10.3, 10.4],
                    "low": [9.9, 10.0, 10.1],
                    "close": [10.1, 10.2, 10.3],
                    "volume": [100, 110, 120],
                }
            )

        if "firstrate.options" in query:
            return pd.DataFrame(
                {
                    "symbol": ["SPY", "SPY", "SPY"],
                    "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                    "strike_price": [300.0, 301.0, 302.0],
                    "expiry_date": pd.to_datetime(["2020-02-01", "2020-02-01", "2020-02-01"]),
                    "option_type": ["C", "C", "P"],
                    "bid": [1.0, 1.1, 1.2],
                    "ask": [1.1, 1.2, 1.3],
                    "bid_iv": [0.20, 0.21, 0.22],
                    "ask_iv": [0.21, 0.22, 0.23],
                    "delta": [0.50, 0.49, -0.25],
                    "open_interest": [1000, 1200, 900],
                    "volume": [100, 150, 80],
                }
            )

        if "firstrate.indices" in query:
            return pd.DataFrame(
                {
                    "ts": pd.to_datetime(
                        ["2020-01-01 16:00:00", "2020-01-02 16:00:00", "2020-01-03 16:00:00"]
                    ),
                    "close": [14.0, 15.0, 16.0],
                }
            )

        raise ValueError(f"Unexpected query: {query}")


def test_cache_hit_uses_local_parquet_without_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Valid cache should satisfy stage-1 without creating a DB client."""
    raw_cache_dir = tmp_path / "raw"
    config = _test_config(raw_cache_dir)
    _build_valid_cache_files(raw_cache_dir, config)

    monkeypatch.setattr(fetch_data, "load_config", lambda: config)

    def _db_call_not_allowed() -> None:
        raise AssertionError("ClickHouse must not be called for a valid cache hit")

    monkeypatch.setattr(fetch_data, "get_clickhouse_client", _db_call_not_allowed)

    fetch_data.main(force=False)


def test_cache_miss_queries_db_and_writes_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing cache should query DB and create parquet plus sidecar files."""
    raw_cache_dir = tmp_path / "raw"
    config = _test_config(raw_cache_dir)
    fake_client = _FakeClient()

    monkeypatch.setattr(fetch_data, "load_config", lambda: config)
    monkeypatch.setattr(fetch_data, "get_clickhouse_client", lambda: fake_client)

    fetch_data.main(force=False)

    parquet_files = [
        raw_cache_dir / "SPY_minutes.parquet",
        raw_cache_dir / "SPY_options.parquet",
        raw_cache_dir / "VIX_daily.parquet",
    ]
    sidecar_files = [
        _sidecar_path(parquet_path, config["cache"]["metadata_suffix"])
        for parquet_path in parquet_files
    ]

    for parquet_path in parquet_files:
        assert parquet_path.exists()

    options_shards = sorted((raw_cache_dir / "SPY_options.parquet").glob("*.parquet"))
    assert options_shards

    for sidecar_path in sidecar_files:
        assert sidecar_path.exists()

    assert len(fake_client.queries) == 3


def test_invalid_cache_triggers_refresh_and_sidecar_rewrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invalid sidecar metadata should force DB refresh and sidecar rewrite."""
    raw_cache_dir = tmp_path / "raw"
    config = _test_config(raw_cache_dir)
    _build_valid_cache_files(raw_cache_dir, config)

    minutes_path = raw_cache_dir / "SPY_minutes.parquet"
    minutes_sidecar = _sidecar_path(minutes_path, config["cache"]["metadata_suffix"])
    invalid_payload = json.loads(minutes_sidecar.read_text(encoding="utf-8"))
    invalid_payload["row_count"] = 999_999
    minutes_sidecar.write_text(json.dumps(invalid_payload, indent=2), encoding="utf-8")

    fake_client = _FakeClient()
    monkeypatch.setattr(fetch_data, "load_config", lambda: config)
    monkeypatch.setattr(fetch_data, "get_clickhouse_client", lambda: fake_client)

    fetch_data.main(force=False)

    rewritten_payload = json.loads(minutes_sidecar.read_text(encoding="utf-8"))
    assert rewritten_payload["row_count"] == 3
    assert any("firstrate.etfs" in query for query in fake_client.queries)


def test_missing_or_invalid_cache_with_db_down_fails_with_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DB outage with incomplete cache must raise a message naming required files."""
    raw_cache_dir = tmp_path / "raw"
    config = _test_config(raw_cache_dir)
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(fetch_data, "load_config", lambda: config)

    def _raise_db_down() -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr(fetch_data, "get_clickhouse_client", _raise_db_down)

    with pytest.raises(RuntimeError) as error_info:
        fetch_data.main(force=False)

    error_text = str(error_info.value)
    assert str(raw_cache_dir) in error_text
    assert "SPY_minutes.parquet" in error_text
    assert "SPY_minutes.parquet.metadata.json" in error_text
    assert "SPY_options.parquet" in error_text
    assert "VIX_daily.parquet" in error_text
