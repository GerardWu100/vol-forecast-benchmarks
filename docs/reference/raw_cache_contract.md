# Raw Cache Contract

This document defines the offline raw-cache payload required for the default
benchmark run.

## Portable Payload

Required versioned artifacts:

- `data/raw/SPY_minutes.parquet`
- `data/raw/SPY_minutes.parquet.metadata.json`
- `data/raw/SPY_options.parquet/part-*.parquet`
- `data/raw/SPY_options.parquet.metadata.json`
- `data/raw/VIX_daily.parquet`
- `data/raw/VIX_daily.parquet.metadata.json`

## Date Window And Scope

- `request_start_date`: `2022-10-01`
- `request_end_date`: `2024-12-31`
- default symbols in `config.toml`: `SPY` primary, no robustness symbols

The window is chosen to keep `data/raw` under approximately 100 MB while still
providing enough history for train/test separation.

## Sidecar Requirements

Each sidecar JSON must include exactly these keys:

- `cache_version`
- `dataset_name`
- `parquet_file`
- `schema_columns`
- `date_column`
- `row_count`
- `min_date`
- `max_date`
- `request_start_date`
- `request_end_date`

Validation checks compare sidecar values against actual parquet content.

## Stage-1 Behavior

- If all required parquet + sidecars validate: stage 1 is a cache hit and exits.
- If some datasets fail validation and ClickHouse is reachable: stage 1 refreshes
  only failed datasets.
- If validation fails and ClickHouse is unreachable: stage 1 raises a clear error
  naming required files and validation issues.

## ClickHouse Policy

ClickHouse is optional and used only for cache refresh. The default pipeline and
notebook execution paths must run without ClickHouse when `data/raw` is present.
