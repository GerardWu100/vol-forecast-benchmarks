# Part 1: Conceptual Explanation

The `src/volcast/data/` folder owns the raw-cache boundary between external data access
and the rest of the benchmark. Its job is not feature engineering or modeling.
Its job is to answer one question clearly:

**Do we have a valid offline raw parquet payload for this run?**

This separation matters because the project is designed to run offline by
default. Stage 1 can refresh cache files from ClickHouse, but stages 2-4 should
work with no database connection once the cache is valid.

The cache contract is strict by design. Every required parquet dataset must
match its metadata sidecar exactly (schema, row count, date range, and cache
version). If one file is broken, stage 1 refreshes only that dataset when
ClickHouse is available. If ClickHouse is unavailable, stage 1 fails with an
actionable error that names the missing or invalid artifacts.

# Part 2: Code Reference

- `src/volcast/data/fetch_raw_cache.py`
  Stage 1 entrypoint. Validates required cache files and refreshes invalid
  datasets from ClickHouse when needed.

- `src/volcast/data/cache_validation.py`
  Cache contract definitions and validation logic, including sidecar payload
  construction and schema/date checks.

Where to start in code:

1. `src/volcast/data/cache_validation.py`
2. `src/volcast/data/fetch_raw_cache.py`

# Part 3: Short Journal

- 2026-04-19: Moved stage-1 cache logic into `src/volcast/data/` and extracted contract
  validation into `cache_validation.py` so offline policy is explicit and easier
  to explain in interviews.
