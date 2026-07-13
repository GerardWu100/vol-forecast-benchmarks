# Offline Workflow

This workflow is the default path for reproducing the benchmark from a clone.

## 1) Install Dependencies

```bash
uv sync
```

## 2) Run Offline Pipeline (Stages 2-4)

```bash
uv run python -m volcast.pipeline.run_pipeline
```

By default this starts at stage 2 and assumes valid raw parquet exists in
`data/raw`.

The portable configuration reserves one calendar year for initial training and
then evaluates the remaining 286 feature dates with an expanding window. Models
are refitted every 21 trading days. If the configured initial period extends
beyond the available feature history, stage 4 raises
`InsufficientTrainingHistoryError` and does not write empty benchmark evidence.

## 3) Optional Explicit Stage Execution

```bash
uv run python -m volcast.features.compute_rv
uv run python -m volcast.features.build_features
uv run python -m volcast.evaluation.train_evaluate
```

## 4) Execute Teaching Notebook

```bash
uv run python -m jupyter nbconvert --to notebook --execute notebooks/offline_pipeline_demo.ipynb --output /tmp/offline_pipeline_demo.executed.ipynb
```

## 5) Run Tests

```bash
uv run --extra dev pytest -q
```

## Optional: Refresh Raw Cache With ClickHouse

Use only when DB access is available:

```bash
uv run python -m volcast.data.fetch_raw_cache --force
```

This step is not required for the default offline clone-and-run flow.

## Longer Research Windows

To use an initial window longer than one year, refresh or replace the raw cache
with enough earlier history before increasing `forecast.initial_train_years` in
`config.toml`. The first evaluation date must occur both after the requested
calendar span and after the minimum 25 training observations.
