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
uv run python -m pytest -q
```

## Optional: Refresh Raw Cache With ClickHouse

Use only when DB access is available:

```bash
uv run python -m volcast.data.fetch_raw_cache --force
```

This step is not required for the default offline clone-and-run flow.
