# VolCast: Offline Volatility Forecast Benchmarks

VolCast is a reproducible benchmark for forecasting next-day and next-week
realised variance from:

- minute-bar realised variance features,
- option-implied volatility features, and
- market-wide context from VIX.

The repository is built as a portable interview project. The default workflow is
offline and starts from versioned parquet in `data/raw`.

## Why This Project Is Worth Reviewing

- **Academic baselines first**: HAR-RV (Heterogeneous Autoregressive Realised
  Volatility) and GARCH(1,1) (Generalized Autoregressive Conditional
  Heteroskedasticity) are included as reference models.
- **Leakage-aware timing**: stage 3 lags options and VIX so date $t$ features use
  information available by $t-1$.
- **Walk-forward evaluation**: stage 4 uses an expanding training window with
  periodic retraining, not random shuffles.
- **Variance-native scoring**: QLIKE (Quasi-Likelihood loss) is the primary
  metric.
- **Offline reproducibility**: the checked-in cache is validated through sidecar
  metadata, so stages 2-4 can run without ClickHouse.
- **Model diagnostics**: floor-hit rates and tail-loss behavior are exported in
  `outputs/model_diagnostics.parquet`.

## Current Default Scope

- **Primary asset**: `SPY`
- **Market context**: `VIX`
- **Date range**: `2022-10-01` to `2024-12-31`
- **Forecast horizons**: 1 trading day and 5 trading days
- **Forecast target**: close-to-close realised variance

## Four-Stage Pipeline

| Stage | Script | Input | Output |
|---|---|---|---|
| 1. Raw cache (optional) | `src/volcast/data/fetch_raw_cache.py` | ClickHouse or valid raw cache | `data/raw/*.parquet` + metadata sidecars |
| 2. RV | `src/volcast/features/compute_rv.py` | Minute bars | `data/processed/*_rv.parquet` |
| 3. Features | `src/volcast/features/build_features.py` | RV + options + VIX | `data/processed/*_features.parquet` |
| 4. Train/Evaluate | `src/volcast/evaluation/train_evaluate.py` | Feature matrices | `outputs/forecasts.parquet`, `outputs/scores.parquet`, `outputs/dm_tests.parquet`, `outputs/model_diagnostics.parquet` |

## Quick Start (Offline Default)

Install dependencies:

```bash
uv sync
```

Run the default offline pipeline from stage 2:

```bash
uv run python -m volcast.pipeline.run_pipeline
```

Run tests:

```bash
uv run python -m pytest -q
```

## Stage Commands (Manual)

```bash
uv run python -m volcast.features.compute_rv
uv run python -m volcast.features.build_features
uv run python -m volcast.evaluation.train_evaluate
```

## Raw Cache Contract

The portable default cache payload is:

- `data/raw/SPY_minutes.parquet`
- `data/raw/SPY_minutes.parquet.metadata.json`
- `data/raw/SPY_options.parquet/part-*.parquet`
- `data/raw/SPY_options.parquet.metadata.json`
- `data/raw/VIX_daily.parquet`
- `data/raw/VIX_daily.parquet.metadata.json`

Stage 1 validates both parquet data and sidecar metadata (schema, row count,
date bounds, and request range). If the cache is valid, stage 1 is a no-op.

## Optional: Refresh Raw Cache With ClickHouse

Use this only when database access is available and you want to refresh the raw
cache package:

```bash
uv run python -m volcast.data.fetch_raw_cache --force
```

## Notebook Teaching Artifact

The primary walkthrough notebook is:

- `notebooks/offline_pipeline_demo.ipynb`

It demonstrates the full offline flow from `data/raw` through stage 4 outputs.

## Outputs That Matter

- `outputs/forecasts.parquet`: per-date out-of-sample predictions.
- `outputs/scores.parquet`: aggregate score table by symbol, horizon, and model.
- `outputs/dm_tests.parquet`: pairwise Diebold-Mariano results.
- `outputs/model_diagnostics.parquet`: floor-hit rates and tail-risk diagnostics.

## Interview-Oriented Reading Order

1. `README.md`
2. `GUIDE_OVERVIEW.md`
3. `src/volcast/features/build_features.py`
4. `src/volcast/evaluation/train_evaluate.py`
5. `tests/test_no_lookahead.py`
