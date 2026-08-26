# vol-forecast-benchmarks

A reproducible benchmark comparing volatility forecasting models on SPY. It
runs offline from checked-in parquet data by default, so it can be cloned and
executed without any external service.

## What it does

The pipeline forecasts next-day and next-5-day realised variance of SPY from:

- minute-bar realised variance (RV), computed with close-to-close, Parkinson,
  and Garman-Klass estimators,
- option-implied features (ATM implied volatility, skew, term slope, IV-RV
  spread), and
- lagged VIX as market-wide context.

Five models are benchmarked: HAR-RV (Heterogeneous Autoregressive Realised
Volatility), GARCH(1,1) (Generalized Autoregressive Conditional
Heteroskedasticity), Ridge, Lasso, and XGBoost. Evaluation uses expanding-window
walk-forward validation with periodic retraining, and features are lagged so
date-$t$ predictors only use information available by $t-1$ (no lookahead).
QLIKE (Quasi-Likelihood loss) is the primary scoring metric, with MSE and
pairwise Diebold-Mariano significance tests also computed. See
`docs/reference/evaluation_protocol.md` for the full protocol and
`GUIDE_OVERVIEW.md` for the target definition and design notes.

## Requirements

- Python 3.13
- `uv`
- ClickHouse, only if you need to refresh the raw data cache. The default
  offline run does not need it. Connection settings are read from `.env`:
  `CLICKHOUSE_HOST`, `CLICKHOUSE_PORT`, `CLICKHOUSE_USER`,
  `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_SECURE`, `CLICKHOUSE_VERIFY`.

## Setup

```bash
uv sync
```

## Usage

Run the full offline pipeline (stages 2-4, starting from the checked-in raw
cache in `data/raw`):

```bash
uv run python -m volcast.pipeline.run_pipeline
```

Or run each stage separately:

```bash
uv run python -m volcast.features.compute_rv        # minute bars -> realised variance
uv run python -m volcast.features.build_features     # RV + options + VIX -> feature matrix
uv run python -m volcast.evaluation.train_evaluate    # walk-forward training and scoring
```

Refresh the raw cache from ClickHouse (optional, requires DB access):

```bash
uv run python -m volcast.data.fetch_raw_cache --force
```

Run tests:

```bash
uv run --extra dev pytest -q
```

## Configuration

All settings live in `config.toml`: symbols, date range, RV estimators and HAR
lags, option-feature parameters (ATM delta, skew delta, DTE range), forecast
horizons, initial training window, retraining cadence, and which models are
enabled. The checked-in default is a one-calendar-year initial training window
over SPY, 2022-10-01 to 2024-12-31, leaving 286 out-of-sample dates. A longer
window requires a correspondingly longer raw-data history; otherwise stage 4
raises `InsufficientTrainingHistoryError` instead of writing empty results.

## Layout

```
src/volcast/       importable package: data, features, evaluation, models, pipeline, shared
data/raw/          checked-in raw parquet cache (SPY minutes, SPY options, VIX daily) + metadata sidecars
data/processed/    computed RV and feature matrices
outputs/           forecasts, scores, DM tests, model diagnostics
notebooks/         offline_pipeline_demo.ipynb, a teaching walkthrough of the full pipeline
docs/reference/    data dictionary, evaluation protocol, offline workflow, raw cache contract
```

## Output

- `outputs/forecasts.parquet`: per-date out-of-sample predictions.
- `outputs/scores.parquet`: aggregate QLIKE and MSE by symbol, horizon, and model.
- `outputs/dm_tests.parquet`: pairwise Diebold-Mariano test results.
- `outputs/model_diagnostics.parquet`: floor-hit rates and tail-loss behavior per model.
