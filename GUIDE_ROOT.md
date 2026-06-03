# Part 1: Conceptual Explanation

The repository root is the coordination layer around an offline-first,
interview-defensible volatility forecasting benchmark. The workflow is designed
so a cloned repo can run from local parquet data without database access.

The quantitative flow is four stages:

1. validate or refresh raw cache,
2. construct realised variance from minute bars,
3. build leakage-aware features from RV, options, and VIX,
4. run walk-forward model evaluation.

The most important design choice at the root is that the default path starts
from stage 2. Stage 1 exists for cache refresh only when ClickHouse is
available.

# Part 2: Code Reference

- `README.md`
  Main project story, offline quick start, and benchmark outputs.

- `config.toml`
  Single configuration file for symbols, date range, cache settings, feature
  settings, horizons, and enabled models.

- `src/volcast/`
  Importable package, organized by responsibility (`data`, `features`,
  `evaluation`, `models`, `pipeline`, `shared`).

- `tests/`
  Unit and smoke tests for formulas, alignment, cache behavior, and stage-2-to-4
  offline execution.

- `data/raw/`
  Versioned portable raw cache payload (SPY minutes/options and VIX daily).

- `notebooks/offline_pipeline_demo.ipynb`
  Teaching notebook that walks through the full offline pipeline from raw
  parquet to stage-4 evaluation outputs.

- `docs/reference/`
  Ground-truth documentation for raw cache contract, workflow, and data
  dictionary.

Where to start in code:

1. `README.md`
2. `GUIDE_OVERVIEW.md`
3. `src/volcast/pipeline/run_pipeline.py`
4. `src/volcast/evaluation/train_evaluate.py`
5. `tests/test_train_evaluate.py`

# Part 3: Short Journal

- 2026-04-19: Removed the HTML report surface and rewired the project around a
  four-stage offline-first workflow with a teaching notebook as the main
  presentation artifact.
