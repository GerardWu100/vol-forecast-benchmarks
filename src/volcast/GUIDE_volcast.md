# Part 1: Conceptual Explanation

The `volcast` package is organized by interview-facing responsibilities rather
than by pipeline stage numbers alone:

- `src/volcast/data/`: raw cache contract and ClickHouse refresh path,
- `src/volcast/features/`: realised-variance and feature construction,
- `src/volcast/evaluation/`: walk-forward scoring and diagnostics,
- `src/volcast/models/`: model family behavior behind one interface,
- `src/volcast/pipeline/`: stage orchestration,
- `src/volcast/shared/`: config and path helpers used across modules.

This structure keeps each module focused on one question and makes it easier to
explain the full benchmark from first principles.

# Part 2: Code Reference

- `src/volcast/data/`
  Raw parquet contract, sidecar validation, and cache refresh entrypoint.

- `src/volcast/features/`
  Stage 2 and stage 3 transformations from raw inputs to model matrix.

- `src/volcast/evaluation/`
  Stage 4 walk-forward engine and benchmark metrics.

- `src/volcast/models/`
  HAR, GARCH, linear, and XGBoost model wrappers with shared interface.

- `src/volcast/pipeline/`
  Four-stage orchestrator entrypoint for end-to-end execution.

- `src/volcast/shared/`
  Cross-cutting configuration, logging setup, and filesystem helpers.

- `src/volcast/shared/logging.py`
  Shared `configure_logging()` used by stage entrypoints and the pipeline orchestrator.

Where to start in code:

1. `src/volcast/pipeline/run_pipeline.py`
2. `src/volcast/features/compute_rv.py`
3. `src/volcast/features/build_features.py`
4. `src/volcast/evaluation/train_evaluate.py`

# Part 3: Short Journal

- 2026-04-19: Reorganized flat stage scripts into subfolders so each file maps
  cleanly to one interview question (cache contract, RV math, feature timing,
  or walk-forward evaluation).
