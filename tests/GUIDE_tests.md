# Part 1: Conceptual Explanation

The `tests/` folder protects the benchmark against the highest-impact failure
modes:

- wrong variance formulas,
- lookahead leakage,
- broken raw-cache validation semantics,
- unstable model wrappers,
- pipeline integration paths that only work with precomputed artifacts.

The suite intentionally mixes unit and integration-style checks. Unit tests pin
mathematical functions. Smoke tests run stage 2 through stage 4 on small local
parquet fixtures so offline execution remains verified.

The strongest interview signal is the no-lookahead module. Temporal alignment is
the easiest way to accidentally break a volatility benchmark, so it has explicit
tests.

# Part 2: Code Reference

- `test_rv_estimators.py`
  Formula checks for close-to-close, Parkinson, Garman-Klass, and HAR lag logic.

- `test_option_features.py`
  Selection logic for ATM implied volatility, skew, and term slope.

- `test_evaluation.py`
  QLIKE, MSE, and Diebold-Mariano behavior checks.

- `test_models.py`
  Interface and behavior checks for HAR, GARCH, Ridge, Lasso, and XGBoost.

- `test_fetch_data_cache.py`
  Stage-1 cache contract tests for cache hit, cache miss, metadata invalidation,
  and DB-down error messaging.

- `test_no_lookahead.py`
  Artifact-based checks for feature and target temporal alignment.

- `test_train_evaluate.py`
  Diagnostics checks and toy stage-2-to-4 offline smoke path.

Where to start in code:

1. `test_no_lookahead.py`
2. `test_train_evaluate.py`
3. `test_fetch_data_cache.py`

# Part 3: Short Journal

- 2026-04-19: Updated smoke coverage from stage-2-to-5 to stage-2-to-4 after
  removing the HTML report stage from the official pipeline.
