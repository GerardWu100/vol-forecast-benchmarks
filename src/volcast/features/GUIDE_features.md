# Part 1: Conceptual Explanation

The `src/volcast/features/` folder turns raw market data into leakage-aware forecast
inputs. It answers two interview-critical questions:

1. **How is realised variance constructed from minute bars?**
2. **How are option and VIX signals aligned so they do not leak future data?**

Stage 2 (`compute_rv.py`) aggregates minute bars into daily realised variance
estimators (close-to-close, Parkinson, Garman-Klass), adds HAR lags, and builds
forward targets.

Stage 3 (`build_features.py`) merges lagged option-derived signals and lagged
VIX with stage-2 outputs, computes IV-RV spread, and drops rows with missing core
fields after bounded forward fill.

The folder keeps all heavy data transformation logic in Python modules so the
notebook can focus on teaching and inspection.

# Part 2: Code Reference

- `src/volcast/features/rv_estimators.py`
  Low-level realised variance formulas and HAR lag builder.

- `src/volcast/features/compute_rv.py`
  Stage 2 runner: minute parquet -> daily RV table with targets.

- `src/volcast/features/option_features.py`
  Daily option-chain feature extraction (ATM IV, skew, term slope).

- `src/volcast/features/build_features.py`
  Stage 3 runner: joins RV, option features, and lagged VIX into final model
  matrix.

Where to start in code:

1. `src/volcast/features/rv_estimators.py`
2. `src/volcast/features/compute_rv.py`
3. `src/volcast/features/build_features.py`

# Part 3: Short Journal

- 2026-04-19: Reorganized stage-2 and stage-3 logic under `src/volcast/features/` to
  make realised-variance construction and no-lookahead feature timing easier to
  navigate.
