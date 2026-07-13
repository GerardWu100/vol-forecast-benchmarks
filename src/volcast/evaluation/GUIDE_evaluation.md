# Part 1: Conceptual Explanation

The `src/volcast/evaluation/` folder owns the out-of-sample benchmark logic. It answers
one core question:

**Given leakage-aware features, which model family forecasts realised variance
best under walk-forward evaluation?**

This folder keeps scoring separate from feature engineering and model
implementations. That separation makes assumptions explicit:

- training windows are expanding over time,
- the portable initial window is one calendar year,
- retraining cadence is controlled by config,
- primary score is QLIKE,
- statistical model comparison uses Diebold-Mariano tests.

Before fitting, the evaluator checks that the requested calendar burn-in leaves
at least one out-of-sample row and at least 25 training observations. An
infeasible request raises an error with the available and required dates. This
prevents empty output tables from being mistaken for a completed benchmark.

It also exports model diagnostics (for example floor-hit rates) so model quality
is not reduced to one leaderboard number.

# Part 2: Code Reference

- `src/volcast/evaluation/metrics.py`
  QLIKE, MSE, and Diebold-Mariano implementations.

- `src/volcast/evaluation/train_evaluate.py`
  Stage 4 walk-forward engine, training-window feasibility check, diagnostics,
  and output writers. `InsufficientTrainingHistoryError` identifies a mismatch
  between configured burn-in and available features.

Where to start in code:

1. `src/volcast/evaluation/metrics.py`
2. `src/volcast/evaluation/train_evaluate.py`

# Part 3: Short Journal

- 2026-04-19: Split evaluation metrics from stage-4 runner and moved both into
  `src/volcast/evaluation/` to make benchmark methodology easier to present and audit.
- 2026-07-13: Stage 4 now raises on an infeasible initial calendar window instead
  of writing schema-correct but empty benchmark outputs.
