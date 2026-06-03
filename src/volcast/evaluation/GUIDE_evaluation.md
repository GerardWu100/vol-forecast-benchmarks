# Part 1: Conceptual Explanation

The `src/volcast/evaluation/` folder owns the out-of-sample benchmark logic. It answers
one core question:

**Given leakage-aware features, which model family forecasts realised variance
best under walk-forward evaluation?**

This folder keeps scoring separate from feature engineering and model
implementations. That separation makes assumptions explicit:

- training windows are expanding over time,
- retraining cadence is controlled by config,
- primary score is QLIKE,
- statistical model comparison uses Diebold-Mariano tests.

It also exports model diagnostics (for example floor-hit rates) so model quality
is not reduced to one leaderboard number.

# Part 2: Code Reference

- `src/volcast/evaluation/metrics.py`
  QLIKE, MSE, and Diebold-Mariano implementations.

- `src/volcast/evaluation/train_evaluate.py`
  Stage 4 walk-forward engine and output writers.

Where to start in code:

1. `src/volcast/evaluation/metrics.py`
2. `src/volcast/evaluation/train_evaluate.py`

# Part 3: Short Journal

- 2026-04-19: Split evaluation metrics from stage-4 runner and moved both into
  `src/volcast/evaluation/` to make benchmark methodology easier to present and audit.
