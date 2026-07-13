# Part 1: Conceptual Explanation

The `src/volcast/pipeline/` folder orchestrates the four benchmark stages without adding
new quantitative logic. It is intentionally thin.

Default behavior is offline-first: the pipeline starts from stage 2 unless you
explicitly ask for stage 1. This mirrors the project contract that versioned raw
parquet in `data/raw` should be enough for a reproducible run.

Stage order:

1. validate/refresh raw cache (optional, database-backed),
2. compute realised variance,
3. build leakage-aware features,
4. run walk-forward evaluation.

The default stage-4 schedule uses one calendar year for initial training and
refits every 21 trading days. If the configured initial span exceeds the
available features, the evaluator raises before the orchestrator reports a
successful pipeline run.

# Part 2: Code Reference

- `src/volcast/pipeline/run_pipeline.py`
  CLI entrypoint that runs stages sequentially with `--from` and optional
  `--force` for stage-1 refresh.

Where to start in code:

1. `src/volcast/pipeline/run_pipeline.py`

# Part 3: Short Journal

- 2026-04-19: Replaced the five-stage orchestrator with a four-stage offline
  pipeline entrypoint under `src/volcast/pipeline/`.
- 2026-07-13: The pipeline now surfaces infeasible training-history settings as
  errors rather than completing with empty stage-4 artifacts.
