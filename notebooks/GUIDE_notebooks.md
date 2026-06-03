# Part 1: Conceptual Explanation

The `notebooks/` folder holds teaching and interpretation artifacts, not core
pipeline logic. Heavy computation stays in `src/` modules so notebook execution
is deterministic and auditable.

The main notebook is offline-first. It starts from parquet files in `data/raw`,
walks through realised-variance and feature construction using project modules,
runs walk-forward evaluation, and interprets outputs.

Notebook structure follows a strict alternating rhythm:

1. one substantial Markdown explanation cell,
2. one code cell that executes the explained step.

This keeps the notebook readable for interview walkthroughs.

# Part 2: Code Reference

- `notebooks/offline_pipeline_demo.ipynb`
  End-to-end teaching notebook for the offline benchmark flow.

Where to start in code:

1. `notebooks/offline_pipeline_demo.ipynb`
2. `src/volcast/features/compute_rv.py`
3. `src/volcast/features/build_features.py`
4. `src/volcast/evaluation/train_evaluate.py`

# Part 3: Short Journal

- 2026-04-19: Replaced the output-review notebook with a full offline pipeline
  teaching notebook that starts from `data/raw`.
