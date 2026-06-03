# Part 1: Conceptual Explanation

The `src/` directory follows the standard Python src layout. Importable product
code lives in the `volcast` package; this file is the entry guide for that tree.

# Part 2: Code Reference

- `src/volcast/`
  Main package: data, features, evaluation, models, pipeline, and shared helpers.

- `src/volcast/GUIDE_volcast.md`
  Package-level map and reading order inside `volcast`.

# Part 3: Short Journal

- 2026-05-20: Moved implementation modules from flat `src/*` into `src/volcast/`
  so imports use `volcast.*` and the repo matches the project structure guide.
