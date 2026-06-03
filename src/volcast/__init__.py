"""Source package for the offline-first volatility forecasting benchmark.

Subpackages are organized by responsibility:
- ``volcast.data``: raw-cache contract and ClickHouse refresh logic,
- ``volcast.features``: realised-variance and feature construction,
- ``volcast.evaluation``: walk-forward scoring and diagnostics,
- ``volcast.models``: model-family wrappers,
- ``volcast.pipeline``: stage orchestration,
- ``volcast.shared``: cross-cutting config/path helpers.
"""
