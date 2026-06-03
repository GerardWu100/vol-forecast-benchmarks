"""Project path utilities for the offline volatility benchmark.

The pipeline relies on consistent project-relative paths so commands work from
any current working directory. This module is the single source of truth for
those paths.
"""

from __future__ import annotations

from pathlib import Path

# Project root resolved from this file path:
# src/volcast/shared/paths.py -> shared -> volcast -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Default raw input cache (stage 1 output, stages 2-3 input).
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Default processed checkpoints (stage 2 and stage 3 outputs).
DEFAULT_PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Default final benchmark outputs (stage 4 outputs).
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
