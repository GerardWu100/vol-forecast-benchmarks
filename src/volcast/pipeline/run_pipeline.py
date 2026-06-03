"""Pipeline orchestrator for the offline-first volatility benchmark workflow."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable

from volcast.shared.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def run_stage_1(force: bool) -> None:
    """Run stage 1 raw-cache validation/fetch."""
    from volcast.data.fetch_raw_cache import main as fetch_main

    fetch_main(force=force)


def run_stage_2() -> None:
    """Run stage 2 realised variance computation."""
    from volcast.features.compute_rv import main as compute_rv_main

    compute_rv_main()


def run_stage_3() -> None:
    """Run stage 3 feature matrix construction."""
    from volcast.features.build_features import main as build_features_main

    build_features_main()


def run_stage_4() -> None:
    """Run stage 4 walk-forward training and evaluation."""
    from volcast.evaluation.train_evaluate import main as train_evaluate_main

    train_evaluate_main()


def main() -> None:
    """Execute configured four-stage pipeline sequentially."""
    configure_logging()

    parser = argparse.ArgumentParser(description="Run offline volatility benchmark pipeline")
    parser.add_argument("--force", action="store_true", help="Force raw cache re-fetch in stage 1")
    parser.add_argument(
        "--from",
        dest="from_stage",
        type=int,
        default=2,
        help="Start from stage number (1-4); default 2 for offline workflow",
    )
    args = parser.parse_args()

    stages: list[tuple[str, Callable[[], None]]] = [
        ("Stage 1: Fetch/validate raw cache", lambda: run_stage_1(force=args.force)),
        ("Stage 2: Compute RV", run_stage_2),
        ("Stage 3: Build features", run_stage_3),
        ("Stage 4: Train and evaluate", run_stage_4),
    ]

    total_start = time.time()
    for stage_index, (stage_name, stage_runner) in enumerate(stages, start=1):
        if stage_index < args.from_stage:
            LOGGER.info("SKIP %s", stage_name)
            continue

        LOGGER.info("%s", "=" * 70)
        LOGGER.info("RUNNING %s", stage_name)
        LOGGER.info("%s", "=" * 70)

        stage_start = time.time()
        stage_runner()
        elapsed = time.time() - stage_start
        LOGGER.info("Completed %s in %.1f seconds", stage_name, elapsed)

    total_elapsed = time.time() - total_start
    LOGGER.info("Pipeline completed in %.1f seconds", total_elapsed)
    LOGGER.info(
        "Output paths: outputs/forecasts.parquet, outputs/scores.parquet, outputs/dm_tests.parquet"
    )


if __name__ == "__main__":
    main()
