"""Generate frozen blog evidence and charts for the VolCast article.

The script leaves the project's configured benchmark unchanged. It records the
default-run coverage audit, then runs a one-year initial-window sensitivity
analysis through the same walk-forward evaluation functions used by stage 4.

Outputs
-------
blog/data/default_run_audit.csv
    Configuration and row-count evidence explaining the empty default output.
blog/data/sensitivity_scores.csv
    QLIKE and MSE scores from the one-year sensitivity analysis.
blog/data/sensitivity_diagnostics.csv
    Forecast-path diagnostics from the sensitivity analysis.
blog/data/sensitivity_dm_tests.csv
    Pairwise Diebold-Mariano tests from the sensitivity analysis.
blog/images/01_realised_volatility.png
    Annualised close-to-close realised volatility and its trailing mean.
blog/images/02_qlike_vs_har.png
    Model QLIKE differences relative to HAR-RV by horizon.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volcast.evaluation.train_evaluate import (
    compute_dm_tests,
    compute_model_diagnostics,
    compute_scores,
    walk_forward_evaluate,
)
from volcast.shared.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLOG_DIR = Path(__file__).resolve().parent
DATA_DIR = BLOG_DIR / "data"
IMAGE_DIR = BLOG_DIR / "images"
TRADING_DAYS_PER_YEAR = 252.0
SENSITIVITY_INITIAL_TRAIN_YEARS = 1
FIGURE_DPI = 180

MODEL_COLORS = {
    "GARCH(1,1)": "#dd9b42",
    "HAR-RV": "#4f8fc9",
    "Lasso": "#70a36b",
    "Ridge": "#ad78b4",
    "XGBoost": "#c65d63",
}


def _load_project_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the generated realised-variance and model-feature tables.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Daily realised-variance rows followed by model-ready feature rows.
    """
    rv_df = pd.read_parquet(PROJECT_ROOT / "data/processed/SPY_rv.parquet")
    features_df = pd.read_parquet(PROJECT_ROOT / "data/processed/SPY_features.parquet")
    rv_df["date"] = pd.to_datetime(rv_df["date"])
    features_df["date"] = pd.to_datetime(features_df["date"])
    return rv_df, features_df


def _write_default_audit(
    rv_df: pd.DataFrame, features_df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Freeze row-count and date-span evidence for the configured run.

    Parameters
    ----------
    rv_df : pd.DataFrame
        Stage-2 daily realised-variance frame.
    features_df : pd.DataFrame
        Stage-3 model-ready feature frame.
    config : dict
        Unmodified project configuration.

    Returns
    -------
    pd.DataFrame
        One-row audit record written to ``blog/data``.
    """
    forecasts_df = pd.read_parquet(PROJECT_ROOT / "outputs/forecasts.parquet")
    scores_df = pd.read_parquet(PROJECT_ROOT / "outputs/scores.parquet")
    audit_df = pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "rv_rows": len(rv_df),
                "feature_rows": len(features_df),
                "feature_start": features_df["date"].min().date().isoformat(),
                "feature_end": features_df["date"].max().date().isoformat(),
                "configured_initial_train_years": config["forecast"][
                    "initial_train_years"
                ],
                "default_forecast_rows": len(forecasts_df),
                "default_score_rows": len(scores_df),
            }
        ]
    )
    audit_df.to_csv(DATA_DIR / "default_run_audit.csv", index=False)
    return audit_df


def _run_sensitivity(features_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, ...]:
    """Run a one-year burn-in sensitivity through the production evaluator.

    Parameters
    ----------
    features_df : pd.DataFrame
        Stage-3 feature matrix with one row per forecast origin.
    config : dict
        Unmodified project configuration. The function changes a deep copy only.

    Returns
    -------
    tuple[pd.DataFrame, ...]
        Forecasts, aggregate scores, diagnostics, and pairwise DM tests.
    """
    sensitivity_config = deepcopy(config)
    sensitivity_config["forecast"]["initial_train_years"] = (
        SENSITIVITY_INITIAL_TRAIN_YEARS
    )

    forecast_frames = [
        walk_forward_evaluate(features_df, "SPY", horizon, sensitivity_config)
        for horizon in sensitivity_config["forecast"]["horizons"]
    ]
    forecasts_df = pd.concat(forecast_frames, ignore_index=True)
    scores_df = compute_scores(forecasts_df)
    diagnostics_df = compute_model_diagnostics(forecasts_df)
    dm_df = compute_dm_tests(forecasts_df)

    scores_df.to_csv(DATA_DIR / "sensitivity_scores.csv", index=False)
    diagnostics_df.to_csv(DATA_DIR / "sensitivity_diagnostics.csv", index=False)
    dm_df.to_csv(DATA_DIR / "sensitivity_dm_tests.csv", index=False)
    return forecasts_df, scores_df, diagnostics_df, dm_df


def _plot_realised_volatility(rv_df: pd.DataFrame) -> None:
    """Plot annualised daily realised volatility and its 22-day trailing mean.

    Parameters
    ----------
    rv_df : pd.DataFrame
        Stage-2 table containing ``date`` and ``rv_cc``.

    Returns
    -------
    None
        Writes a PNG chart under ``blog/images``.
    """
    daily_vol = np.sqrt(TRADING_DAYS_PER_YEAR * rv_df["rv_cc"]) * 100.0
    monthly_mean = daily_vol.rolling(window=22, min_periods=22).mean()

    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    ax.plot(rv_df["date"], daily_vol, color="#92b9dc", linewidth=0.9, alpha=0.65)
    ax.plot(
        rv_df["date"],
        monthly_mean,
        color="#183a5a",
        linewidth=2.3,
        label="22-day trailing mean",
    )
    ax.fill_between(rv_df["date"], monthly_mean, color="#4f8fc9", alpha=0.12)
    ax.set_title("SPY realised volatility clusters through time", loc="left", weight="bold")
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_xlabel("Trading date")
    ax.grid(axis="y", color="#d7dde3", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(IMAGE_DIR / "01_realised_volatility.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_qlike_difference(scores_df: pd.DataFrame) -> None:
    """Plot QLIKE loss differences from HAR-RV for each forecast horizon.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Sensitivity score table with model, horizon, and QLIKE columns.

    Returns
    -------
    None
        Writes a PNG chart under ``blog/images``.
    """
    baseline = (
        scores_df[scores_df["model"] == "HAR-RV"]
        .set_index("horizon")["qlike"]
        .rename("har_qlike")
    )
    plot_df = scores_df.join(baseline, on="horizon")
    plot_df["qlike_difference"] = plot_df["qlike"] - plot_df["har_qlike"]

    horizons = sorted(plot_df["horizon"].unique())
    models = [name for name in MODEL_COLORS if name != "HAR-RV"]
    x_positions = np.arange(len(models))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    for horizon_idx, horizon in enumerate(horizons):
        horizon_df = plot_df[plot_df["horizon"] == horizon].set_index("model")
        values = [horizon_df.loc[model, "qlike_difference"] for model in models]
        offset = (horizon_idx - (len(horizons) - 1) / 2) * bar_width
        ax.bar(
            x_positions + offset,
            values,
            width=bar_width,
            label=f"{horizon}-day horizon",
            alpha=0.88,
        )

    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_xticks(x_positions, models)
    ax.set_ylabel("QLIKE minus HAR-RV QLIKE (lower is better)")
    ax.set_title(
        "One-year burn-in sensitivity: loss relative to HAR-RV",
        loc="left",
        weight="bold",
    )
    ax.grid(axis="y", color="#d7dde3", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(IMAGE_DIR / "02_qlike_vs_har.png", dpi=FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    """Generate all frozen evidence and charts for the bilingual article."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    rv_df, features_df = _load_project_frames()
    _write_default_audit(rv_df, features_df, config)
    _, scores_df, _, _ = _run_sensitivity(features_df, config)
    _plot_realised_volatility(rv_df)
    _plot_qlike_difference(scores_df)


if __name__ == "__main__":
    main()
