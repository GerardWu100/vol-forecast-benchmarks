"""Freeze default VolCast evidence and generate the blog's technical charts.

The script reads stage-2, stage-3, and stage-4 artifacts produced by the
unchanged portable configuration. It does not refit models or alter project
settings. Run the offline pipeline before this script.

Outputs
-------
blog/data/run_audit.csv
    Date coverage, configuration, and output row counts for the default run.
blog/data/default_scores.csv
    Aggregate QLIKE and MSE scores copied from stage 4.
blog/data/default_diagnostics.csv
    Forecast-health diagnostics copied from stage 4.
blog/data/default_dm_tests.csv
    Pairwise Diebold-Mariano results copied from stage 4.
blog/images/01_realised_volatility.png
    Annualised daily realised volatility and its 22-day trailing mean.
blog/images/02_qlike_vs_har.png
    Model QLIKE differences relative to HAR-RV by horizon.
blog/images/03_forecast_paths.png
    One-day realised volatility beside HAR-RV and Lasso forecasts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volcast.shared.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLOG_DIR = Path(__file__).resolve().parent
DATA_DIR = BLOG_DIR / "data"
IMAGE_DIR = BLOG_DIR / "images"
TRADING_DAYS_PER_YEAR = 252.0
FIGURE_DPI = 180


def _load_artifacts() -> dict[str, pd.DataFrame]:
    """Load generated project artifacts required by the article.

    Returns
    -------
    dict[str, pd.DataFrame]
        Frames keyed by ``rv``, ``features``, ``forecasts``, ``scores``,
        ``diagnostics``, and ``dm_tests``.
    """
    paths = {
        "rv": PROJECT_ROOT / "data/processed/SPY_rv.parquet",
        "features": PROJECT_ROOT / "data/processed/SPY_features.parquet",
        "forecasts": PROJECT_ROOT / "outputs/forecasts.parquet",
        "scores": PROJECT_ROOT / "outputs/scores.parquet",
        "diagnostics": PROJECT_ROOT / "outputs/model_diagnostics.parquet",
        "dm_tests": PROJECT_ROOT / "outputs/dm_tests.parquet",
    }
    missing_paths = [str(path) for path in paths.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Run `uv run python -m volcast.pipeline.run_pipeline` before chart "
            f"generation. Missing: {missing_paths}"
        )

    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    for frame_name in ("rv", "features", "forecasts"):
        frames[frame_name]["date"] = pd.to_datetime(frames[frame_name]["date"])
    if frames["forecasts"].empty or frames["scores"].empty:
        raise ValueError("Default stage-4 artifacts are empty; the blog requires scored forecasts")
    return frames


def _freeze_evidence(frames: dict[str, pd.DataFrame], config: dict[str, Any]) -> None:
    """Write compact, reviewable CSV evidence for the article.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Generated project artifacts returned by :func:`_load_artifacts`.
    config : dict[str, Any]
        Unmodified portable project configuration.

    Returns
    -------
    None
        Writes four CSV files under ``blog/data``.
    """
    forecasts_df = frames["forecasts"]
    features_df = frames["features"]
    audit_df = pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "rv_rows": len(frames["rv"]),
                "feature_rows": len(features_df),
                "feature_start": features_df["date"].min().date().isoformat(),
                "feature_end": features_df["date"].max().date().isoformat(),
                "initial_train_years": config["forecast"]["initial_train_years"],
                "retrain_every_days": config["forecast"]["retrain_every_days"],
                "first_forecast_date": forecasts_df["date"].min().date().isoformat(),
                "last_forecast_date": forecasts_df["date"].max().date().isoformat(),
                "forecast_rows": len(forecasts_df),
                "score_rows": len(frames["scores"]),
            }
        ]
    )
    audit_df.to_csv(DATA_DIR / "run_audit.csv", index=False)
    frames["scores"].to_csv(DATA_DIR / "default_scores.csv", index=False)
    frames["diagnostics"].to_csv(DATA_DIR / "default_diagnostics.csv", index=False)
    frames["dm_tests"].to_csv(DATA_DIR / "default_dm_tests.csv", index=False)


def _plot_realised_volatility(rv_df: pd.DataFrame) -> None:
    """Plot annualised daily realised volatility and its trailing mean.

    Parameters
    ----------
    rv_df : pd.DataFrame
        Stage-2 table containing ``date`` and ``rv_cc``.

    Returns
    -------
    None
        Writes ``01_realised_volatility.png``.
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
    """Plot default QLIKE differences from HAR-RV by horizon.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Default score table containing model, horizon, and QLIKE columns.

    Returns
    -------
    None
        Writes ``02_qlike_vs_har.png``.
    """
    baseline = (
        scores_df[scores_df["model"] == "HAR-RV"].set_index("horizon")["qlike"].rename("har_qlike")
    )
    plot_df = scores_df.join(baseline, on="horizon")
    plot_df["qlike_difference"] = plot_df["qlike"] - plot_df["har_qlike"]

    models = ["GARCH(1,1)", "Lasso", "Ridge", "XGBoost"]
    horizons = sorted(plot_df["horizon"].unique())
    x_positions = np.arange(len(models))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    for horizon_index, horizon in enumerate(horizons):
        horizon_df = plot_df[plot_df["horizon"] == horizon].set_index("model")
        values = [horizon_df.loc[model, "qlike_difference"] for model in models]
        offset = (horizon_index - (len(horizons) - 1) / 2) * bar_width
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
    ax.set_title("Portable default: loss relative to HAR-RV", loc="left", weight="bold")
    ax.grid(axis="y", color="#d7dde3", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(IMAGE_DIR / "02_qlike_vs_har.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_forecast_paths(forecasts_df: pd.DataFrame) -> None:
    """Plot one-day realised volatility beside HAR-RV and Lasso forecasts.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Long-form default forecasts with actual and predicted variance.

    Returns
    -------
    None
        Writes ``03_forecast_paths.png``.
    """
    one_day = forecasts_df[
        (forecasts_df["horizon"] == 1) & forecasts_df["model"].isin(["HAR-RV", "Lasso"])
    ].copy()
    actual = one_day.drop_duplicates("date").sort_values("date")
    actual_vol = np.sqrt(TRADING_DAYS_PER_YEAR * actual["y_true"]) * 100.0

    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    ax.plot(
        actual["date"],
        actual_vol,
        color="#b9c1c9",
        linewidth=1.0,
        label="Realised",
    )
    for model, color in (("HAR-RV", "#183a5a"), ("Lasso", "#d98936")):
        model_df = one_day[one_day["model"] == model].sort_values("date")
        model_vol = np.sqrt(TRADING_DAYS_PER_YEAR * model_df["y_pred"]) * 100.0
        ax.plot(model_df["date"], model_vol, color=color, linewidth=1.7, label=model)

    ax.set_title(
        "One-day forecasts smooth the realised volatility spikes", loc="left", weight="bold"
    )
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_xlabel("Forecast date")
    ax.grid(axis="y", color="#d7dde3", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(IMAGE_DIR / "03_forecast_paths.png", dpi=FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    """Freeze default evidence and regenerate all article charts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    frames = _load_artifacts()
    config = load_config()
    _freeze_evidence(frames, config)
    _plot_realised_volatility(frames["rv"])
    _plot_qlike_difference(frames["scores"])
    _plot_forecast_paths(frames["forecasts"])


if __name__ == "__main__":
    main()
