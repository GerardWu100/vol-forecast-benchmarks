"""Stage 4: Walk-forward model training and evaluation.

This stage performs expanding-window evaluation for each configured symbol and
horizon, then computes summary scores and pairwise Diebold-Mariano tests.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from volcast.evaluation.metrics import diebold_mariano_test, mse_loss, qlike_loss
from volcast.features.build_features import FEATURE_COLS
from volcast.models.garch import GARCHModel
from volcast.models.har import HARModel
from volcast.models.linear import LassoModel, RidgeModel
from volcast.models.xgboost_model import XGBoostModel
from volcast.shared.config import configured_symbols, load_config
from volcast.shared.logging import configure_logging
from volcast.shared.paths import DEFAULT_OUTPUT_DIR, DEFAULT_PROCESSED_DATA_DIR

PROCESSED_DATA_DIR = DEFAULT_PROCESSED_DATA_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

LOGGER = logging.getLogger(__name__)

MIN_OBS_FOR_SCORE = 10
MIN_OBS_FOR_DM = 30
MIN_TRAIN_OBS = 25
PREDICTION_FLOOR_TOLERANCE = 1.0000001e-10

FORECAST_COLUMNS = ["date", "symbol", "horizon", "model", "y_true", "y_pred"]

MODEL_REGISTRY = {
    "har": HARModel,
    "garch": GARCHModel,
    "ridge": RidgeModel,
    "lasso": LassoModel,
    "xgboost": XGBoostModel,
}

SCORES_COLUMNS = ["symbol", "horizon", "model", "qlike", "mse", "n_obs"]
DM_COLUMNS = ["symbol", "horizon", "model_a", "model_b", "t_stat", "p_value"]
DIAGNOSTICS_COLUMNS = [
    "symbol",
    "horizon",
    "model",
    "n_obs",
    "floor_hit_count",
    "floor_hit_rate",
    "min_prediction",
    "median_prediction",
    "max_prediction",
    "qlike_median",
    "qlike_p95",
]


def _initial_train_end_index(dates: pd.Series, initial_train_years: int) -> int:
    """Find the first evaluation index after the initial training calendar span."""
    first_date = pd.Timestamp(dates.iloc[0])
    cutoff_date = first_date + pd.DateOffset(years=initial_train_years)
    matching_indices = dates[dates >= cutoff_date].index

    if len(matching_indices) == 0:
        return len(dates)

    return int(matching_indices[0])


def _fit_and_predict(
    model_key: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, str]:
    """Fit one registered model and return forecasts plus the model display name."""
    model = MODEL_REGISTRY[model_key]()
    y_train = train_df[target_col]

    # GARCH consumes return history in addition to the shared feature matrix.
    if model_key == "garch":
        garch_columns = FEATURE_COLS + ["daily_returns"]
        model.fit(train_df[garch_columns], y_train)
        return model.predict(test_df[garch_columns]), model.name

    model.fit(train_df[FEATURE_COLS], y_train)
    return model.predict(test_df[FEATURE_COLS]), model.name


def _forecast_chunk(
    test_df: pd.DataFrame,
    symbol: str,
    horizon: int,
    model_name: str,
    target_col: str,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Build one walk-forward forecast block for a single model and test window."""
    return pd.DataFrame(
        {
            "date": test_df["date"].to_numpy(),
            "symbol": symbol,
            "horizon": horizon,
            "model": model_name,
            "y_true": test_df[target_col].to_numpy(dtype=float),
            "y_pred": np.asarray(predictions, dtype=float),
        }
    )


def walk_forward_evaluate(
    features_df: pd.DataFrame, symbol: str, horizon: int, config: dict
) -> pd.DataFrame:
    """Run expanding-window walk-forward forecast generation for one symbol/horizon."""
    working_df = features_df.sort_values("date").reset_index(drop=True)
    target_col = f"rv_{horizon}d_ahead"

    initial_train_years = config["forecast"]["initial_train_years"]
    retrain_every_days = config["forecast"]["retrain_every_days"]
    enabled_models = config["models"]["enabled"]

    # First out-of-sample date must leave enough history for model fitting.
    train_end_idx = _initial_train_end_index(working_df["date"], initial_train_years)
    train_end_idx = max(train_end_idx, MIN_TRAIN_OBS)
    if train_end_idx >= len(working_df):
        return pd.DataFrame(columns=FORECAST_COLUMNS)

    retrain_points = list(range(train_end_idx, len(working_df), retrain_every_days))
    forecast_chunks: list[pd.DataFrame] = []

    for point_idx, retrain_idx in enumerate(retrain_points):
        train_df = working_df.iloc[:retrain_idx]

        # Each retrain point forecasts until the next retrain boundary.
        if point_idx + 1 < len(retrain_points):
            test_end = retrain_points[point_idx + 1]
        else:
            test_end = len(working_df)

        test_df = working_df.iloc[retrain_idx:test_end]
        if test_df.empty:
            continue

        for model_key in enabled_models:
            try:
                predictions, model_name = _fit_and_predict(
                    model_key, train_df, test_df, target_col
                )
            except Exception as error:
                LOGGER.warning(
                    "%s failed (%s, h=%s, retrain=%s): %s",
                    model_key,
                    symbol,
                    horizon,
                    point_idx,
                    error,
                )
                predictions = np.full(len(test_df), np.nan)
                model_name = MODEL_REGISTRY[model_key]().name

            forecast_chunks.append(
                _forecast_chunk(test_df, symbol, horizon, model_name, target_col, predictions)
            )

    if not forecast_chunks:
        return pd.DataFrame(columns=FORECAST_COLUMNS)

    return pd.concat(forecast_chunks, ignore_index=True)


def _qlike_summary(valid_positive: pd.DataFrame) -> tuple[float, float]:
    """Return median and 95th percentile QLIKE for strictly positive forecast rows."""
    if valid_positive.empty:
        return (float("nan"), float("nan"))

    qlike_values = (
        np.log(valid_positive["y_pred"]) + valid_positive["y_true"] / valid_positive["y_pred"]
    )
    return (float(qlike_values.median()), float(qlike_values.quantile(0.95)))


def compute_model_diagnostics(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise prediction-path health for each symbol, horizon, and model."""
    diagnostics_rows: list[dict[str, object]] = []

    for (symbol, horizon, model_name), group in forecasts_df.groupby(
        ["symbol", "horizon", "model"]
    ):
        valid = group.dropna(subset=["y_true", "y_pred"])
        if valid.empty:
            continue

        # Floor hits flag forecasts clipped to the minimum positive variance bound.
        floor_hits = valid["y_pred"] <= PREDICTION_FLOOR_TOLERANCE
        valid_positive = valid[(valid["y_true"] > 0) & (valid["y_pred"] > 0)]
        qlike_median, qlike_p95 = _qlike_summary(valid_positive)

        diagnostics_rows.append(
            {
                "symbol": symbol,
                "horizon": horizon,
                "model": model_name,
                "n_obs": int(len(valid)),
                "floor_hit_count": int(floor_hits.sum()),
                "floor_hit_rate": float(floor_hits.mean()),
                "min_prediction": float(valid["y_pred"].min()),
                "median_prediction": float(valid["y_pred"].median()),
                "max_prediction": float(valid["y_pred"].max()),
                "qlike_median": qlike_median,
                "qlike_p95": qlike_p95,
            }
        )

    if not diagnostics_rows:
        return pd.DataFrame(columns=DIAGNOSTICS_COLUMNS)

    return pd.DataFrame(diagnostics_rows, columns=DIAGNOSTICS_COLUMNS)


def compute_scores(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate QLIKE and MSE per symbol/horizon/model."""
    score_rows: list[dict[str, object]] = []

    for (symbol, horizon, model_name), group in forecasts_df.groupby(
        ["symbol", "horizon", "model"]
    ):
        valid = group.dropna(subset=["y_true", "y_pred"])
        valid = valid[(valid["y_true"] > 0) & (valid["y_pred"] > 0)]
        if len(valid) < MIN_OBS_FOR_SCORE:
            continue

        y_true = valid["y_true"].to_numpy(dtype=float)
        y_pred = valid["y_pred"].to_numpy(dtype=float)

        score_rows.append(
            {
                "symbol": symbol,
                "horizon": horizon,
                "model": model_name,
                "qlike": qlike_loss(y_true, y_pred),
                "mse": mse_loss(y_true, y_pred),
                "n_obs": len(valid),
            }
        )

    if not score_rows:
        return pd.DataFrame(columns=SCORES_COLUMNS)

    return pd.DataFrame(score_rows, columns=SCORES_COLUMNS)


def compute_dm_tests(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise DM test statistics across models."""
    dm_rows: list[dict[str, object]] = []

    for (symbol, horizon), group in forecasts_df.groupby(["symbol", "horizon"]):
        models = sorted(group["model"].unique())

        # Align model forecasts on common evaluation dates before loss comparison.
        pivot = group.pivot_table(index="date", columns="model", values=["y_true", "y_pred"])
        pivot = pivot.dropna()
        if len(pivot) < MIN_OBS_FOR_DM:
            continue

        y_true = pivot["y_true"].iloc[:, 0].to_numpy(dtype=float)

        for idx_a in range(len(models)):
            for idx_b in range(idx_a + 1, len(models)):
                model_a = models[idx_a]
                model_b = models[idx_b]

                pred_a = pivot["y_pred"][model_a].to_numpy(dtype=float)
                pred_b = pivot["y_pred"][model_b].to_numpy(dtype=float)

                loss_a = np.log(pred_a) + y_true / pred_a
                loss_b = np.log(pred_b) + y_true / pred_b

                t_stat, p_value = diebold_mariano_test(loss_a, loss_b)
                dm_rows.append(
                    {
                        "symbol": symbol,
                        "horizon": horizon,
                        "model_a": model_a,
                        "model_b": model_b,
                        "t_stat": t_stat,
                        "p_value": p_value,
                    }
                )

    if not dm_rows:
        return pd.DataFrame(columns=DM_COLUMNS)

    return pd.DataFrame(dm_rows, columns=DM_COLUMNS)


def main() -> None:
    """Run stage 4 for all configured symbols and horizons."""
    configure_logging()
    config = load_config()

    symbols = configured_symbols(config)
    horizons = config["forecast"]["horizons"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    forecast_frames: list[pd.DataFrame] = []
    for symbol in symbols:
        path = PROCESSED_DATA_DIR / f"{symbol}_features.parquet"
        symbol_features = pd.read_parquet(path)
        symbol_features["date"] = pd.to_datetime(symbol_features["date"])

        for horizon in horizons:
            LOGGER.info("Walk-forward: %s horizon=%s", symbol, horizon)
            horizon_forecasts = walk_forward_evaluate(symbol_features, symbol, horizon, config)
            forecast_frames.append(horizon_forecasts)

    if forecast_frames:
        forecasts_df = pd.concat(forecast_frames, ignore_index=True)
    else:
        forecasts_df = pd.DataFrame(columns=FORECAST_COLUMNS)

    scores_df = compute_scores(forecasts_df)
    dm_df = compute_dm_tests(forecasts_df)
    diagnostics_df = compute_model_diagnostics(forecasts_df)

    forecasts_path = OUTPUT_DIR / "forecasts.parquet"
    scores_path = OUTPUT_DIR / "scores.parquet"
    dm_path = OUTPUT_DIR / "dm_tests.parquet"
    diagnostics_path = OUTPUT_DIR / "model_diagnostics.parquet"

    forecasts_df.to_parquet(forecasts_path, index=False)
    scores_df.to_parquet(scores_path, index=False)
    dm_df.to_parquet(dm_path, index=False)
    diagnostics_df.to_parquet(diagnostics_path, index=False)

    LOGGER.info("Saved forecasts: %s", forecasts_path)
    LOGGER.info("Saved scores: %s", scores_path)
    LOGGER.info("Saved DM tests: %s", dm_path)
    LOGGER.info("Saved diagnostics: %s", diagnostics_path)
    if not scores_df.empty:
        LOGGER.info(
            "\n%s", scores_df.sort_values(["symbol", "horizon", "qlike"]).to_string(index=False)
        )


if __name__ == "__main__":
    main()
