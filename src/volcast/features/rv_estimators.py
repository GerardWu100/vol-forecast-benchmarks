"""Realised variance estimators and HAR lag builders.

This module converts minute-level OHLCV (Open, High, Low, Close, Volume)
bars into daily realised variance features for volatility forecasting.

Implemented estimators:
- close_to_close_rv: sum of squared intraday log returns
- parkinson_rv: range-based estimator using daily high and low
- garman_klass_rv: OHLC-based estimator using daily open, high, low, close

The module also adds HAR (Heterogeneous Autoregressive) lag features that
summarise volatility at daily, weekly, and monthly horizons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def close_to_close_rv(closes: np.ndarray) -> float:
    """Compute close-to-close realised variance for one day.

    Parameters
    ----------
    closes : np.ndarray
        Minute-level close prices for a single day.

    Returns
    -------
    float
        Sum of squared intraday log returns.
    """
    if closes.size < 2:
        return 0.0

    # Intraday log returns: r_i = log(C_i / C_{i-1}).
    log_returns = np.diff(np.log(closes))

    # Daily realised variance proxy: RV = sum_i r_i^2.
    realised_variance = np.sum(log_returns**2)
    return float(realised_variance)


def parkinson_rv(high: float, low: float) -> float:
    """Compute Parkinson realised variance for one day.

    Parameters
    ----------
    high : float
        Daily high price.
    low : float
        Daily low price.

    Returns
    -------
    float
        Range-based realised variance estimate.
    """
    if high == low:
        return 0.0

    # Parkinson estimator: RV = [1 / (4 * ln(2))] * [ln(H/L)]^2.
    scale = 1.0 / (4.0 * np.log(2.0))
    log_range = np.log(high / low)
    realised_variance = scale * (log_range**2)
    return float(realised_variance)


def garman_klass_rv(open_price: float, high: float, low: float, close: float) -> float:
    """Compute Garman-Klass realised variance for one day.

    Parameters
    ----------
    open_price : float
        Daily open price.
    high : float
        Daily high price.
    low : float
        Daily low price.
    close : float
        Daily close price.

    Returns
    -------
    float
        OHLC-based realised variance estimate.
    """
    if high == low and open_price == close:
        return 0.0

    # First term captures intraday range variation.
    range_term = 0.5 * (np.log(high / low) ** 2)

    # Second term adjusts for open-to-close drift.
    drift_scale = 2.0 * np.log(2.0) - 1.0
    drift_term = drift_scale * (np.log(close / open_price) ** 2)

    realised_variance = range_term - drift_term
    return float(realised_variance)


def compute_daily_rv(minutes_df: pd.DataFrame, min_minutes: int = 300) -> pd.DataFrame:
    """Aggregate minute bars into daily realised variance estimators.

    Parameters
    ----------
    minutes_df : pd.DataFrame
        Minute bars with required columns:
        - ts: timestamp
        - open, high, low, close, volume
    min_minutes : int, default=300
        Minimum number of minute rows required to keep a day.

    Returns
    -------
    pd.DataFrame
        Daily realised variance DataFrame with columns:
        date, rv_cc, rv_parkinson, rv_gk, n_minutes,
        daily_open, daily_high, daily_low, daily_close.
    """
    working_df = minutes_df.copy()
    working_df["ts"] = pd.to_datetime(working_df["ts"])
    working_df["date"] = working_df["ts"].dt.date

    daily_rows: list[dict[str, float | int | pd.Timestamp]] = []

    # Group by trading date to compute one row per day.
    for day, day_bars in working_df.groupby("date"):
        minute_count = len(day_bars)
        if minute_count < min_minutes:
            continue

        sorted_day_bars = day_bars.sort_values("ts")

        day_open = float(sorted_day_bars["open"].iloc[0])
        day_high = float(sorted_day_bars["high"].max())
        day_low = float(sorted_day_bars["low"].min())
        day_close = float(sorted_day_bars["close"].iloc[-1])

        close_prices = sorted_day_bars["close"].to_numpy(dtype=float)
        rv_close_to_close = close_to_close_rv(close_prices)
        rv_parkinson_est = parkinson_rv(day_high, day_low)
        rv_garman_klass = garman_klass_rv(day_open, day_high, day_low, day_close)

        daily_rows.append(
            {
                "date": pd.Timestamp(day),
                "rv_cc": rv_close_to_close,
                "rv_parkinson": rv_parkinson_est,
                "rv_gk": rv_garman_klass,
                "n_minutes": minute_count,
                "daily_open": day_open,
                "daily_high": day_high,
                "daily_low": day_low,
                "daily_close": day_close,
            }
        )

    daily_df = pd.DataFrame(daily_rows)
    if daily_df.empty:
        return daily_df

    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    return daily_df


def add_har_lags(rv_df: pd.DataFrame, column: str, lags: list[int]) -> pd.DataFrame:
    """Add HAR lag features for a realised variance column.

    Parameters
    ----------
    rv_df : pd.DataFrame
        Input DataFrame with a ``date`` column and the target variance column.
    column : str
        Name of the realised variance column to lag.
    lags : list[int]
        Lag windows in trading days.

    Returns
    -------
    pd.DataFrame
        Copy of input with lag columns added.
    """
    suffix_by_lag = {1: "d", 5: "w", 22: "m"}

    lagged_df = rv_df.copy()
    lagged_df = lagged_df.sort_values("date").reset_index(drop=True)

    for lag_window in lags:
        lag_suffix = suffix_by_lag.get(lag_window, str(lag_window))
        lag_column_name = f"{column}_{lag_suffix}"

        if lag_window == 1:
            # Daily HAR component: previous day value.
            lagged_df[lag_column_name] = lagged_df[column].shift(1)
            continue

        # Weekly/monthly HAR components: mean of previous k days, shifted by 1.
        shifted_series = lagged_df[column].shift(1)
        rolling_mean = shifted_series.rolling(window=lag_window, min_periods=lag_window).mean()
        lagged_df[lag_column_name] = rolling_mean

    return lagged_df
