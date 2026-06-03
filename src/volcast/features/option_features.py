"""Option-implied volatility feature extraction utilities.

This module computes daily features from option chains for volatility
forecasting models:
- atm_iv: near-ATM call implied volatility
- iv_skew: put-call implied volatility skew at fixed absolute delta
- iv_term_slope: front-vs-second-expiry ATM implied volatility spread
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _mid_iv(option_row: pd.Series) -> float:
    """Compute midpoint implied volatility from bid and ask IV quotes."""
    return float((option_row["bid_iv"] + option_row["ask_iv"]) / 2.0)


def _filter_by_dte(chain: pd.DataFrame, min_dte: int, max_dte: int) -> pd.DataFrame:
    """Filter an option chain by days-to-expiry bounds."""
    trade_date = chain["trade_date"].iloc[0]
    days_to_expiry = (chain["expiry_date"] - trade_date).dt.days
    valid_mask = (days_to_expiry >= min_dte) & (days_to_expiry <= max_dte)
    return chain[valid_mask].copy()


def _nearest_expiry(chain: pd.DataFrame) -> pd.Timestamp:
    """Return the nearest expiry date present in the chain."""
    return pd.Timestamp(chain["expiry_date"].min())


def _second_nearest_expiry(chain: pd.DataFrame) -> pd.Timestamp | None:
    """Return the second nearest expiry date if present, otherwise None."""
    unique_expiries = sorted(pd.to_datetime(chain["expiry_date"]).unique())
    if len(unique_expiries) < 2:
        return None
    return pd.Timestamp(unique_expiries[1])


def _atm_call_iv(chain: pd.DataFrame, expiry: pd.Timestamp, target_delta: float) -> float:
    """Return midpoint IV for the call closest to ``target_delta`` at one expiry."""
    calls = chain[(chain["expiry_date"] == expiry) & (chain["option_type"] == "c")]
    if calls.empty:
        return float("nan")

    best_index = (calls["delta"] - target_delta).abs().idxmin()
    return _mid_iv(calls.loc[best_index])


def compute_atm_iv(
    chain: pd.DataFrame,
    target_delta: float = 0.50,
    min_dte: int = 7,
    max_dte: int = 45,
) -> float:
    """Compute ATM implied volatility from nearest-expiry calls.

    ATM (at-the-money) is proxied by call delta closest to ``target_delta``.
    """
    filtered_chain = _filter_by_dte(chain, min_dte=min_dte, max_dte=max_dte)
    if filtered_chain.empty:
        return float("nan")

    front_expiry = _nearest_expiry(filtered_chain)
    return _atm_call_iv(filtered_chain, front_expiry, target_delta)


def compute_iv_skew(
    chain: pd.DataFrame,
    skew_delta: float = 0.25,
    min_dte: int = 7,
    max_dte: int = 45,
) -> float:
    """Compute implied volatility skew: put_iv(-delta) - call_iv(+delta)."""
    filtered_chain = _filter_by_dte(chain, min_dte=min_dte, max_dte=max_dte)
    if filtered_chain.empty:
        return float("nan")

    front_expiry = _nearest_expiry(filtered_chain)
    front_chain = filtered_chain[filtered_chain["expiry_date"] == front_expiry]

    puts = front_chain[front_chain["option_type"] == "p"]
    calls = front_chain[front_chain["option_type"] == "c"]
    if puts.empty or calls.empty:
        return float("nan")

    # Match puts and calls at symmetric absolute delta targets.
    put_target = -skew_delta
    put_idx = (puts["delta"] - put_target).abs().idxmin()
    call_idx = (calls["delta"] - skew_delta).abs().idxmin()

    put_iv = _mid_iv(puts.loc[put_idx])
    call_iv = _mid_iv(calls.loc[call_idx])
    return float(put_iv - call_iv)


def compute_iv_term_slope(
    chain: pd.DataFrame,
    target_delta: float = 0.50,
    min_dte: int = 7,
    max_dte: int = 60,
) -> float:
    """Compute IV term slope as ATM IV(second expiry) - ATM IV(front expiry)."""
    filtered_chain = _filter_by_dte(chain, min_dte=min_dte, max_dte=max_dte)
    if filtered_chain.empty:
        return float("nan")

    first_expiry = _nearest_expiry(filtered_chain)
    second_expiry = _second_nearest_expiry(filtered_chain)
    if second_expiry is None:
        return float("nan")

    front_iv = _atm_call_iv(filtered_chain, first_expiry, target_delta)
    second_iv = _atm_call_iv(filtered_chain, second_expiry, target_delta)

    if np.isnan(front_iv) or np.isnan(second_iv):
        return float("nan")

    return float(second_iv - front_iv)


def build_option_features(options_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build daily option-derived features for each trade date.

    Parameters
    ----------
    options_df : pd.DataFrame
        Option chain table containing at least:
        trade_date, expiry_date, option_type, delta, bid_iv, ask_iv.
    config : dict
        Option feature settings with keys:
        atm_delta, skew_delta, min_dte, max_dte.

    Returns
    -------
    pd.DataFrame
        One row per trade_date with columns:
        trade_date, atm_iv, iv_skew, iv_term_slope.
    """
    atm_delta = config["atm_delta"]
    skew_delta = config["skew_delta"]
    min_dte = config["min_dte"]
    max_dte = config["max_dte"]
    # Term slope needs a longer DTE window so a second expiry can appear.
    term_max_dte = max(max_dte, 60)

    daily_rows: list[dict[str, object]] = []

    for trade_date, day_chain in options_df.groupby("trade_date"):
        daily_rows.append(
            {
                "trade_date": pd.Timestamp(trade_date),
                "atm_iv": compute_atm_iv(
                    day_chain,
                    target_delta=atm_delta,
                    min_dte=min_dte,
                    max_dte=max_dte,
                ),
                "iv_skew": compute_iv_skew(
                    day_chain,
                    skew_delta=skew_delta,
                    min_dte=min_dte,
                    max_dte=max_dte,
                ),
                "iv_term_slope": compute_iv_term_slope(
                    day_chain,
                    target_delta=atm_delta,
                    min_dte=min_dte,
                    max_dte=term_max_dte,
                ),
            }
        )

    feature_df = pd.DataFrame(daily_rows)
    if feature_df.empty:
        return feature_df

    return feature_df.sort_values("trade_date").reset_index(drop=True)
