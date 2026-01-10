from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "lag_24",
    "lag_48",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "rolling_std_168",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

CATEGORICAL_FEATURES = ["building_id", "site_id", "primary_use"]


def build_features(
    frame: pd.DataFrame,
    forecast_horizon_hours: int = 24,
) -> pd.DataFrame:
    """Create features available at least one forecast horizon before target.

    Phase 1 deliberately supports one fixed 24-hour horizon. For a target at
    time ``t``, every load-derived feature uses observations from ``t-24`` or
    earlier. Target calendar fields remain valid because the target timestamp is
    known when a day-ahead forecast is issued.
    """
    if forecast_horizon_hours != 24:
        raise ValueError("Phase 1 supports forecast_horizon_hours=24 only")
    ordered = frame.sort_values(["building_id", "timestamp"]).copy()
    deltas = ordered.groupby("building_id", sort=False, observed=True)[
        "timestamp"
    ].diff()
    unexpected_intervals = deltas.notna() & (deltas != pd.Timedelta(hours=1))
    if unexpected_intervals.any():
        raise ValueError(
            "Feature construction requires a complete hourly key grid per building; "
            f"found {int(unexpected_intervals.sum())} non-hourly intervals"
        )
    groups = ordered.groupby("building_id", sort=False, observed=True)["load"]
    for lag in (24, 48, 168):
        ordered[f"lag_{lag}"] = groups.shift(lag)

    shifted = groups.shift(forecast_horizon_hours)
    shifted_groups = shifted.groupby(ordered["building_id"], sort=False, observed=True)
    for window in (24, 168):
        rolling = shifted_groups.rolling(window=window, min_periods=max(4, window // 4))
        ordered[f"rolling_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        ordered[f"rolling_std_{window}"] = (
            rolling.std(ddof=0).reset_index(level=0, drop=True)
        )

    hour = ordered["timestamp"].dt.hour.astype(float)
    day = ordered["timestamp"].dt.dayofweek.astype(float)
    ordered["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    ordered["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    ordered["dow_sin"] = np.sin(2 * np.pi * day / 7.0)
    ordered["dow_cos"] = np.cos(2 * np.pi * day / 7.0)
    return ordered


def seasonal_naive_prediction(frame: pd.DataFrame) -> pd.Series:
    prediction = frame["lag_168"].copy()
    prediction = prediction.fillna(frame["lag_24"])
    return prediction
