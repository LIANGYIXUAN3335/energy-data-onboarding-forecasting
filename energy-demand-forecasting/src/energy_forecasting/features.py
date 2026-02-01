from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


# v2 feature set: reproduces the committed canonical_24h_v2_final run exactly.
NUMERIC_FEATURES_V2 = [
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

# v3 feature set (protocol amendment of 2026-09-28): richer load history,
# a full calendar, and lagged site weather. Every load- and weather-derived
# value is observed at t-24 or earlier for a target at t.
LOAD_FEATURES_V3 = [
    "lag_24",
    "lag_48",
    "lag_168",
    "lag_336",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_min_24",
    "rolling_max_24",
    "rolling_mean_168",
    "rolling_std_168",
]
CALENDAR_FEATURES_V3 = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_us_federal_holiday",
]
WEATHER_FEATURES_V3 = [
    "air_temperature_lag_24",
    "air_temperature_lag_48",
    "air_temperature_lag_168",
    "dew_temperature_lag_24",
    "wind_speed_lag_24",
    "air_temperature_mean_24_lag_24",
    "air_temperature_mean_168_lag_24",
]
NUMERIC_FEATURES_V3 = LOAD_FEATURES_V3 + CALENDAR_FEATURES_V3

# Backwards-compatible aliases used by the v2 code paths.
NUMERIC_FEATURES = NUMERIC_FEATURES_V2
CATEGORICAL_FEATURES = ["building_id", "site_id", "primary_use"]

WEATHER_VARIABLES = [
    "air_temperature",
    "dew_temperature",
    "wind_speed",
    "cloud_coverage",
    "precip_depth_1hr",
    "sea_lvl_pressure",
]

FEATURE_SETS = {"v2", "v3"}


def numeric_feature_columns(feature_set: str = "v2", *, weather: bool = False) -> list[str]:
    """Return the ordered numeric feature list for a feature set."""

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}")
    if feature_set == "v2":
        if weather:
            raise ValueError("Weather features require feature_set='v3'")
        return list(NUMERIC_FEATURES_V2)
    columns = list(NUMERIC_FEATURES_V3)
    if weather:
        columns += WEATHER_FEATURES_V3
    return columns


def _match_timezone(frame: pd.DataFrame, column: str, reference: pd.Series) -> pd.DataFrame:
    """Give ``frame[column]`` the same tz-awareness as ``reference`` for a safe merge.

    Producer files are UTC-marked; small fixtures may be naive. The marker is a
    comparison convention, so aligning it never shifts a wall-clock value.
    """

    result = frame.copy()
    target_tz = getattr(reference.dt, "tz", None)
    values = pd.to_datetime(result[column])
    if target_tz is None and values.dt.tz is not None:
        result[column] = values.dt.tz_convert("UTC").dt.tz_localize(None)
    elif target_tz is not None and values.dt.tz is None:
        result[column] = values.dt.tz_localize("UTC")
    else:
        result[column] = values
    return result


def _check_hourly_grid(ordered: pd.DataFrame, key: str) -> None:
    deltas = ordered.groupby(key, sort=False, observed=True)["timestamp"].diff()
    unexpected_intervals = deltas.notna() & (deltas != pd.Timedelta(hours=1))
    if unexpected_intervals.any():
        raise ValueError(
            f"Feature construction requires a complete hourly key grid per {key}; "
            f"found {int(unexpected_intervals.sum())} non-hourly intervals"
        )


def build_weather_features(
    weather: pd.DataFrame, forecast_horizon_hours: int = 24
) -> pd.DataFrame:
    """Lagged site weather that is known at forecast issuance.

    For a target at ``t`` every weather feature is an observation at ``t-24``
    or earlier (lags of 24, 48 and 168 hours, and trailing means ending at
    ``t-24``). Observed weather closer to the target, which would only be
    available through a weather forecast, is intentionally excluded.
    """

    if forecast_horizon_hours != 24:
        raise ValueError("Phase 1 supports forecast_horizon_hours=24 only")
    required = {"timestamp", "site_id", "air_temperature", "dew_temperature", "wind_speed"}
    missing = sorted(required.difference(weather.columns))
    if missing:
        raise ValueError(f"Weather frame is missing columns: {missing}")
    ordered = weather.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce", utc=True)
    ordered["site_id"] = ordered["site_id"].astype("string")
    ordered = ordered.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    if ordered.duplicated(["site_id", "timestamp"]).any():
        raise ValueError("Weather frame contains duplicate site/timestamp keys")
    _check_hourly_grid(ordered, "site_id")
    groups = ordered.groupby("site_id", sort=False, observed=True)
    for lag in (24, 48, 168):
        ordered[f"air_temperature_lag_{lag}"] = groups["air_temperature"].shift(lag)
    ordered["dew_temperature_lag_24"] = groups["dew_temperature"].shift(24)
    ordered["wind_speed_lag_24"] = groups["wind_speed"].shift(24)
    shifted = groups["air_temperature"].shift(forecast_horizon_hours)
    shifted_groups = shifted.groupby(ordered["site_id"], sort=False, observed=True)
    for window in (24, 168):
        rolling = shifted_groups.rolling(window=window, min_periods=max(4, window // 4))
        ordered[f"air_temperature_mean_{window}_lag_24"] = rolling.mean().reset_index(
            level=0, drop=True
        )
    return ordered[["timestamp", "site_id", *WEATHER_FEATURES_V3]]


def build_features(
    frame: pd.DataFrame,
    forecast_horizon_hours: int = 24,
    *,
    feature_set: str = "v2",
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create features available at least one forecast horizon before target.

    Phase 1 deliberately supports one fixed 24-hour horizon. For a target at
    time ``t``, every load- and weather-derived feature uses observations from
    ``t-24`` or earlier. Target calendar fields remain valid because the target
    timestamp is known when a day-ahead forecast is issued.
    """
    if forecast_horizon_hours != 24:
        raise ValueError("Phase 1 supports forecast_horizon_hours=24 only")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}")
    if weather is not None and feature_set != "v3":
        raise ValueError("Weather features require feature_set='v3'")
    ordered = frame.sort_values(["building_id", "timestamp"]).copy()
    _check_hourly_grid(ordered, "building_id")
    groups = ordered.groupby("building_id", sort=False, observed=True)["load"]
    lags = (24, 48, 168, 336) if feature_set == "v3" else (24, 48, 168)
    for lag in lags:
        ordered[f"lag_{lag}"] = groups.shift(lag)

    shifted = groups.shift(forecast_horizon_hours)
    shifted_groups = shifted.groupby(ordered["building_id"], sort=False, observed=True)
    for window in (24, 168):
        rolling = shifted_groups.rolling(window=window, min_periods=max(4, window // 4))
        ordered[f"rolling_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        ordered[f"rolling_std_{window}"] = (
            rolling.std(ddof=0).reset_index(level=0, drop=True)
        )
        if feature_set == "v3" and window == 24:
            ordered["rolling_min_24"] = rolling.min().reset_index(level=0, drop=True)
            ordered["rolling_max_24"] = rolling.max().reset_index(level=0, drop=True)

    hour = ordered["timestamp"].dt.hour.astype(float)
    day = ordered["timestamp"].dt.dayofweek.astype(float)
    ordered["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    ordered["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    ordered["dow_sin"] = np.sin(2 * np.pi * day / 7.0)
    ordered["dow_cos"] = np.cos(2 * np.pi * day / 7.0)
    if feature_set == "v3":
        day_of_year = ordered["timestamp"].dt.dayofyear.astype(float)
        month = ordered["timestamp"].dt.month.astype(float)
        ordered["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        ordered["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
        ordered["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        ordered["month_cos"] = np.cos(2 * np.pi * month / 12.0)
        ordered["is_weekend"] = (day >= 5).astype(float)
        dates = ordered["timestamp"].dt.tz_localize(None).dt.normalize()
        holidays = USFederalHolidayCalendar().holidays(
            start=dates.min() - pd.Timedelta(days=1), end=dates.max() + pd.Timedelta(days=1)
        )
        ordered["is_us_federal_holiday"] = dates.isin(holidays).astype(float)
        if weather is not None:
            weather_features = build_weather_features(weather, forecast_horizon_hours)
            weather_features = _match_timezone(weather_features, "timestamp", ordered["timestamp"])
            ordered["site_id"] = ordered["site_id"].astype("string")
            before = len(ordered)
            ordered = ordered.merge(
                weather_features, on=["site_id", "timestamp"], how="left", validate="many_to_one"
            )
            if len(ordered) != before:
                raise AssertionError("Weather join changed the number of rows")
            ordered = ordered.sort_values(["building_id", "timestamp"])
    return ordered


def seasonal_naive_prediction(frame: pd.DataFrame) -> pd.Series:
    prediction = frame["lag_168"].copy()
    prediction = prediction.fillna(frame["lag_24"])
    return prediction
