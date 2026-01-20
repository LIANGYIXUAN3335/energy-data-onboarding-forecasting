"""Site-level weather onboarding for the selected BDG2 sites.

Weather is published as one additional, condition-independent artifact
(`weather.csv.gz`). It is joined by `(timestamp, site_id)` downstream and is
never modified by the fault suites: the three load conditions differ only in
their meter values. Timestamps follow the same source-local wall-clock
convention with a UTC marker as the meter data.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


SOURCE_COLUMNS = {
    "airTemperature": "air_temperature",
    "dewTemperature": "dew_temperature",
    "windSpeed": "wind_speed",
    "cloudCoverage": "cloud_coverage",
    "precipDepth1HR": "precip_depth_1hr",
    "seaLvlPressure": "sea_lvl_pressure",
}

WEATHER_COLUMNS = ["timestamp", "site_id", *SOURCE_COLUMNS.values()]


def read_weather_csv(path: str | Any) -> pd.DataFrame:
    """Read the BDG2 weather file and keep only the documented variables."""

    source = pd.read_csv(path)
    missing = sorted({"timestamp", "site_id", *SOURCE_COLUMNS}.difference(source.columns))
    if missing:
        raise ValueError(f"Weather CSV is missing required columns: {missing}")
    frame = source.loc[:, ["timestamp", "site_id", *SOURCE_COLUMNS]].rename(columns=SOURCE_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["site_id"] = frame["site_id"].astype("string").str.strip()
    for column in SOURCE_COLUMNS.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def prepare_weather(
    weather: pd.DataFrame,
    *,
    site_ids: Iterable[str],
    grid_start: pd.Timestamp,
    grid_end: pd.Timestamp,
    frequency: str = "1h",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize weather for the selected sites onto the complete hourly grid.

    Duplicate `(timestamp, site_id)` rows are averaged, missing grid hours are
    materialized as null rows, and non-finite values become null. No value is
    imputed here; the consumer imputes inside a training-only preprocessing
    step so that no test-period weather leaks into training statistics.
    """

    selected = sorted({str(site) for site in site_ids})
    frame = weather.loc[weather["site_id"].isin(selected) & weather["timestamp"].notna()].copy()
    unparseable = int(weather["timestamp"].isna().sum())
    duplicates = int(frame.duplicated(["timestamp", "site_id"]).sum())
    frame = (
        frame.groupby(["timestamp", "site_id"], as_index=False, sort=True)[list(SOURCE_COLUMNS.values())]
        .mean()
    )
    grid = pd.date_range(grid_start, grid_end, freq=frequency, tz="UTC")
    full = pd.MultiIndex.from_product([grid, selected], names=["timestamp", "site_id"]).to_frame(index=False)
    merged = full.merge(frame, on=["timestamp", "site_id"], how="left", validate="1:1")
    for column in SOURCE_COLUMNS.values():
        values = merged[column].to_numpy(dtype=float)
        merged[column] = np.where(np.isfinite(values), values, np.nan)
    merged["site_id"] = merged["site_id"].astype("string")
    merged = merged.sort_values(["timestamp", "site_id"], kind="stable").reset_index(drop=True)
    merged = merged.loc[:, WEATHER_COLUMNS]

    materialized = int(len(merged) - len(frame))
    summary: dict[str, Any] = {
        "sites": selected,
        "rows": int(len(merged)),
        "grid_start": grid_start.isoformat(),
        "grid_end": grid_end.isoformat(),
        "source_rows_for_selected_sites": int(len(frame) + duplicates),
        "duplicate_keys_averaged": duplicates,
        "grid_hours_materialized_as_null": max(materialized, 0),
        "unparseable_timestamps_in_source": unparseable,
        "missing_rate_by_site": {
            site: {
                column: float(group[column].isna().mean())
                for column in SOURCE_COLUMNS.values()
            }
            for site, group in merged.groupby("site_id", sort=True)
        },
        "variables": list(SOURCE_COLUMNS.values()),
        "policy": (
            "condition-independent; not touched by fault suites; nulls preserved; "
            "downstream imputation is fitted on the training partition only"
        ),
    }
    return merged, summary


def validate_weather_frame(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if list(frame.columns) != WEATHER_COLUMNS:
        errors.append(f"weather columns must be exactly {WEATHER_COLUMNS}")
        return errors
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any():
        errors.append("weather contains unparseable timestamps")
    if frame.duplicated(["timestamp", "site_id"]).any():
        errors.append("weather contains duplicate timestamp/site_id keys")
    return errors
