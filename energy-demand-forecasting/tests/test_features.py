from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.features import build_features, seasonal_naive_prediction


def test_features_are_backward_looking(hourly_frame: pd.DataFrame) -> None:
    featured = build_features(hourly_frame)
    building = featured[featured["building_id"] == "SiteA_office_A"].reset_index(drop=True)
    assert pd.isna(building.loc[23, "lag_24"])
    assert building.loc[24, "lag_24"] == building.loc[0, "load"]
    assert building.loc[48, "lag_48"] == building.loc[0, "load"]
    assert building.loc[168, "lag_168"] == building.loc[0, "load"]
    assert pd.isna(building.loc[28, "rolling_mean_24"])
    assert np.isclose(
        building.loc[29, "rolling_mean_24"], building.loc[0:5, "load"].mean()
    )


def test_day_ahead_features_ignore_values_after_target_cutoff(
    hourly_frame: pd.DataFrame,
) -> None:
    building_id = "SiteA_office_A"
    original = hourly_frame.copy()
    target_time = pd.Timestamp("2024-01-13 12:00:00")
    cutoff = target_time - pd.Timedelta(hours=24)
    changed = original.copy()
    changed.loc[
        (changed["building_id"] == building_id)
        & (changed["timestamp"] > cutoff)
        & (changed["timestamp"] <= target_time),
        "load",
    ] = 9_999_999.0
    original_features = build_features(original)
    changed_features = build_features(changed)
    original_row = original_features[
        (original_features["building_id"] == building_id)
        & (original_features["timestamp"] == target_time)
    ].iloc[0]
    changed_row = changed_features[
        (changed_features["building_id"] == building_id)
        & (changed_features["timestamp"] == target_time)
    ].iloc[0]
    feature_columns = [column for column in original_features if column.startswith(("lag_", "rolling_"))]
    assert np.allclose(
        original_row[feature_columns].to_numpy(dtype=float),
        changed_row[feature_columns].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_seasonal_naive_uses_ordered_fallbacks(hourly_frame: pd.DataFrame) -> None:
    featured = build_features(hourly_frame)
    prediction = seasonal_naive_prediction(featured)
    row = featured.index[
        (featured["building_id"] == "SiteA_office_A")
        & (featured["timestamp"] == pd.Timestamp("2024-01-08 00:00:00"))
    ][0]
    assert prediction.loc[row] == featured.loc[row, "lag_168"]


def test_seasonal_naive_has_no_one_hour_fallback(hourly_frame: pd.DataFrame) -> None:
    featured = build_features(hourly_frame)
    building = featured[featured["building_id"] == "SiteA_office_A"]
    early_row = building.index[10]
    assert pd.isna(seasonal_naive_prediction(featured).loc[early_row])


def test_feature_builder_rejects_missing_hour_key(hourly_frame: pd.DataFrame) -> None:
    incomplete = hourly_frame.drop(index=10)
    with pytest.raises(ValueError, match="complete hourly key grid"):
        build_features(incomplete)
