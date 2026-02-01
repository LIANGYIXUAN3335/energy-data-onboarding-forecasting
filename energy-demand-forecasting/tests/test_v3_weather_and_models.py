"""v3 feature set, lagged weather, and multi-model runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.dataset import sha256_file
from energy_forecasting.experiment import load_config, run_experiment
from energy_forecasting.features import (
    CATEGORICAL_FEATURES,
    WEATHER_FEATURES_V3,
    build_features,
    build_weather_features,
    numeric_feature_columns,
)
from energy_forecasting.sensitivity import (
    build_direct_examples,
    run_sensitivity_experiment,
    verify_sensitivity_result_directory,
)
from energy_forecasting.verification import verify_result_directory


def _weather_frame(hourly_frame: pd.DataFrame) -> pd.DataFrame:
    timestamps = hourly_frame["timestamp"].drop_duplicates().sort_values()
    hours = np.arange(len(timestamps), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps.to_numpy(),
            "site_id": "SiteA",
            "air_temperature": 10 + 8 * np.sin(2 * np.pi * hours / 24) + hours * 0.01,
            "dew_temperature": 5 + 2 * np.cos(2 * np.pi * hours / 24),
            "wind_speed": 3.0 + (hours % 7),
            "cloud_coverage": np.nan,
            "precip_depth_1hr": 0.0,
            "sea_lvl_pressure": 1013.0,
        }
    )


def _register_weather(bundle_dir: Path, weather: pd.DataFrame) -> Path:
    path = bundle_dir / "weather.csv.gz"
    weather.to_csv(path, index=False, compression="gzip")
    manifest_path = bundle_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["weather.csv.gz"] = {
        "path": "weather.csv.gz",
        "sha256": sha256_file(path),
        "rows": int(len(weather)),
        "columns": list(weather.columns),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _v3_config(input_bundle: dict[str, Path], tmp_path: Path, **overrides: object) -> Path:
    config = json.loads(input_bundle["config"].read_text(encoding="utf-8"))
    config.update(
        {
            "models": ["seasonal_naive", "ridge", "random_forest", "hist_gradient_boosting"],
            "feature_set": "v3",
            "weather_features": True,
            "random_forest_n_estimators": 12,
            "random_forest_n_jobs": 1,
        }
    )
    config.update(overrides)
    path = tmp_path / "config_v3.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_v3_feature_lists_are_documented_and_exceed_twenty_five() -> None:
    assert numeric_feature_columns("v2") == [
        "lag_24", "lag_48", "lag_168", "rolling_mean_24", "rolling_std_24",
        "rolling_mean_168", "rolling_std_168", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]
    v3 = numeric_feature_columns("v3", weather=True)
    assert len(v3) + len(CATEGORICAL_FEATURES) >= 25
    assert set(WEATHER_FEATURES_V3) <= set(v3)
    with pytest.raises(ValueError, match="Weather features require"):
        numeric_feature_columns("v2", weather=True)


def test_weather_features_ignore_observations_after_target_minus_24h(
    hourly_frame: pd.DataFrame,
) -> None:
    weather = _weather_frame(hourly_frame)
    target_time = pd.Timestamp("2024-01-20 15:00:00")
    cutoff = target_time - pd.Timedelta(hours=24)
    changed = weather.copy()
    late = (changed["timestamp"] > cutoff) & (changed["timestamp"] <= target_time + pd.Timedelta(hours=48))
    changed.loc[late, ["air_temperature", "dew_temperature", "wind_speed"]] = 9_999.0
    original = build_features(hourly_frame, feature_set="v3", weather=weather)
    altered = build_features(hourly_frame, feature_set="v3", weather=changed)
    row = (original["building_id"] == "SiteA_office_A") & (original["timestamp"] == target_time)
    assert np.allclose(
        original.loc[row, WEATHER_FEATURES_V3].to_numpy(dtype=float),
        altered.loc[row, WEATHER_FEATURES_V3].to_numpy(dtype=float),
        equal_nan=True,
    )
    # And the lag really is the value observed 24 hours earlier.
    weather_features = build_weather_features(weather)
    weather_features["timestamp"] = weather_features["timestamp"].dt.tz_localize(None)
    source = weather.loc[weather["timestamp"] == cutoff, "air_temperature"].iloc[0]
    observed = weather_features.loc[weather_features["timestamp"] == target_time, "air_temperature_lag_24"].iloc[0]
    assert np.isclose(source, observed)


def test_v3_calendar_features_are_known_at_issuance(hourly_frame: pd.DataFrame) -> None:
    featured = build_features(hourly_frame, feature_set="v3")
    new_year = featured.loc[featured["timestamp"] == pd.Timestamp("2024-01-01 12:00:00")]
    assert (new_year["is_us_federal_holiday"] == 1.0).all()
    weekday = featured.loc[featured["timestamp"] == pd.Timestamp("2024-01-03 12:00:00")]
    assert (weekday["is_us_federal_holiday"] == 0.0).all()
    assert (weekday["is_weekend"] == 0.0).all()
    saturday = featured.loc[featured["timestamp"] == pd.Timestamp("2024-01-06 12:00:00")]
    assert (saturday["is_weekend"] == 1.0).all()
    assert featured["lag_336"].notna().sum() > 0


def test_v3_end_to_end_with_four_models_and_weather(
    input_bundle: dict[str, Path], hourly_frame: pd.DataFrame, tmp_path: Path
) -> None:
    weather_path = _register_weather(input_bundle["input_dir"], _weather_frame(hourly_frame))
    config_path = _v3_config(input_bundle, tmp_path)
    output = tmp_path / "results_v3"
    manifest = run_experiment(
        input_bundle["reference"],
        input_bundle["corrupted"],
        input_bundle["remediated"],
        config_path,
        output,
        producer_manifest_path=input_bundle["producer_manifest"],
        fault_manifest_path=input_bundle["fault_manifest"],
    )
    assert manifest["feature_set"] == "v3"
    assert manifest["weather"]["sha256"] == sha256_file(weather_path)
    assert manifest["model_order"] == ["seasonal_naive", "ridge", "random_forest", "hist_gradient_boosting"]
    assert manifest["models"]["random_forest"]["n_estimators"] == 12
    assert manifest["feature_count"] >= 25
    metrics = pd.read_csv(output / "metrics.csv")
    assert len(metrics) == 12
    assert set(metrics["model"]) == set(manifest["model_order"])
    paired = pd.read_csv(output / "paired_differences.csv")
    assert set(paired["model"]) == set(manifest["model_order"])
    summary = verify_result_directory(output, input_dir=input_bundle["input_dir"])
    assert summary["source_binding_verified"] is True
    card = (output / "model_card.md").read_text(encoding="utf-8")
    assert "Ridge regression" in card and "Random forest" in card and "air_temperature_lag_24" in card


def test_weather_bundle_is_hash_bound_to_the_producer_manifest(
    input_bundle: dict[str, Path], hourly_frame: pd.DataFrame, tmp_path: Path
) -> None:
    weather_path = _register_weather(input_bundle["input_dir"], _weather_frame(hourly_frame))
    config_path = _v3_config(input_bundle, tmp_path, models=["seasonal_naive", "ridge"])
    tampered = _weather_frame(hourly_frame)
    tampered["air_temperature"] += 1.0
    tampered.to_csv(weather_path, index=False, compression="gzip")
    with pytest.raises(ValueError, match="does not match the producer manifest hash"):
        run_experiment(
            input_bundle["reference"],
            input_bundle["corrupted"],
            input_bundle["remediated"],
            config_path,
            tmp_path / "out",
            producer_manifest_path=input_bundle["producer_manifest"],
            fault_manifest_path=input_bundle["fault_manifest"],
        )


def test_config_rejects_unknown_models_or_weather_with_v2(input_bundle: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="models must list"):
        load_config(_v3_config(input_bundle, tmp_path, models=["seasonal_naive", "lstm"]))
    with pytest.raises(ValueError, match="models must list"):
        load_config(_v3_config(input_bundle, tmp_path, models=["ridge"]))
    with pytest.raises(ValueError, match="Weather features require"):
        load_config(_v3_config(input_bundle, tmp_path, feature_set="v2"))


def test_sensitivity_v3_with_weather_and_ridge(
    input_bundle: dict[str, Path], hourly_frame: pd.DataFrame, tmp_path: Path
) -> None:
    _register_weather(input_bundle["input_dir"], _weather_frame(hourly_frame))
    unique_times = hourly_frame["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    fault_cutoff = pd.Timestamp(json.loads(input_bundle["fault_manifest"].read_text())["fault_cutoff"])
    train_end = fault_cutoff - pd.Timedelta(hours=1)
    validation_end = unique_times.iloc[int(len(unique_times) * 0.7)]
    config = {
        "analysis_kind": "direct_horizons_1_to_24_sensitivity",
        "seed": 42,
        "horizons_hours": list(range(1, 25)),
        "origin_hour": 23,
        "origin_stride_hours": 24,
        "seasonal_period_hours": 168,
        "train_end": train_end.isoformat(),
        "validation_end": validation_end.isoformat(),
        "max_model_iterations": 8,
        "learning_rate": 0.1,
        "max_leaf_nodes": 15,
        "l2_regularization": 0.1,
        "bootstrap_repetitions": 5,
        "bootstrap_block": "building_week",
        "models": ["seasonal_naive", "ridge", "hist_gradient_boosting"],
        "weather_features": True,
    }
    config_path = tmp_path / "sens_v3.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "sens_out"
    manifest = run_sensitivity_experiment(
        input_bundle["reference"],
        input_bundle["corrupted"],
        input_bundle["remediated"],
        config_path,
        output,
        producer_manifest_path=input_bundle["producer_manifest"],
        fault_manifest_path=input_bundle["fault_manifest"],
    )
    assert "origin_air_temperature" in manifest["features"]
    assert manifest["model_order"] == ["seasonal_naive", "ridge", "hist_gradient_boosting"]
    assert len(pd.read_csv(output / "paired_differences.csv")) == 3
    assert len(pd.read_csv(output / "horizon_metrics.csv")) == 9 * 24
    summary = verify_sensitivity_result_directory(output, input_dir=input_bundle["input_dir"])
    assert summary["source_binding_verified"] is True


def test_direct_weather_never_reads_after_the_origin(hourly_frame: pd.DataFrame) -> None:
    weather = _weather_frame(hourly_frame)
    origin = pd.Timestamp("2024-01-20 23:00:00")
    changed = weather.copy()
    changed.loc[changed["timestamp"] > origin, ["air_temperature", "dew_temperature", "wind_speed"]] = 9_999.0
    original = build_direct_examples(hourly_frame, weather=weather)
    altered = build_direct_examples(hourly_frame, weather=changed)
    rows = original["forecast_origin"] == origin
    columns = [column for column in original.columns if column.startswith("origin_air") or column.startswith("origin_dew") or column.startswith("origin_wind")]
    assert np.allclose(
        original.loc[rows, columns].to_numpy(dtype=float),
        altered.loc[rows, columns].to_numpy(dtype=float),
        equal_nan=True,
    )
