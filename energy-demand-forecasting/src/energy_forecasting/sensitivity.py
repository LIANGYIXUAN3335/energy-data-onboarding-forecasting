from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import __version__
from .dataset import (
    SplitBoundary,
    choose_explicit_split_boundaries,
    partition,
    sha256_file,
)
from .experiment import _source_tree_revision
from .figures import render_horizon_lines_svg
from .metrics import (
    building_scale_coverage,
    calculate_metrics_blocked,
    paired_absolute_error_difference_blocked,
    per_building_mase,
    seasonal_scale,
    seasonal_scales_by_building,
)
from .features import WEATHER_VARIABLES
from .models import LEARNED_MODELS, SUPPORTED_MODELS, ModelConfig, build_model
from .verification import (
    _json_integer,
    expected_models,
    verify_input_bundle,
    verify_post_cutoff_condition_parity,
    verify_result_source_binding,
)


HORIZONS = tuple(range(1, 25))
DIRECT_NUMERIC_FEATURES = [
    "origin_load",
    "origin_lag_24",
    "origin_lag_48",
    "origin_lag_168",
    "origin_rolling_mean_24",
    "origin_rolling_std_24",
    "origin_rolling_mean_168",
    "origin_rolling_std_168",
    "horizon_hours",
    "target_hour_sin",
    "target_hour_cos",
    "target_dow_sin",
    "target_dow_cos",
]
DIRECT_CATEGORICAL_FEATURES = ["building_id", "site_id", "primary_use"]
# v3: site weather observed at or before the forecast origin.
DIRECT_WEATHER_FEATURES = [
    "origin_air_temperature",
    "origin_air_temperature_lag_24",
    "origin_air_temperature_lag_168",
    "origin_dew_temperature",
    "origin_wind_speed",
    "origin_air_temperature_mean_24",
]
FORECAST_KEYS = [
    "forecast_origin",
    "horizon_hours",
    "target_timestamp",
    "building_id",
]
PREDICTION_KEYS = FORECAST_KEYS + ["condition", "model"]
SENSITIVITY_RESULT_FILES = {
    "metrics.csv",
    "horizon_metrics.csv",
    "building_mase.csv",
    "paired_differences.csv",
    "horizon_paired_differences.csv",
    "predictions.csv.gz",
    "run_manifest.json",
    "report.md",
    "figures/mae_by_horizon.svg",
}


def load_sensitivity_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "analysis_kind",
        "seed",
        "horizons_hours",
        "origin_hour",
        "origin_stride_hours",
        "seasonal_period_hours",
        "train_end",
        "validation_end",
        "max_model_iterations",
        "learning_rate",
        "max_leaf_nodes",
        "l2_regularization",
        "bootstrap_repetitions",
        "bootstrap_block",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Sensitivity configuration is missing keys: {sorted(missing)}")
    if config["analysis_kind"] != "direct_horizons_1_to_24_sensitivity":
        raise ValueError("analysis_kind must explicitly identify the 1–24h sensitivity")
    horizons = tuple(int(value) for value in config["horizons_hours"])
    if horizons != HORIZONS:
        raise ValueError("Sensitivity horizons_hours must be exactly 1 through 24")
    if max(horizons) > 24:
        raise ValueError("Sensitivity horizon must not exceed 24 hours")
    origin_hour = int(config["origin_hour"])
    if not 0 <= origin_hour <= 23:
        raise ValueError("origin_hour must be an integer from 0 through 23")
    if int(config["origin_stride_hours"]) != 24:
        raise ValueError("origin_stride_hours must be 24 for one complete daily origin")
    if config["bootstrap_block"] != "building_week":
        raise ValueError("bootstrap_block must be 'building_week'")
    if int(config["bootstrap_repetitions"]) < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    models = config.get("models", ["seasonal_naive", "hist_gradient_boosting"])
    if (
        not isinstance(models, list)
        or len(models) != len(set(models))
        or "seasonal_naive" not in models
        or not any(model in LEARNED_MODELS for model in models)
        or any(model not in SUPPORTED_MODELS for model in models)
    ):
        raise ValueError(
            "models must list seasonal_naive plus at least one learned model from "
            f"{LEARNED_MODELS}, without duplicates"
        )
    config["models"] = list(models)
    config["weather_features"] = bool(config.get("weather_features", False))
    return config


def _complete_hourly_grid(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["building_id", "timestamp"]).copy()
    # utc=True is a comparison-marker normalization, not a true UTC
    # instant conversion. Source-local hour/day fields are intentionally retained.
    ordered["timestamp"] = pd.to_datetime(
        ordered["timestamp"], errors="coerce", utc=True
    )
    if ordered["timestamp"].isna().any():
        raise ValueError("Direct sensitivity received unparseable timestamps")
    deltas = ordered.groupby("building_id", sort=False, observed=True)[
        "timestamp"
    ].diff()
    unexpected = deltas.notna() & (deltas != pd.Timedelta(hours=1))
    if unexpected.any():
        raise ValueError(
            "Direct sensitivity requires a complete hourly key grid per building; "
            f"found {int(unexpected.sum())} non-hourly intervals"
        )
    return ordered


def build_direct_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Site weather features observed at or before each candidate origin hour."""

    ordered = weather.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce", utc=True)
    ordered["site_id"] = ordered["site_id"].astype("string")
    ordered = ordered.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    if ordered.duplicated(["site_id", "timestamp"]).any():
        raise ValueError("Weather frame contains duplicate site/timestamp keys")
    deltas = ordered.groupby("site_id", sort=False, observed=True)["timestamp"].diff()
    if (deltas.notna() & (deltas != pd.Timedelta(hours=1))).any():
        raise ValueError("Weather features require a complete hourly grid per site")
    groups = ordered.groupby("site_id", sort=False, observed=True)
    ordered["origin_air_temperature"] = ordered["air_temperature"]
    ordered["origin_air_temperature_lag_24"] = groups["air_temperature"].shift(24)
    ordered["origin_air_temperature_lag_168"] = groups["air_temperature"].shift(168)
    ordered["origin_dew_temperature"] = ordered["dew_temperature"]
    ordered["origin_wind_speed"] = ordered["wind_speed"]
    ordered["origin_air_temperature_mean_24"] = (
        groups["air_temperature"].rolling(window=24, min_periods=6).mean().reset_index(level=0, drop=True)
    )
    return ordered[["timestamp", "site_id", *DIRECT_WEATHER_FEATURES]].rename(
        columns={"timestamp": "forecast_origin"}
    )


def build_direct_examples(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    origin_hour: int = 23,
    origin_stride_hours: int = 24,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one daily origin with 24 direct targets and past-only features.

    ``forecast_origin`` and ``cutoff_timestamp`` are identical. Every load
    feature is observed at or before that cutoff. Target hour/day features use
    the preserved source-local wall-clock calendar represented by the canonical
    timestamp marker; they do not assert a true UTC instant.
    """

    if tuple(horizons) != HORIZONS or max(horizons) > 24:
        raise ValueError("Direct sensitivity requires the exact horizons 1 through 24")
    if not 0 <= int(origin_hour) <= 23:
        raise ValueError("origin_hour must be from 0 through 23")
    if int(origin_stride_hours) != 24:
        raise ValueError("origin_stride_hours must be 24")

    ordered = _complete_hourly_grid(frame)
    groups = ordered.groupby("building_id", sort=False, observed=True)["load"]
    ordered["origin_load"] = ordered["load"]
    for lag in (24, 48, 168):
        ordered[f"origin_lag_{lag}"] = groups.shift(lag)
    for window in (24, 168):
        rolling = groups.rolling(window=window, min_periods=max(4, window // 4))
        ordered[f"origin_rolling_mean_{window}"] = (
            rolling.mean().reset_index(level=0, drop=True)
        )
        ordered[f"origin_rolling_std_{window}"] = (
            rolling.std(ddof=0).reset_index(level=0, drop=True)
        )

    origin_columns = [
        "timestamp",
        "building_id",
        "site_id",
        "primary_use",
        *DIRECT_NUMERIC_FEATURES[:8],
    ]
    origins = ordered.loc[
        ordered["timestamp"].dt.hour == int(origin_hour), origin_columns
    ].rename(columns={"timestamp": "forecast_origin"})
    if origins.empty:
        raise ValueError("No rows match the configured daily origin hour")
    expanded = origins.loc[origins.index.repeat(len(horizons))].reset_index(drop=True)
    expanded["horizon_hours"] = np.tile(np.asarray(horizons, dtype=int), len(origins))
    expanded["cutoff_timestamp"] = expanded["forecast_origin"]
    expanded["target_timestamp"] = expanded["forecast_origin"] + pd.to_timedelta(
        expanded["horizon_hours"], unit="h"
    )

    target_lookup = ordered[["timestamp", "building_id", "load"]].rename(
        columns={"timestamp": "target_timestamp", "load": "condition_target"}
    )
    expanded = expanded.merge(
        target_lookup,
        on=["target_timestamp", "building_id"],
        how="left",
        validate="many_to_one",
        indicator="_target_key_status",
    )
    full = (
        expanded["_target_key_status"].eq("both")
        .groupby([expanded["forecast_origin"], expanded["building_id"]], observed=True)
        .sum()
    )
    complete_keys = full[full == len(horizons)].index
    complete = pd.DataFrame(
        complete_keys.tolist(), columns=["forecast_origin", "building_id"]
    )
    expanded = expanded.merge(
        complete,
        on=["forecast_origin", "building_id"],
        how="inner",
        validate="many_to_one",
    ).drop(columns="_target_key_status")
    if expanded.empty:
        raise ValueError("No origin has a complete 1–24 hour target key set")

    history = ordered[["timestamp", "building_id", "load"]]
    for lag in (168, 24):
        source_column = f"_direct_source_{lag}"
        expanded[source_column] = expanded["target_timestamp"] - pd.Timedelta(hours=lag)
        lookup = history.rename(
            columns={"timestamp": source_column, "load": f"direct_lag_{lag}"}
        )
        expanded = expanded.merge(
            lookup,
            on=[source_column, "building_id"],
            how="left",
            validate="many_to_one",
        )
        if (expanded[source_column] > expanded["forecast_origin"]).any():
            raise AssertionError(f"direct_lag_{lag} would cross the forecast cutoff")
        expanded = expanded.drop(columns=source_column)

    # These fields intentionally use the preserved source-local calendar.
    target_hour = expanded["target_timestamp"].dt.hour.astype(float)
    target_day = expanded["target_timestamp"].dt.dayofweek.astype(float)
    expanded["target_hour_sin"] = np.sin(2 * np.pi * target_hour / 24.0)
    expanded["target_hour_cos"] = np.cos(2 * np.pi * target_hour / 24.0)
    expanded["target_dow_sin"] = np.sin(2 * np.pi * target_day / 7.0)
    expanded["target_dow_cos"] = np.cos(2 * np.pi * target_day / 7.0)
    if weather is not None:
        from .features import _match_timezone

        origin_weather = _match_timezone(
            build_direct_weather(weather), "forecast_origin", expanded["forecast_origin"]
        )
        expanded["site_id"] = expanded["site_id"].astype("string")
        before = len(expanded)
        expanded = expanded.merge(
            origin_weather, on=["site_id", "forecast_origin"], how="left", validate="many_to_one"
        )
        if len(expanded) != before:
            raise AssertionError("Weather join changed the number of sensitivity rows")
    return expanded.sort_values(FORECAST_KEYS).reset_index(drop=True)


def assign_origin_partitions(
    examples: pd.DataFrame,
    boundary: SplitBoundary,
) -> pd.Series:
    """Partition full 24-target origin blocks by target time.

    Origins that straddle train/validation or validation/test boundaries are
    explicitly excluded, providing a target-time label embargo without moving
    the pre-specified calendar boundaries.
    """

    ranges = examples.groupby("forecast_origin", observed=True)["target_timestamp"].agg(
        ["min", "max", "nunique"]
    )
    labels: dict[pd.Timestamp, str] = {}
    for origin, row in ranges.iterrows():
        if int(row["nunique"]) != 24:
            labels[pd.Timestamp(origin)] = "excluded_incomplete_origin"
        elif row["max"] <= boundary.train_end:
            labels[pd.Timestamp(origin)] = "train"
        elif row["min"] > boundary.train_end and row["max"] <= boundary.validation_end:
            labels[pd.Timestamp(origin)] = "validation"
        elif row["min"] > boundary.validation_end and row["max"] <= boundary.test_end:
            labels[pd.Timestamp(origin)] = "test"
        else:
            labels[pd.Timestamp(origin)] = "excluded_boundary_straddle"
    return examples["forecast_origin"].map(labels).astype("string")


def validate_exclusive_fault_boundary(
    fault_cutoff: pd.Timestamp,
    train_end: pd.Timestamp,
) -> None:
    """Validate the hourly producer's exclusive no-fault boundary.

    ``fault_cutoff`` is the first timestamp at which faults are forbidden,
    whereas ``train_end`` is the final timestamp included in training. For an
    hourly dataset, a cutoff exactly one hour after the inclusive training end
    is therefore contiguous and valid. A later cutoff would leave an
    unprotected post-training interval.
    """

    cutoff = pd.Timestamp(fault_cutoff)
    end = pd.Timestamp(train_end)
    cutoff = (
        cutoff.tz_localize("UTC")
        if cutoff.tzinfo is None
        else cutoff.tz_convert("UTC")
    )
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if cutoff > end + pd.Timedelta(hours=1):
        raise ValueError(
            "fault_cutoff must be no later than the first hourly timestamp "
            "after train_end for the sensitivity protocol"
        )


def _direct_numeric_features(weather: bool) -> list[str]:
    return DIRECT_NUMERIC_FEATURES + (DIRECT_WEATHER_FEATURES if weather else [])


def _build_direct_model(
    config: ModelConfig,
    model_name: str = "hist_gradient_boosting",
    numeric_features: list[str] | None = None,
) -> Pipeline:
    return build_model(
        model_name,
        config,
        list(numeric_features or DIRECT_NUMERIC_FEATURES),
        DIRECT_CATEGORICAL_FEATURES,
    )


def _prediction_frame(
    condition: str,
    examples: pd.DataFrame,
    labels: pd.Series,
    model_config: ModelConfig,
    models: list[str] | None = None,
    numeric_features: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    train = examples.loc[labels == "train"].copy()
    test = examples.loc[labels == "test"].copy()
    if train.empty or test.empty:
        raise ValueError(f"Sensitivity {condition} has an empty train or test partition")
    usable = train[train["condition_target"].notna()].copy()
    if usable.empty:
        raise ValueError(f"Sensitivity {condition} has no finite training targets")
    models = list(models or ["seasonal_naive", "hist_gradient_boosting"])
    numeric = list(numeric_features or DIRECT_NUMERIC_FEATURES)

    key_columns = FORECAST_KEYS + ["cutoff_timestamp", "site_id", "primary_use"]
    parts: list[pd.DataFrame] = []
    for model_name in models:
        block = test[key_columns].copy()
        block["condition"] = condition
        block["model"] = model_name
        if model_name == "seasonal_naive":
            block["prediction"] = test["direct_lag_168"].fillna(test["direct_lag_24"]).to_numpy()
        else:
            estimator = _build_direct_model(model_config, model_name, numeric)
            estimator.fit(usable[numeric + DIRECT_CATEGORICAL_FEATURES], usable["condition_target"])
            block["prediction"] = estimator.predict(test[numeric + DIRECT_CATEGORICAL_FEATURES])
        parts.append(block)
    counts = labels.value_counts().to_dict()
    return pd.concat(parts, ignore_index=True), {
        "total_example_rows": int(len(examples)),
        "train_rows": int((labels == "train").sum()),
        "train_target_rows": int(len(usable)),
        "validation_rows": int((labels == "validation").sum()),
        "test_rows": int((labels == "test").sum()),
        "excluded_boundary_straddle_rows": int(
            counts.get("excluded_boundary_straddle", 0)
        ),
        "excluded_incomplete_origin_rows": int(
            counts.get("excluded_incomplete_origin", 0)
        ),
    }


def _common_predictions(
    predictions: pd.DataFrame,
    expected_pairs: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if predictions.duplicated(PREDICTION_KEYS).any():
        raise ValueError("Sensitivity predictions contain duplicate forecast keys")
    counts = predictions.groupby(FORECAST_KEYS, observed=True).size()
    common_index = counts[counts == expected_pairs].index
    common = pd.DataFrame(common_index.tolist(), columns=FORECAST_KEYS)
    aligned = predictions.merge(common, on=FORECAST_KEYS, how="inner", validate="many_to_one")
    if aligned.empty:
        raise ValueError("No common finite sensitivity prediction keys remain")

    # A single finite key is not enough for the direct-vector protocol: retain
    # an origin/building only when all 24 horizons survived common-key
    # alignment across every condition/model pair. This matters on real data,
    # where one missing target or lag can otherwise leave a partial vector.
    origin_keys = ["forecast_origin", "building_id"]
    horizon_counts = aligned.groupby(origin_keys, observed=True)[
        "horizon_hours"
    ].nunique()
    complete_index = horizon_counts[horizon_counts == len(HORIZONS)].index
    complete = pd.DataFrame(complete_index.tolist(), columns=origin_keys)
    retained = aligned.merge(
        complete, on=origin_keys, how="inner", validate="many_to_one"
    )
    if retained.empty:
        raise ValueError("No complete common 1–24h origin vectors remain")
    return retained, {
        "candidate_common_forecast_keys": int(len(common)),
        "candidate_origin_building_groups": int(len(horizon_counts)),
        "retained_complete_origin_building_groups": int(len(complete)),
        "excluded_incomplete_origin_building_groups": int(
            len(horizon_counts) - len(complete)
        ),
    }


def _block_ids(frame: pd.DataFrame) -> pd.Series:
    week = pd.to_datetime(frame["target_timestamp"], utc=True).dt.strftime("%G-W%V")
    return frame["building_id"].astype(str) + "|" + week


def _metric_tables(
    predictions: pd.DataFrame,
    pooled_scale: float,
    building_scales: dict[str, float],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    building_parts: list[pd.DataFrame] = []
    for (condition, model), group in predictions.groupby(
        ["condition", "model"], observed=True, sort=True
    ):
        metric = calculate_metrics_blocked(
            group["target"],
            group["prediction"],
            pooled_scale,
            _block_ids(group),
            repetitions,
            seed,
        )
        building, summary = per_building_mase(
            group["target"], group["prediction"], group["building_id"], building_scales
        )
        building.insert(0, "model", str(model))
        building.insert(0, "condition", str(condition))
        building_parts.append(building)
        metric_rows.append(
            {
                "condition": condition,
                "model": model,
                **asdict(metric),
                "mase_macro_building": summary.macro_mase,
                "mase_buildings_evaluated": summary.buildings_evaluated,
                "mase_buildings_undefined": summary.buildings_undefined,
            }
        )
        for horizon, horizon_group in group.groupby(
            "horizon_hours", observed=True, sort=True
        ):
            horizon_metric = calculate_metrics_blocked(
                horizon_group["target"],
                horizon_group["prediction"],
                pooled_scale,
                _block_ids(horizon_group),
                repetitions,
                seed + int(horizon),
            )
            _, horizon_summary = per_building_mase(
                horizon_group["target"],
                horizon_group["prediction"],
                horizon_group["building_id"],
                building_scales,
            )
            horizon_rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "horizon_hours": int(horizon),
                    **asdict(horizon_metric),
                    "mase_macro_building": horizon_summary.macro_mase,
                    "mase_buildings_evaluated": horizon_summary.buildings_evaluated,
                    "mase_buildings_undefined": horizon_summary.buildings_undefined,
                }
            )
    metrics = pd.DataFrame(metric_rows).sort_values(["model", "condition"])
    horizons = pd.DataFrame(horizon_rows).sort_values(
        ["model", "condition", "horizon_hours"]
    )
    buildings = pd.concat(building_parts, ignore_index=True).sort_values(
        ["model", "condition", "building_id"]
    )
    return metrics, horizons, buildings


def _paired_metric_tables(
    predictions: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute aggregate and per-horizon paired block-bootstrap differences."""

    aggregate_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for model in sorted(predictions["model"].astype(str).unique()):
        corrupted = predictions[
            (predictions["condition"] == "corrupted")
            & (predictions["model"] == model)
        ][FORECAST_KEYS + ["prediction", "target"]]
        remediated = predictions[
            (predictions["condition"] == "remediated")
            & (predictions["model"] == model)
        ][FORECAST_KEYS + ["prediction", "target"]]
        joined = corrupted.merge(
            remediated,
            on=FORECAST_KEYS,
            suffixes=("_corrupted", "_remediated"),
            validate="one_to_one",
        ).sort_values(FORECAST_KEYS)
        if len(joined) != len(corrupted) or len(joined) != len(remediated):
            raise ValueError(f"Sensitivity paired key alignment failed for {model}")
        if not np.allclose(
            joined["target_corrupted"],
            joined["target_remediated"],
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"Sensitivity paired targets differ for {model}")
        paired = paired_absolute_error_difference_blocked(
            joined["target_corrupted"],
            joined["prediction_corrupted"],
            joined["prediction_remediated"],
            _block_ids(joined),
            repetitions,
            seed,
        )
        aggregate_rows.append({"model": model, **asdict(paired)})
        for horizon, group in joined.groupby(
            "horizon_hours", observed=True, sort=True
        ):
            horizon_paired = paired_absolute_error_difference_blocked(
                group["target_corrupted"],
                group["prediction_corrupted"],
                group["prediction_remediated"],
                _block_ids(group),
                repetitions,
                seed + int(horizon),
            )
            horizon_rows.append(
                {
                    "model": model,
                    "horizon_hours": int(horizon),
                    **asdict(horizon_paired),
                }
            )
    return (
        pd.DataFrame(aggregate_rows).sort_values("model"),
        pd.DataFrame(horizon_rows).sort_values(["model", "horizon_hours"]),
    )


def _write_report(
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    manifest: dict[str, Any],
    output: Path,
) -> None:
    shown = metrics.copy()
    for column in (
        "mae",
        "rmse",
        "wmape",
        "mase",
        "mase_macro_building",
        "r2",
        "mae_ci_low",
        "mae_ci_high",
    ):
        shown[column] = shown[column].map(
            lambda value: "NA" if pd.isna(value) else f"{value:.4f}"
        )
    headers = [str(column) for column in shown.columns]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    )
    paired_shown = paired.copy()
    for column in (
        "corrupted_mae",
        "remediated_mae",
        "mean_difference",
        "difference_ci_low",
        "difference_ci_high",
    ):
        paired_shown[column] = paired_shown[column].map(
            lambda value: "NA" if pd.isna(value) else f"{value:.4f}"
        )
    paired_headers = [str(column) for column in paired_shown.columns]
    paired_table = [
        "| " + " | ".join(paired_headers) + " |",
        "| " + " | ".join(["---"] * len(paired_headers)) + " |",
    ]
    paired_table.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in paired_shown.itertuples(index=False, name=None)
    )
    split = manifest["split"]
    report = f"""# Direct 1–24 Hour Forecast Sensitivity

This is a separate sensitivity analysis, not a replacement for the canonical
fixed-horizon 24-hour run and not a deployment claim.

## Locked design

- One forecast origin per source-local calendar day at hour `{manifest['origin_protocol']['origin_hour']}`.
- Each origin predicts the complete 1–24 hour vector; stride is 24 hours.
- Full origin blocks are partitioned by target time. Blocks crossing a split
  boundary are embargoed and excluded.
- Train end marker: `{split['train_end']}`
- Validation end marker: `{split['validation_end']}`
- Test end marker: `{split['test_end']}`
- Test targets: reference condition only.
- Condition parity at and after the fault cutoff: verified before fitting.
- Calendar features: preserved source-local wall-clock hour/day. The trailing
  `Z` comparison marker does not imply true UTC offset conversion.
- MAE intervals: whole building-week block bootstrap, streamed one repetition
  at a time; no rows-by-repetitions matrix.
- Pooled MASE-168 remains the primary sensitivity summary. Per-building and
  macro-building MASE are supplementary and use reference-training scales only.

## Complete results

{chr(10).join(table)}

## Per-horizon results

See `horizon_metrics.csv` and the deterministic chart below. Every condition,
model, and horizon from 1 through 24 is retained.

![MAE by horizon](figures/mae_by_horizon.svg)

## Paired corrupted-versus-remediated differences

`mean_difference` is remediated absolute error minus corrupted absolute error
on the identical forecast keys. Negative values favor remediation. Intervals
resample the same whole building-week blocks used for sensitivity MAE.

{chr(10).join(paired_table)}

Per-horizon paired results are retained in
`horizon_paired_differences.csv`; no horizon is selected or omitted.

## Interpretation boundary

The sensitivity isolates horizon behavior under the recorded public-data
experiment. It does not alter the canonical fixed-h=24 result, establish
operational performance, or evaluate real-time deployment.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def _prepare_empty_output(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Sensitivity output directory must be new or empty: {output}. "
            "This prevents stale files from masquerading as a completed run."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def run_sensitivity_experiment(
    reference_path: str | Path,
    corrupted_path: str | Path,
    remediated_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    producer_manifest_path: str | Path,
    fault_manifest_path: str | Path,
    weather_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_sensitivity_config(config_path)
    output = _prepare_empty_output(output_dir)
    paths = {
        "reference": Path(reference_path),
        "corrupted": Path(corrupted_path),
        "remediated": Path(remediated_path),
    }
    verified = verify_input_bundle(
        paths["reference"],
        paths["corrupted"],
        paths["remediated"],
        producer_manifest_path,
        fault_manifest_path,
    )
    frames = verified.frames
    reference = frames["reference"]
    boundary = choose_explicit_split_boundaries(
        reference, config["train_end"], config["validation_end"]
    )
    validate_exclusive_fault_boundary(verified.fault_cutoff, boundary.train_end)
    max_fault = pd.Timestamp(verified.summary["max_fault_timestamp"])
    if max_fault > boundary.train_end:
        raise ValueError("All documented faults must remain inside the training partition")
    parity = verify_post_cutoff_condition_parity(frames, verified.fault_cutoff)

    reference_parts = partition(reference, boundary)
    reference_train = reference.loc[reference_parts == "train"]
    period = int(config["seasonal_period_hours"])
    pooled_scale = seasonal_scale(reference_train, period)
    building_scales = seasonal_scales_by_building(reference_train, period)
    seed = int(config["seed"])
    from .experiment import _model_config, load_weather_bundle

    model_config = _model_config(config, seed)
    models = list(config["models"])
    numeric_features = _direct_numeric_features(bool(config["weather_features"]))
    weather: pd.DataFrame | None = None
    weather_summary: dict[str, Any] | None = None
    if bool(config["weather_features"]):
        weather, weather_summary = load_weather_bundle(
            paths["reference"], verified.producer_manifest, weather_path
        )

    example_frames: dict[str, pd.DataFrame] = {}
    prediction_parts: list[pd.DataFrame] = []
    partition_counts: dict[str, dict[str, int]] = {}
    condition_order = config.get(
        "condition_order", ["reference", "corrupted", "remediated"]
    )
    if (
        not isinstance(condition_order, list)
        or len(condition_order) != 3
        or len(set(condition_order)) != 3
        or set(condition_order) != {"reference", "corrupted", "remediated"}
    ):
        raise ValueError(
            "condition_order must contain reference, corrupted, and remediated exactly once"
        )
    for condition in condition_order:
        if condition not in frames:
            raise ValueError(f"Unknown condition in condition_order: {condition}")
        examples = build_direct_examples(
            frames[condition],
            horizons=HORIZONS,
            origin_hour=int(config["origin_hour"]),
            origin_stride_hours=int(config["origin_stride_hours"]),
            weather=weather,
        )
        labels = assign_origin_partitions(examples, boundary)
        predictions, counts = _prediction_frame(
            condition, examples, labels, model_config, models, numeric_features
        )
        example_frames[condition] = examples.assign(_partition=labels)
        prediction_parts.append(predictions)
        partition_counts[condition] = counts
    predictions = pd.concat(prediction_parts, ignore_index=True)
    reference_test = example_frames["reference"].loc[
        example_frames["reference"]["_partition"] == "test",
        FORECAST_KEYS + ["condition_target"],
    ].rename(columns={"condition_target": "target"})
    predictions = predictions.merge(
        reference_test,
        on=FORECAST_KEYS,
        how="inner",
        validate="many_to_one",
    )
    predictions = predictions[
        np.isfinite(predictions["prediction"]) & np.isfinite(predictions["target"])
    ].copy()
    produced_pairs = set(
        zip(predictions["condition"], predictions["model"], strict=True)
    )
    expected_pair_set = {
        (condition, model) for condition in condition_order for model in models
    }
    if produced_pairs != expected_pair_set:
        raise ValueError(
            "Sensitivity did not produce every configured condition/model pair"
        )
    expected_pairs = len(expected_pair_set)
    predictions, common_alignment = _common_predictions(predictions, expected_pairs)
    predictions = predictions.sort_values(PREDICTION_KEYS).reset_index(drop=True)
    common_count = int(predictions[FORECAST_KEYS].drop_duplicates().shape[0])

    repetitions = int(config["bootstrap_repetitions"])
    metrics, horizon_metrics, building_mase = _metric_tables(
        predictions,
        pooled_scale,
        building_scales,
        repetitions,
        seed,
    )
    paired, horizon_paired = _paired_metric_tables(
        predictions,
        repetitions,
        seed,
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    horizon_metrics.to_csv(output / "horizon_metrics.csv", index=False)
    building_mase.to_csv(output / "building_mase.csv", index=False)
    paired.to_csv(output / "paired_differences.csv", index=False)
    horizon_paired.to_csv(output / "horizon_paired_differences.csv", index=False)
    predictions.to_csv(
        output / "predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_kind": "direct_horizons_1_to_24_sensitivity",
        "canonical_run_unchanged": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "code_revision": _source_tree_revision(),
        "runtime_packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "python_version": sys.version,
        "platform": platform.platform(),
        "state_policy": "stateless batch run; public data and reproducibility metadata only, no personal data",
        "timestamp_semantics": {
            "storage": "UTC-marked canonical comparison timestamps",
            "source_semantics": "source-local wall-clock calendar values",
            "calendar_feature_basis": "source-local hour and day retained under the marker",
            "true_utc_instant_established": False,
        },
        "seed": seed,
        "origin_protocol": {
            "origin_hour": int(config["origin_hour"]),
            "origin_stride_hours": 24,
            "horizons_hours": list(HORIZONS),
            "cutoff_equals_forecast_origin": True,
            "full_target_vector_required": True,
        },
        "split": {
            "strategy": "target_time_full_origin_blocks_with_boundary_embargo",
            "train_end": boundary.train_end.isoformat(),
            "validation_end": boundary.validation_end.isoformat(),
            "test_end": boundary.test_end.isoformat(),
        },
        "target_source": "reference",
        "post_cutoff_condition_parity": parity,
        "fault_cutoff": verified.fault_cutoff.isoformat(),
        "seasonal_scale": pooled_scale,
        "reference_training_scales_by_building": building_scales,
        "reference_training_scale_coverage": building_scale_coverage(
            building_scales
        ),
        "metric_policy": {
            "primary": "pooled_mase_168",
            "primary_column": "mase",
            "supplementary": "per_building_and_macro_building_mase_168",
            "scale_source": "reference_training_only",
            "bootstrap": "streamed whole building-week blocks",
        },
        "paired_difference_policy": {
            "comparison": "remediated_absolute_error_minus_corrupted_absolute_error",
            "negative_favors": "remediated",
            "paired_on": FORECAST_KEYS,
            "bootstrap": "streamed whole building-week blocks",
            "reported_scopes": ["aggregate", "every_horizon_1_through_24"],
        },
        "producer_manifest": {
            "path": Path(producer_manifest_path).name,
            "sha256": verified.producer_manifest_sha256,
        },
        "fault_manifest": {
            "path": Path(fault_manifest_path).name,
            "sha256": verified.fault_manifest_sha256,
        },
        "input_verification": verified.summary,
        "input_files": {
            condition: {
                "path": path.name,
                "sha256": sha256_file(path),
                "rows": int(len(frames[condition])),
            }
            for condition, path in paths.items()
        },
        "config": {
            "path": Path(config_path).name,
            "sha256": sha256_file(config_path),
            "values": config,
        },
        "models": {
            name: (
                asdict(model_config)
                if name == "hist_gradient_boosting" and not config["weather_features"] and models == ["seasonal_naive", "hist_gradient_boosting"]
                else model_config.parameters(name)
            )
            for name in models
        },
        "model_order": models,
        "weather": weather_summary,
        "features": numeric_features + DIRECT_CATEGORICAL_FEATURES,
        "feature_count": len(numeric_features) + len(DIRECT_CATEGORICAL_FEATURES),
        "prediction_key": PREDICTION_KEYS,
        "partition_row_counts": partition_counts,
        "common_alignment": common_alignment,
        "common_forecast_key_count": common_count,
        "prediction_rows": int(len(predictions)),
    }
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    (figure_dir / "mae_by_horizon.svg").write_text(
        render_horizon_lines_svg(horizon_metrics), encoding="utf-8"
    )
    _write_report(metrics, paired, manifest, output)
    hashed_files = SENSITIVITY_RESULT_FILES - {"run_manifest.json"}
    manifest["result_files"] = {
        name: {
            "sha256": sha256_file(output / name),
            "size_bytes": (output / name).stat().st_size,
        }
        for name in sorted(hashed_files)
    }
    with (output / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    verify_sensitivity_result_directory(output)
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read sensitivity manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("Sensitivity manifest must be one JSON object")
    return value


def _assert_metric_frame_matches(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    label: str,
) -> None:
    observed = observed.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    if len(observed) != len(expected) or not observed[keys].astype(str).equals(
        expected[keys].astype(str)
    ):
        raise ValueError(f"{label} keys do not match recomputation")
    numeric = [column for column in expected.columns if column not in keys]
    for column in numeric:
        if column not in observed.columns or not np.allclose(
            observed[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"{label} {column} does not match recomputation")


def _validated_horizons(frame: pd.DataFrame, label: str) -> pd.Series:
    if "horizon_hours" not in frame.columns:
        raise ValueError(f"{label} is missing horizon_hours")
    numeric = pd.to_numeric(frame["horizon_hours"], errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
        or not np.isin(values, HORIZONS).all()
    ):
        raise ValueError(f"{label} horizon_hours must be exact integers 1 through 24")
    return numeric.astype("int64")


def verify_sensitivity_result_directory(
    result_dir: str | Path,
    input_dir: str | Path | None = None,
    expected_model_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    directory = Path(result_dir)
    missing = sorted(
        name for name in SENSITIVITY_RESULT_FILES if not (directory / name).is_file()
    )
    if missing:
        raise ValueError(f"Sensitivity result directory is missing files: {missing}")
    manifest = _load_json(directory / "run_manifest.json")
    if manifest.get("run_kind") != "direct_horizons_1_to_24_sensitivity":
        raise ValueError("Result manifest is not the direct 1–24h sensitivity")
    if manifest.get("canonical_run_unchanged") is not True:
        raise ValueError("Sensitivity manifest must preserve canonical-run status")
    semantics = manifest.get("timestamp_semantics", {})
    if semantics.get("true_utc_instant_established") is not False:
        raise ValueError("Sensitivity timestamp semantics overclaim true UTC conversion")
    if not manifest.get("post_cutoff_condition_parity", {}).get("passed"):
        raise ValueError("Sensitivity manifest lacks post-cutoff condition parity")
    paired_policy = manifest.get("paired_difference_policy", {})
    if (
        paired_policy.get("comparison")
        != "remediated_absolute_error_minus_corrupted_absolute_error"
        or paired_policy.get("paired_on") != FORECAST_KEYS
        or paired_policy.get("bootstrap") != "streamed whole building-week blocks"
    ):
        raise ValueError("Sensitivity manifest has an invalid paired-difference policy")
    if not str(manifest.get("state_policy", "")).startswith("stateless"):
        raise ValueError("Sensitivity manifest lacks the stateless policy")

    predictions = pd.read_csv(
        directory / "predictions.csv.gz",
        parse_dates=["forecast_origin", "cutoff_timestamp", "target_timestamp"],
    )
    required_columns = set(PREDICTION_KEYS) | {
        "cutoff_timestamp",
        "site_id",
        "primary_use",
        "prediction",
        "target",
    }
    if missing_columns := required_columns - set(predictions.columns):
        raise ValueError(
            f"Sensitivity predictions are missing columns: {sorted(missing_columns)}"
        )
    if predictions.duplicated(PREDICTION_KEYS).any():
        raise ValueError("Sensitivity predictions contain duplicate prediction keys")
    predictions["horizon_hours"] = _validated_horizons(
        predictions, "Sensitivity predictions"
    )
    if not (predictions["cutoff_timestamp"] == predictions["forecast_origin"]).all():
        raise ValueError("Sensitivity cutoff_timestamp must equal forecast_origin")
    if set(predictions["horizon_hours"].astype(int)) != set(HORIZONS):
        raise ValueError("Sensitivity predictions do not cover exact horizons 1 through 24")
    expected_target = predictions["forecast_origin"] + pd.to_timedelta(
        predictions["horizon_hours"], unit="h"
    )
    if not (predictions["target_timestamp"] == expected_target).all():
        raise ValueError("Sensitivity target_timestamp does not match origin plus horizon")
    horizon_sets = predictions.groupby(
        ["forecast_origin", "building_id", "condition", "model"], observed=True
    )["horizon_hours"].agg(lambda values: tuple(sorted(int(value) for value in values)))
    if not all(value == HORIZONS for value in horizon_sets):
        raise ValueError("Every sensitivity origin/pair must retain all 24 horizons")

    split = manifest.get("split", {})
    validation_end = pd.Timestamp(split.get("validation_end"))
    test_end = pd.Timestamp(split.get("test_end"))
    if validation_end.tzinfo is None:
        validation_end = validation_end.tz_localize("UTC")
    if test_end.tzinfo is None:
        test_end = test_end.tz_localize("UTC")
    if not (
        (predictions["target_timestamp"] > validation_end)
        & (predictions["target_timestamp"] <= test_end)
    ).all():
        raise ValueError("Sensitivity output contains targets outside the test partition")

    metrics = pd.read_csv(directory / "metrics.csv")
    horizon_metrics = pd.read_csv(directory / "horizon_metrics.csv")
    horizon_metrics["horizon_hours"] = _validated_horizons(
        horizon_metrics, "horizon_metrics.csv"
    )
    pairs = set(zip(metrics["condition"], metrics["model"], strict=True))
    prediction_pairs = set(
        zip(predictions["condition"], predictions["model"], strict=True)
    )
    if pairs != prediction_pairs:
        raise ValueError("Sensitivity metric and prediction pairs differ")
    manifest_pairs = {
        (condition, model)
        for condition in ("reference", "corrupted", "remediated")
        for model in expected_models(manifest, expected_model_names)
    }
    if pairs != manifest_pairs or metrics.duplicated(["condition", "model"]).any():
        raise ValueError(
            "Sensitivity results must contain exactly the expected "
            "condition/model pairs"
        )
    expected_horizon_keys = {
        (condition, model, horizon)
        for condition, model in manifest_pairs
        for horizon in HORIZONS
    }
    observed_horizon_keys = set(
        zip(
            horizon_metrics["condition"],
            horizon_metrics["model"],
            horizon_metrics["horizon_hours"].astype(int),
            strict=True,
        )
    )
    if (
        len(horizon_metrics) != len(expected_horizon_keys)
        or horizon_metrics.duplicated(
            ["condition", "model", "horizon_hours"]
        ).any()
        or observed_horizon_keys != expected_horizon_keys
    ):
        raise ValueError(
            "horizon_metrics.csv must contain exactly six pairs times horizons 1 through 24"
        )
    key_counts = predictions.groupby(FORECAST_KEYS, observed=True).size()
    if key_counts.empty or not (key_counts == len(pairs)).all():
        raise ValueError("Sensitivity pairs are not aligned on common forecast keys")
    target_counts = predictions.groupby(FORECAST_KEYS, observed=True)["target"].nunique(
        dropna=False
    )
    if not (target_counts == 1).all():
        raise ValueError("Sensitivity pairs do not share one reference-only target per key")
    common_count = int(len(key_counts))
    if int(manifest.get("common_forecast_key_count", -1)) != common_count:
        raise ValueError("Sensitivity common key count does not match manifest")
    if int(manifest.get("prediction_rows", -1)) != len(predictions):
        raise ValueError("Sensitivity prediction row count does not match manifest")

    source_binding = None
    if input_dir is not None:
        source_binding = verify_result_source_binding(
            manifest,
            predictions,
            input_dir,
            target_timestamp_column="target_timestamp",
        )

    try:
        building_scales = {
            str(name): float(value)
            for name, value in manifest["reference_training_scales_by_building"].items()
        }
    except (KeyError, AttributeError, TypeError, ValueError) as error:
        raise ValueError("Sensitivity manifest has invalid building scales") from error
    prediction_buildings = set(predictions["building_id"].astype(str).unique())
    if set(building_scales) != prediction_buildings:
        raise ValueError(
            "Sensitivity building scale map does not exactly cover prediction buildings"
        )
    if manifest.get("reference_training_scale_coverage") != building_scale_coverage(
        building_scales
    ):
        raise ValueError("Sensitivity building scale coverage is inconsistent")
    repetitions = int(manifest.get("config", {}).get("values", {}).get(
        "bootstrap_repetitions", 0
    ))
    seed = int(manifest.get("seed", -1))
    pooled_scale = float(manifest.get("seasonal_scale", float("nan")))
    expected_metrics, expected_horizons, expected_buildings = _metric_tables(
        predictions,
        pooled_scale,
        building_scales,
        repetitions,
        seed,
    )
    _assert_metric_frame_matches(
        metrics, expected_metrics, ["condition", "model"], "metrics.csv"
    )
    _assert_metric_frame_matches(
        horizon_metrics,
        expected_horizons,
        ["condition", "model", "horizon_hours"],
        "horizon_metrics.csv",
    )
    observed_buildings = pd.read_csv(directory / "building_mase.csv")
    _assert_metric_frame_matches(
        observed_buildings,
        expected_buildings,
        ["condition", "model", "building_id"],
        "building_mase.csv",
    )
    expected_paired, expected_horizon_paired = _paired_metric_tables(
        predictions,
        repetitions,
        seed,
    )
    manifest_models = expected_models(manifest, expected_model_names)
    observed_paired = pd.read_csv(directory / "paired_differences.csv")
    if (
        len(observed_paired) != len(manifest_models)
        or observed_paired.duplicated(["model"]).any()
        or set(observed_paired["model"]) != set(manifest_models)
    ):
        raise ValueError(
            "paired_differences.csv must contain exactly one aggregate row per model"
        )
    _assert_metric_frame_matches(
        observed_paired,
        expected_paired,
        ["model"],
        "paired_differences.csv",
    )
    observed_horizon_paired = pd.read_csv(
        directory / "horizon_paired_differences.csv"
    )
    observed_horizon_paired["horizon_hours"] = _validated_horizons(
        observed_horizon_paired, "horizon_paired_differences.csv"
    )
    expected_paired_horizon_keys = {
        (model, horizon) for model in manifest_models for horizon in HORIZONS
    }
    observed_paired_horizon_keys = set(
        zip(
            observed_horizon_paired["model"],
            observed_horizon_paired["horizon_hours"].astype(int),
            strict=True,
        )
    )
    if (
        len(observed_horizon_paired) != len(expected_paired_horizon_keys)
        or observed_horizon_paired.duplicated(["model", "horizon_hours"]).any()
        or observed_paired_horizon_keys != expected_paired_horizon_keys
    ):
        raise ValueError(
            "horizon_paired_differences.csv must contain exactly the expected models "
            "times horizons 1 through 24"
        )
    _assert_metric_frame_matches(
        observed_horizon_paired,
        expected_horizon_paired,
        ["model", "horizon_hours"],
        "horizon_paired_differences.csv",
    )

    expected_figure = render_horizon_lines_svg(horizon_metrics)
    if (directory / "figures/mae_by_horizon.svg").read_text(
        encoding="utf-8"
    ) != expected_figure:
        raise ValueError("Sensitivity deterministic figure content mismatch")
    result_files = manifest.get("result_files")
    expected_hashed = SENSITIVITY_RESULT_FILES - {"run_manifest.json"}
    if not isinstance(result_files, dict) or set(result_files) != expected_hashed:
        raise ValueError("Sensitivity result hash inventory is incomplete")
    for filename, metadata in result_files.items():
        path = directory / filename
        if not isinstance(metadata, dict) or metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"Sensitivity result hash mismatch for {filename}")
        try:
            expected_size = _json_integer(
                metadata["size_bytes"],
                f"sensitivity result size_bytes for {filename}",
            )
        except KeyError as error:
            raise ValueError(
                f"Invalid sensitivity result size_bytes for {filename}"
            ) from error
        if expected_size != path.stat().st_size:
            raise ValueError(f"Sensitivity result size mismatch for {filename}")

    return {
        "schema_version": "1.0",
        "checks": {
            "required_files": True,
            "unique_prediction_keys": True,
            "complete_horizon_vectors": True,
            "target_timestamp_alignment": True,
            "target_time_test_partition": True,
            "common_condition_model_alignment": True,
            "shared_reference_targets": True,
            "metrics_recomputed": True,
            "horizon_metrics_recomputed": True,
            "building_mase_recomputed": True,
            "paired_block_differences_recomputed": True,
            "horizon_paired_block_differences_recomputed": True,
            "deterministic_figure_recomputed": True,
            "result_hashes": True,
            "result_sizes": True,
            "source_binding": source_binding is not None,
            "post_cutoff_condition_parity": True,
            "stateless_policy": True,
        },
        "condition_model_pairs": len(pairs),
        "common_forecast_key_count": common_count,
        "prediction_rows": int(len(predictions)),
        "source_binding_verified": source_binding is not None,
        "source_binding": source_binding,
    }
