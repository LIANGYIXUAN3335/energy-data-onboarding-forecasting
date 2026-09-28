from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .dataset import (
    SplitBoundary,
    choose_explicit_split_boundaries,
    choose_split_boundaries,
    partition,
    sha256_file,
)
from .features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
    seasonal_naive_prediction,
)
from .figures import write_canonical_figures
from .metrics import (
    building_scale_coverage,
    calculate_metrics,
    paired_absolute_error_difference,
    per_building_mase,
    seasonal_scale,
    seasonal_scales_by_building,
)
from .models import ModelConfig, fit_and_predict
from .report import write_reports
from .verification import verify_input_bundle, verify_result_directory


def _source_tree_revision() -> dict[str, Any]:
    """Identify runnable source content without asserting a Git history."""

    root = Path(__file__).resolve().parents[2]
    candidates = [root / "pyproject.toml", *sorted((root / "src").rglob("*.py"))]
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in candidates
        if path.is_file()
    }
    if not files:
        raise ValueError("No package source files were found for code revision hashing")
    payload = json.dumps(
        files, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return {
        "kind": "source_tree_sha256",
        "value": hashlib.sha256(payload).hexdigest(),
        "file_count": len(files),
        "files": files,
        "git_commit": None,
        "note": "Content identity only; no Git commit or earlier creation date is asserted.",
    }


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "seed",
        "forecast_horizon_hours",
        "seasonal_period_hours",
        "max_model_iterations",
        "learning_rate",
        "max_leaf_nodes",
        "l2_regularization",
        "bootstrap_repetitions",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration is missing keys: {sorted(missing)}")
    if int(config["forecast_horizon_hours"]) != 24:
        raise ValueError("Phase 1 supports forecast_horizon_hours=24 only")
    has_dates = "train_end" in config and "validation_end" in config
    has_fractions = "train_fraction" in config and "validation_fraction" in config
    if not has_dates and not has_fractions:
        raise ValueError(
            "Configuration requires explicit train_end/validation_end or fixture-only "
            "train_fraction/validation_fraction"
        )
    return config


def _prediction_frame(
    condition: str,
    featured: pd.DataFrame,
    boundary: SplitBoundary,
    model_config: ModelConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    parts = partition(featured, boundary)
    train = featured.loc[parts == "train"].copy()
    test = featured.loc[parts == "test"].copy()
    if train.empty or test.empty:
        raise ValueError(f"{condition} has an empty train or test partition")

    keys = ["timestamp", "building_id", "site_id", "primary_use"]
    seasonal = test[keys].copy()
    seasonal["condition"] = condition
    seasonal["model"] = "seasonal_naive"
    seasonal["prediction"] = seasonal_naive_prediction(test).to_numpy()

    gradient = test[keys].copy()
    gradient["condition"] = condition
    gradient["model"] = "hist_gradient_boosting"
    gradient["prediction"] = fit_and_predict(train, test, model_config)
    return pd.concat([seasonal, gradient], ignore_index=True), {
        "total_rows": int(len(featured)),
        "train_rows": int(len(train)),
        "train_target_rows": int(train["load"].notna().sum()),
        "validation_rows": int((parts == "validation").sum()),
        "test_rows": int(len(test)),
    }


def _common_prediction_keys(predictions: pd.DataFrame, expected_pairs: int) -> pd.DataFrame:
    keys = ["timestamp", "building_id"]
    pair_keys = keys + ["condition", "model"]
    if predictions.duplicated(pair_keys).any():
        raise ValueError("Predictions contain duplicate condition/model evaluation keys")
    counts = predictions.groupby(keys, observed=True).size()
    common_index = counts[counts == expected_pairs].index
    common = pd.DataFrame(common_index.tolist(), columns=keys)
    return predictions.merge(common, on=keys, how="inner", validate="many_to_one")


def _subgroup_rows(
    predictions: pd.DataFrame,
    scale: float,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = predictions.copy()
    grouped["quarter"] = (
        grouped["timestamp"].dt.year.astype(str)
        + "Q"
        + grouped["timestamp"].dt.quarter.astype(str)
    )
    for group_column in ("site_id", "primary_use", "building_id", "quarter"):
        for keys, group in grouped.groupby(
            ["condition", "model", group_column], observed=True, dropna=False
        ):
            condition, model, group_value = keys
            metric = calculate_metrics(
                group["target"],
                group["prediction"],
                scale,
                repetitions,
                seed,
            )
            rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "group_dimension": group_column,
                    "group_value": str(group_value),
                    **asdict(metric),
                }
            )
    return rows


def _paired_rows(
    predictions: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    key_columns = ["timestamp", "building_id", "model"]
    selected = predictions[predictions["condition"].isin(["corrupted", "remediated"])]
    wide = selected.pivot(
        index=key_columns,
        columns="condition",
        values=["prediction", "target"],
    )
    required = {
        ("prediction", "corrupted"),
        ("prediction", "remediated"),
        ("target", "corrupted"),
        ("target", "remediated"),
    }
    if not required.issubset(set(wide.columns)):
        raise ValueError("Corrupted/remediated predictions cannot be paired")
    wide = wide.dropna(subset=list(required))
    for model, group in wide.groupby(level="model", observed=True):
        corrupted_target = group[("target", "corrupted")].to_numpy(dtype=float)
        remediated_target = group[("target", "remediated")].to_numpy(dtype=float)
        if not np.allclose(corrupted_target, remediated_target, equal_nan=True):
            raise ValueError(f"Paired targets differ for model {model}")
        difference = paired_absolute_error_difference(
            corrupted_target,
            group[("prediction", "corrupted")],
            group[("prediction", "remediated")],
            repetitions,
            seed,
        )
        rows.append({"model": str(model), **asdict(difference)})
    return rows


def run_experiment(
    reference_path: str | Path,
    corrupted_path: str | Path,
    remediated_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    producer_manifest_path: str | Path,
    fault_manifest_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Canonical output directory must be new or empty: {output}. "
            "This prevents stale files from masquerading as a completed run."
        )
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "reference": Path(reference_path),
        "corrupted": Path(corrupted_path),
        "remediated": Path(remediated_path),
    }
    config = load_config(config_path)
    seed = int(config["seed"])
    np.random.seed(seed)
    verified = verify_input_bundle(
        paths["reference"],
        paths["corrupted"],
        paths["remediated"],
        producer_manifest_path,
        fault_manifest_path,
    )
    frames = verified.frames
    reference = frames["reference"]
    condition_order = config.get(
        "condition_order", ["reference", "corrupted", "remediated"]
    )
    if (
        not isinstance(condition_order, list)
        or len(condition_order) != len(set(condition_order))
        or set(condition_order) != set(frames)
    ):
        raise ValueError(
            "condition_order must contain reference, corrupted, and remediated exactly once"
        )

    if "train_end" in config and "validation_end" in config:
        boundary = choose_explicit_split_boundaries(
            reference,
            config["train_end"],
            config["validation_end"],
        )
        split_strategy = "pre_specified_calendar_boundaries"
    else:
        boundary = choose_split_boundaries(
            reference,
            float(config["train_fraction"]),
            float(config["validation_fraction"]),
        )
        split_strategy = "fixture_fraction_fallback"
    max_fault_timestamp = pd.Timestamp(verified.summary["max_fault_timestamp"])
    if max_fault_timestamp > boundary.train_end:
        raise ValueError(
            "Fault manifest affects validation/test data; all downstream faults must be "
            "training-only"
        )
    reference_parts = partition(reference, boundary)
    scale = seasonal_scale(
        reference.loc[reference_parts == "train"],
        int(config["seasonal_period_hours"]),
    )
    building_scales = seasonal_scales_by_building(
        reference.loc[reference_parts == "train"],
        int(config["seasonal_period_hours"]),
    )
    model_config = ModelConfig(
        seed=seed,
        max_iter=int(config["max_model_iterations"]),
        learning_rate=float(config["learning_rate"]),
        max_leaf_nodes=int(config["max_leaf_nodes"]),
        l2_regularization=float(config["l2_regularization"]),
    )

    prediction_parts: list[pd.DataFrame] = []
    partition_counts: dict[str, dict[str, int]] = {}
    forecast_horizon = int(config["forecast_horizon_hours"])
    for condition in condition_order:
        featured = build_features(frames[condition], forecast_horizon)
        condition_predictions, condition_counts = _prediction_frame(
            condition, featured, boundary, model_config
        )
        prediction_parts.append(condition_predictions)
        partition_counts[condition] = condition_counts
    predictions = pd.concat(prediction_parts, ignore_index=True)
    expected_pairs = len(prediction_parts) * 2

    targets = reference.loc[
        reference["timestamp"] > boundary.validation_end,
        ["timestamp", "building_id", "load"],
    ].rename(columns={"load": "target"})
    predictions = predictions.merge(
        targets,
        on=["timestamp", "building_id"],
        how="inner",
        validate="many_to_one",
    )
    predictions = predictions[
        np.isfinite(predictions["prediction"]) & np.isfinite(predictions["target"])
    ].copy()
    predictions = _common_prediction_keys(predictions, expected_pairs)
    if predictions.empty:
        raise ValueError("No common finite test predictions remain after alignment")
    common_prediction_count = int(
        predictions[["timestamp", "building_id"]].drop_duplicates().shape[0]
    )

    metric_rows: list[dict[str, Any]] = []
    building_mase_parts: list[pd.DataFrame] = []
    repetitions = int(config["bootstrap_repetitions"])
    for (condition, model), group in predictions.groupby(
        ["condition", "model"], observed=True
    ):
        metric = calculate_metrics(
            group["target"], group["prediction"], scale, repetitions, seed
        )
        building_table, building_summary = per_building_mase(
            group["target"],
            group["prediction"],
            group["building_id"],
            building_scales,
        )
        building_table.insert(0, "model", str(model))
        building_table.insert(0, "condition", str(condition))
        building_mase_parts.append(building_table)
        counts = partition_counts[str(condition)]
        reference_training_rows = partition_counts["reference"]["train_rows"]
        metric_rows.append(
            {
                "condition": condition,
                "model": model,
                **asdict(metric),
                "mase_macro_building": building_summary.macro_mase,
                "mase_buildings_evaluated": building_summary.buildings_evaluated,
                "mase_buildings_undefined": building_summary.buildings_undefined,
                "training_rows": counts["train_target_rows"],
                "data_retained": (
                    counts["train_target_rows"] / reference_training_rows
                    if reference_training_rows
                    else float("nan")
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values(["model", "condition"])
    building_mase = pd.concat(building_mase_parts, ignore_index=True).sort_values(
        ["model", "condition", "building_id"]
    )
    subgroup = pd.DataFrame(
        _subgroup_rows(predictions, scale, repetitions, seed)
    ).sort_values(["group_dimension", "model", "condition", "group_value"])
    paired = pd.DataFrame(
        _paired_rows(predictions, repetitions, seed)
    ).sort_values("model")

    metrics.to_csv(output / "metrics.csv", index=False)
    building_mase.to_csv(output / "building_mase.csv", index=False)
    subgroup.to_csv(output / "subgroup_metrics.csv", index=False)
    paired.to_csv(output / "paired_differences.csv", index=False)
    predictions.to_csv(
        output / "predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )

    manifest = {
        "schema_version": "1.0",
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
        "state_policy": "stateless; no conversational state or personal data",
        "timestamp_semantics": (
            "canonical comparison timestamps; intended Panther/Eagle/Rat inputs "
            "preserve US/Eastern source-local wall-clock calendar values and do not "
            "establish true UTC offset conversion"
        ),
        "seed": seed,
        "split": {
            "strategy": split_strategy,
            "train_end": boundary.train_end.isoformat(),
            "validation_end": boundary.validation_end.isoformat(),
            "test_end": boundary.test_end.isoformat(),
        },
        "seasonal_scale": scale,
        "reference_training_scales_by_building": building_scales,
        "reference_training_scale_coverage": building_scale_coverage(
            building_scales
        ),
        "metric_policy": {
            "primary": "pooled_mase_168",
            "primary_column": "mase",
            "primary_status": "unchanged",
            "supplementary": "macro_mean_of_finite_per_building_mase_168",
            "supplementary_column": "mase_macro_building",
            "scale_source": "reference_training_only",
            "undefined_scale_policy": (
                "retain per-building NaN and exclude it from the finite macro mean"
            ),
        },
        "target_source": "reference",
        "fault_cutoff": verified.fault_cutoff.isoformat(),
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
            "seasonal_naive": {
                "fallback_order_hours": [168, 24],
            },
            "hist_gradient_boosting": asdict(model_config),
        },
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "partition_row_counts": partition_counts,
        "common_prediction_count": common_prediction_count,
        "prediction_rows": int(len(predictions)),
    }
    with (output / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    write_reports(metrics, subgroup, paired, manifest, output)
    figure_names = write_canonical_figures(metrics, output)
    result_names = [
        "metrics.csv",
        "building_mase.csv",
        "subgroup_metrics.csv",
        "paired_differences.csv",
        "predictions.csv.gz",
        "model_card.md",
        "report.md",
        "report.html",
        *figure_names,
    ]
    manifest["result_files"] = {
        name: {
            "sha256": sha256_file(output / name),
            "size_bytes": (output / name).stat().st_size,
        }
        for name in result_names
    }
    with (output / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    verify_result_directory(output)
    return manifest
