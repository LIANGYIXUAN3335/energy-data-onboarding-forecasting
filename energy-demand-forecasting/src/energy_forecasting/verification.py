from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .dataset import read_condition, sha256_file
from .figures import render_metric_bars_svg
from .metrics import (
    building_scale_coverage,
    calculate_metrics,
    paired_absolute_error_difference,
    per_building_mase,
    seasonal_scale,
    seasonal_scales_by_building,
)


CONDITION_FILES = {
    "reference": "reference.csv.gz",
    "corrupted": "corrupted.csv.gz",
    "remediated": "remediated.csv.gz",
}

EXPECTED_CONDITIONS = frozenset(CONDITION_FILES)
# v2 default model set; v3 bundles declare their models in the run manifest.
EXPECTED_MODELS = frozenset({"seasonal_naive", "hist_gradient_boosting"})
EXPECTED_CONDITION_MODEL_PAIRS = frozenset(
    (condition, model)
    for condition in EXPECTED_CONDITIONS
    for model in EXPECTED_MODELS
)
KNOWN_MODELS = frozenset(
    {"seasonal_naive", "hist_gradient_boosting", "ridge", "random_forest"}
)


def expected_models(
    manifest: dict[str, Any], required: Iterable[str] | None = None
) -> frozenset[str]:
    """Models a bundle must contain.

    Legacy bundles without a ``models`` map are held to the v2 pair. Newer
    bundles must declare the same set consistently in ``models``,
    ``model_order`` and ``config.values.models``. Because the run manifest is
    not signed, a caller that knows which experiment it is checking should
    pass ``required`` (the CLI flag ``--expected-models``); the bundle is then
    rejected unless its declared set equals exactly that set.
    """

    declared = manifest.get("models")
    if not isinstance(declared, dict) or not declared:
        models = EXPECTED_MODELS
    else:
        models = frozenset(str(name) for name in declared)
        order = manifest.get("model_order")
        config_models = (manifest.get("config", {}) or {}).get("values", {}).get("models")
        for label, other in (("model_order", order), ("config.values.models", config_models)):
            if other is None:
                continue
            if not isinstance(other, list) or frozenset(map(str, other)) != models:
                raise ValueError(f"Run manifest {label} does not match its models map")
        if "seasonal_naive" not in models or len(models) < 2:
            raise ValueError("Run manifest models must include seasonal_naive and a learned model")
        if unknown := models - KNOWN_MODELS:
            raise ValueError(f"Run manifest declares unknown models: {sorted(unknown)}")
    if required is not None:
        wanted = frozenset(str(name) for name in required)
        if wanted != models:
            raise ValueError(
                f"Bundle declares models {sorted(models)} but {sorted(wanted)} were required"
            )
    return models


def expected_condition_model_pairs(
    manifest: dict[str, Any], required: Iterable[str] | None = None
) -> frozenset[tuple[str, str]]:
    return frozenset(
        (condition, model)
        for condition in EXPECTED_CONDITIONS
        for model in expected_models(manifest, required)
    )

LEGACY_REQUIRED_RESULT_FILES = {
    "metrics.csv",
    "subgroup_metrics.csv",
    "paired_differences.csv",
    "predictions.csv.gz",
    "run_manifest.json",
    "model_card.md",
    "report.md",
    "report.html",
}

REQUIRED_RESULT_FILES = LEGACY_REQUIRED_RESULT_FILES | {
    "building_mase.csv",
    "figures/mae_by_condition.svg",
    "figures/macro_building_mase.svg",
}


@dataclass(frozen=True)
class VerifiedInputs:
    frames: dict[str, pd.DataFrame]
    fault_cutoff: pd.Timestamp
    producer_manifest: dict[str, Any]
    fault_manifest: dict[str, Any]
    producer_manifest_sha256: str
    fault_manifest_sha256: str
    summary: dict[str, Any]


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} must contain one JSON object")
    return value


def _as_utc(value: Any, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a valid timestamp: {value!r}") from error
    if pd.isna(timestamp):
        raise ValueError(f"{label} is not a valid timestamp: {value!r}")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _manifest_files(manifest: dict[str, Any]) -> dict[str, Any]:
    """Read the canonical producer file map, with narrow legacy aliases."""
    for field in ("files", "outputs", "artifacts", "output_files"):
        value = manifest.get(field)
        if isinstance(value, dict):
            return value
    raise ValueError(
        "Producer manifest must contain a 'files' mapping of published filenames "
        "to sha256 metadata"
    )


def _file_entry(files: dict[str, Any], filename: str, condition: str) -> dict[str, Any]:
    direct = files.get(filename, files.get(condition))
    if isinstance(direct, str):
        return {"sha256": direct}
    if isinstance(direct, dict):
        return direct
    for value in files.values():
        if not isinstance(value, dict):
            continue
        candidate = value.get("path") or value.get("filename") or value.get("name")
        if candidate is not None and Path(str(candidate)).name == filename:
            return value
    raise ValueError(f"Producer manifest has no entry for {filename}")


def _expected_hash(entry: dict[str, Any], filename: str) -> str:
    value = entry.get("sha256", entry.get("hash"))
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Producer manifest has no valid sha256 for {filename}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"Producer manifest sha256 for {filename} is not hexadecimal") from error
    return value.lower()


def _equal_number(left: Any, right: Any) -> bool:
    left_missing = left is None or (not isinstance(left, (list, dict)) and pd.isna(left))
    right_missing = right is None or (
        not isinstance(right, (list, dict)) and pd.isna(right)
    )
    if left_missing and right_missing:
        return True
    if left_missing or right_missing:
        return False
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return left == right
    if math.isnan(left_value) and math.isnan(right_value):
        return True
    return bool(np.isclose(left_value, right_value, rtol=1e-10, atol=1e-12))


def _json_integer(value: Any, label: str) -> int:
    """Return a genuine JSON-style integer without truncating floats/bools."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _key_tuples(frame: pd.DataFrame) -> set[tuple[pd.Timestamp, str]]:
    return set(
        zip(
            pd.to_datetime(frame["timestamp"], utc=True),
            frame["building_id"].astype(str),
            strict=True,
        )
    )


def _load_faults(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    faults = manifest.get("faults", manifest.get("records"))
    if not isinstance(faults, list) or not faults:
        raise ValueError("Fault manifest must contain a non-empty 'faults' list")
    if not all(isinstance(item, dict) for item in faults):
        raise ValueError("Every fault-manifest item must be a JSON object")
    return faults


def verify_post_cutoff_condition_parity(
    frames: dict[str, pd.DataFrame],
    cutoff: pd.Timestamp,
) -> dict[str, Any]:
    """Fail unless every non-reference load equals reference after cutoff.

    The fault cutoff is the first timestamp at which the three published
    conditions must again be identical. This protects validation/test labels
    and feature histories from condition-specific post-cutoff processing.
    """

    reference = frames["reference"][["timestamp", "building_id", "load"]].copy()
    reference["timestamp"] = pd.to_datetime(reference["timestamp"], utc=True)
    checked_rows: dict[str, int] = {}
    for condition in ("corrupted", "remediated"):
        candidate = frames[condition][["timestamp", "building_id", "load"]].copy()
        candidate["timestamp"] = pd.to_datetime(candidate["timestamp"], utc=True)
        joined = reference.merge(
            candidate,
            on=["timestamp", "building_id"],
            suffixes=("_reference", f"_{condition}"),
            validate="one_to_one",
        )
        post = joined["timestamp"] >= cutoff
        left = joined.loc[post, "load_reference"].to_numpy(dtype=float)
        right = joined.loc[post, f"load_{condition}"].to_numpy(dtype=float)
        unequal = ~np.isclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True)
        if unequal.any():
            raise ValueError(
                f"{condition} differs from reference at {int(unequal.sum())} "
                "keys at or after fault_cutoff; post-cutoff parity failed"
            )
        checked_rows[condition] = int(post.sum())
    return {
        "passed": True,
        "rule": "all condition loads equal reference at timestamp >= fault_cutoff",
        "fault_cutoff": cutoff.isoformat(),
        "rows_checked": checked_rows,
    }


def verify_input_bundle(
    reference_path: str | Path,
    corrupted_path: str | Path,
    remediated_path: str | Path,
    producer_manifest_path: str | Path,
    fault_manifest_path: str | Path,
) -> VerifiedInputs:
    """Verify Repo A's complete cross-repository contract and fail closed."""
    paths = {
        "reference": Path(reference_path),
        "corrupted": Path(corrupted_path),
        "remediated": Path(remediated_path),
    }
    producer_path = Path(producer_manifest_path)
    fault_path = Path(fault_manifest_path)
    producer = _load_json(producer_path, "producer manifest")
    fault_manifest = _load_json(fault_path, "fault manifest")
    if str(producer.get("schema_version")) != "1.0":
        raise ValueError("Producer manifest schema_version must be '1.0'")
    if str(fault_manifest.get("schema_version")) != "1.0":
        raise ValueError("Fault manifest schema_version must be '1.0'")
    if fault_manifest.get("suite") != "downstream_forecasting":
        raise ValueError("Fault manifest suite must be 'downstream_forecasting'")
    files = _manifest_files(producer)

    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    reference_keys: set[tuple[pd.Timestamp, str]] | None = None
    for condition, path in paths.items():
        filename = CONDITION_FILES[condition]
        if path.name != filename:
            raise ValueError(
                f"{condition} input must be named {filename}, received {path.name}"
            )
        entry = _file_entry(files, filename, condition)
        expected_hash = _expected_hash(entry, filename)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Hash mismatch for {filename}: producer={expected_hash}, actual={actual_hash}"
            )
        frame = read_condition(path, condition)
        expected_rows = entry.get("rows", entry.get("row_count"))
        if expected_rows is not None and int(expected_rows) != len(frame):
            raise ValueError(
                f"Row-count mismatch for {filename}: producer={expected_rows}, actual={len(frame)}"
            )
        expected_columns = entry.get("columns")
        if expected_columns is not None and set(expected_columns) != set(frame.columns):
            raise ValueError(f"Column mismatch for {filename} against producer manifest")
        keys = _key_tuples(frame)
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            missing = len(reference_keys - keys)
            extra = len(keys - reference_keys)
            raise ValueError(
                f"{condition} key set differs from reference: missing={missing}, extra={extra}"
            )
        hashes[condition] = actual_hash
        frames[condition] = frame

    reference_metadata = frames["reference"].set_index(
        ["timestamp", "building_id"]
    )[["site_id", "primary_use"]].sort_index()
    for condition in ("corrupted", "remediated"):
        condition_metadata = frames[condition].set_index(
            ["timestamp", "building_id"]
        )[["site_id", "primary_use"]].sort_index()
        if not condition_metadata.equals(reference_metadata):
            raise ValueError(
                f"{condition} site_id/primary_use metadata differs from reference"
            )

    reference_hash = fault_manifest.get(
        "reference_sha256", fault_manifest.get("reference_hash")
    )
    if not isinstance(reference_hash, str):
        raise ValueError("Fault manifest must contain reference_sha256")
    if reference_hash.lower() != hashes["reference"]:
        raise ValueError("Fault manifest reference_sha256 does not match reference input")

    cutoff = _as_utc(fault_manifest.get("fault_cutoff"), "fault_cutoff")
    parity = verify_post_cutoff_condition_parity(frames, cutoff)
    faults = _load_faults(fault_manifest)
    affected_keys: set[tuple[pd.Timestamp, str]] = set()
    fault_timestamps: list[pd.Timestamp] = []
    first_original: dict[tuple[pd.Timestamp, str], Any] = {}
    final_corrupted: dict[tuple[pd.Timestamp, str], Any] = {}
    required_fault_fields = {
        "timestamp",
        "building_id",
        "original_value",
        "corrupted_value",
        "fault_type",
        "severity",
    }
    for index, fault in enumerate(faults):
        missing = required_fault_fields - set(fault)
        if missing:
            raise ValueError(
                f"Fault record {index} is missing fields: {sorted(missing)}"
            )
        condition = fault.get("condition", "corrupted")
        if condition != "corrupted":
            raise ValueError(f"Fault record {index} has unsupported condition {condition!r}")
        if not str(fault["fault_type"]).strip() or not str(fault["severity"]).strip():
            raise ValueError(f"Fault record {index} has an empty type or severity")
        timestamp = _as_utc(fault["timestamp"], f"faults[{index}].timestamp")
        if timestamp >= cutoff:
            raise ValueError(
                f"Fault record {index} at {timestamp.isoformat()} is not before "
                f"fault_cutoff {cutoff.isoformat()}"
            )
        key = (timestamp, str(fault["building_id"]))
        if reference_keys is None or key not in reference_keys:
            raise ValueError(f"Fault record {index} identifies a key absent from the inputs")
        affected_keys.add(key)
        fault_timestamps.append(timestamp)
        first_original.setdefault(key, fault["original_value"])
        final_corrupted[key] = fault["corrupted_value"]

    reference_values = frames["reference"].copy()
    corrupted_values = frames["corrupted"].copy()
    reference_values["_utc_timestamp"] = pd.to_datetime(
        reference_values["timestamp"], utc=True
    )
    corrupted_values["_utc_timestamp"] = pd.to_datetime(
        corrupted_values["timestamp"], utc=True
    )
    reference_lookup = reference_values.set_index(["_utc_timestamp", "building_id"])["load"]
    corrupted_lookup = corrupted_values.set_index(["_utc_timestamp", "building_id"])["load"]
    for key in affected_keys:
        if not _equal_number(reference_lookup.loc[key], first_original[key]):
            raise ValueError(f"Fault original_value does not match reference at key {key}")
        if not _equal_number(corrupted_lookup.loc[key], final_corrupted[key]):
            raise ValueError(f"Fault corrupted_value does not match corrupted input at key {key}")

    joined = frames["reference"][["timestamp", "building_id", "load"]].merge(
        frames["corrupted"][["timestamp", "building_id", "load"]],
        on=["timestamp", "building_id"],
        suffixes=("_reference", "_corrupted"),
        validate="one_to_one",
    )
    left = joined["load_reference"].to_numpy(dtype=float)
    right = joined["load_corrupted"].to_numpy(dtype=float)
    changed = ~np.isclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True)
    changed_keys = _key_tuples(joined.loc[changed, ["timestamp", "building_id"]])
    undocumented = changed_keys - affected_keys
    ineffective = affected_keys - changed_keys
    if undocumented:
        raise ValueError(
            f"Corrupted input contains {len(undocumented)} changed keys absent from fault manifest"
        )
    if ineffective:
        raise ValueError(
            f"Fault manifest contains {len(ineffective)} keys with no reference/corrupted change"
        )

    remediated_joined = frames["reference"][["timestamp", "building_id", "load"]].merge(
        frames["remediated"][["timestamp", "building_id", "load"]],
        on=["timestamp", "building_id"],
        suffixes=("_reference", "_remediated"),
        validate="one_to_one",
    )
    remediated_changed = ~np.isclose(
        remediated_joined["load_reference"].to_numpy(dtype=float),
        remediated_joined["load_remediated"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
    remediated_changed_keys = _key_tuples(
        remediated_joined.loc[remediated_changed, ["timestamp", "building_id"]]
    )
    unexpected_remediation = remediated_changed_keys - affected_keys

    summary = {
        "schema_version": "1.0",
        "checks": {
            "required_columns": True,
            "timestamps_parse": True,
            "unique_keys": True,
            "equal_condition_keys": True,
            "equal_condition_metadata": True,
            "producer_hashes": True,
            "producer_row_counts": True,
            "manifest_versions_and_suite": True,
            "fault_cutoff": True,
            "fault_values": True,
            "corrupted_changes_documented": True,
            "post_cutoff_condition_parity": True,
        },
        "rows_per_condition": {name: int(len(frame)) for name, frame in frames.items()},
        "unique_keys": int(len(reference_keys or set())),
        "documented_fault_records": int(len(faults)),
        "documented_affected_keys": int(len(affected_keys)),
        "remediated_changed_keys": int(len(remediated_changed_keys)),
        "remediated_changes_outside_injected_fault_set": int(
            len(unexpected_remediation)
        ),
        "fault_cutoff": cutoff.isoformat(),
        "post_cutoff_condition_parity": parity,
        "max_fault_timestamp": max(fault_timestamps).isoformat(),
        "input_sha256": hashes,
        "producer_manifest_sha256": sha256_file(producer_path),
        "fault_manifest_sha256": sha256_file(fault_path),
    }
    return VerifiedInputs(
        frames=frames,
        fault_cutoff=cutoff,
        producer_manifest=producer,
        fault_manifest=fault_manifest,
        producer_manifest_sha256=summary["producer_manifest_sha256"],
        fault_manifest_sha256=summary["fault_manifest_sha256"],
        summary=summary,
    )


def verify_input_directory(input_dir: str | Path) -> VerifiedInputs:
    directory = Path(input_dir)
    return verify_input_bundle(
        directory / CONDITION_FILES["reference"],
        directory / CONDITION_FILES["corrupted"],
        directory / CONDITION_FILES["remediated"],
        directory / "dataset_manifest.json",
        directory / "fault_manifest.json",
    )


def verify_result_source_binding(
    manifest: dict[str, Any],
    predictions: pd.DataFrame,
    input_dir: str | Path,
    *,
    target_timestamp_column: str,
) -> dict[str, Any]:
    """Bind a result bundle back to one independently verified Repo A bundle.

    Internal result hashes can show that a bundle is self-consistent, but they
    cannot establish which producer inputs supplied its targets and reference
    denominators.  This optional verification layer closes that gap without
    recording the caller's filesystem path in the returned summary.
    """

    verified = verify_input_directory(input_dir)
    producer_entry = manifest.get("producer_manifest")
    fault_entry = manifest.get("fault_manifest")
    if not isinstance(producer_entry, dict) or (
        str(producer_entry.get("sha256", "")).lower()
        != verified.producer_manifest_sha256
    ):
        raise ValueError("Result producer-manifest hash does not match input bundle")
    if not isinstance(fault_entry, dict) or (
        str(fault_entry.get("sha256", "")).lower()
        != verified.fault_manifest_sha256
    ):
        raise ValueError("Result fault-manifest hash does not match input bundle")

    input_files = manifest.get("input_files")
    if not isinstance(input_files, dict) or set(input_files) != EXPECTED_CONDITIONS:
        raise ValueError("Result input_files must cover the three conditions exactly")
    expected_hashes = verified.summary["input_sha256"]
    expected_rows = verified.summary["rows_per_condition"]
    for condition in sorted(EXPECTED_CONDITIONS):
        entry = input_files.get(condition)
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid result input_files entry for {condition}")
        if Path(str(entry.get("path", ""))).name != CONDITION_FILES[condition]:
            raise ValueError(f"Result input path is not canonical for {condition}")
        if str(entry.get("sha256", "")).lower() != expected_hashes[condition]:
            raise ValueError(f"Result input hash mismatch for {condition}")
        stored_rows = _json_integer(
            entry.get("rows"), f"Result input row count for {condition}"
        )
        if stored_rows != int(expected_rows[condition]):
            raise ValueError(f"Result input row-count mismatch for {condition}")

    if _as_utc(manifest.get("fault_cutoff"), "result fault_cutoff") != verified.fault_cutoff:
        raise ValueError("Result fault_cutoff does not match input bundle")

    embedded = manifest.get("input_verification")
    if not isinstance(embedded, dict):
        raise ValueError("Result manifest lacks its original input-verification summary")
    for field, expected in (
        ("producer_manifest_sha256", verified.producer_manifest_sha256),
        ("fault_manifest_sha256", verified.fault_manifest_sha256),
    ):
        if str(embedded.get(field, "")).lower() != expected:
            raise ValueError(f"Embedded input verification {field} does not match input bundle")
    embedded_hashes = embedded.get("input_sha256")
    embedded_rows = embedded.get("rows_per_condition")
    if not isinstance(embedded_hashes, dict) or {
        str(name): str(value).lower() for name, value in embedded_hashes.items()
    } != expected_hashes:
        raise ValueError("Embedded input verification hashes do not match input bundle")
    try:
        normalized_embedded_rows = {
            str(name): _json_integer(
                value, f"Embedded input row count for {name}"
            )
            for name, value in embedded_rows.items()
        }
    except AttributeError as error:
        raise ValueError("Embedded input verification row counts are invalid") from error
    if normalized_embedded_rows != expected_rows:
        raise ValueError("Embedded input verification row counts do not match input bundle")

    if target_timestamp_column not in predictions.columns:
        raise ValueError(
            f"Predictions lack source-binding timestamp column {target_timestamp_column}"
        )
    target_rows = predictions[
        [target_timestamp_column, "building_id", "target"]
    ].copy()
    target_rows["_source_timestamp"] = pd.to_datetime(
        target_rows[target_timestamp_column], errors="coerce", utc=True
    )
    if target_rows["_source_timestamp"].isna().any():
        raise ValueError("Predictions contain unparseable target timestamps")
    target_rows["_source_building"] = target_rows["building_id"].astype(str)
    target_rows["_stored_target"] = pd.to_numeric(
        target_rows["target"], errors="coerce"
    )
    if not np.isfinite(target_rows["_stored_target"].to_numpy(dtype=float)).all():
        raise ValueError("Predictions contain nonnumeric or nonfinite targets")
    reference_targets = verified.frames["reference"][
        ["timestamp", "building_id", "load"]
    ].copy()
    reference_targets["_source_timestamp"] = pd.to_datetime(
        reference_targets["timestamp"], utc=True
    )
    reference_targets["_source_building"] = reference_targets[
        "building_id"
    ].astype(str)
    reference_targets = reference_targets[
        ["_source_timestamp", "_source_building", "load"]
    ]
    bound_targets = target_rows.merge(
        reference_targets,
        on=["_source_timestamp", "_source_building"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing_targets = bound_targets["_merge"].ne("both")
    if missing_targets.any():
        raise ValueError(
            f"Predictions contain {int(missing_targets.sum())} targets absent from reference input"
        )
    if not np.allclose(
        bound_targets["_stored_target"].to_numpy(dtype=float),
        bound_targets["load"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError("Prediction targets do not match reference input by key")

    split = manifest.get("split")
    config_values = manifest.get("config", {}).get("values", {})
    if not isinstance(split, dict) or "train_end" not in split:
        raise ValueError("Result manifest lacks split.train_end for scale verification")
    if not isinstance(config_values, dict):
        raise ValueError("Result manifest lacks config values for scale verification")
    train_end = _as_utc(split["train_end"], "split.train_end")
    try:
        period = _json_integer(
            config_values["seasonal_period_hours"], "seasonal_period_hours"
        )
    except KeyError as error:
        raise ValueError("Invalid seasonal_period_hours in result manifest") from error
    if period <= 0:
        raise ValueError("seasonal_period_hours must be positive")
    reference = verified.frames["reference"]
    reference_train = reference.loc[
        pd.to_datetime(reference["timestamp"], utc=True) <= train_end
    ]
    expected_pooled_scale = seasonal_scale(reference_train, period)
    if not _equal_number(manifest.get("seasonal_scale"), expected_pooled_scale):
        raise ValueError("Result pooled seasonal scale does not match reference training input")
    stored_building_scales = manifest.get("reference_training_scales_by_building")
    if not isinstance(stored_building_scales, dict):
        raise ValueError("Result manifest lacks reference-training building scales")
    expected_building_scales = seasonal_scales_by_building(reference_train, period)
    if set(map(str, stored_building_scales)) != set(expected_building_scales):
        raise ValueError(
            "Result building-scale keys do not match reference training input"
        )
    for building_id, expected in expected_building_scales.items():
        if not _equal_number(stored_building_scales.get(building_id), expected):
            raise ValueError(
                "Result building seasonal scale does not match reference training input "
                f"for {building_id}"
            )

    return {
        "verified": True,
        "producer_manifest_sha256": verified.producer_manifest_sha256,
        "fault_manifest_sha256": verified.fault_manifest_sha256,
        "input_sha256": dict(expected_hashes),
        "rows_per_condition": dict(expected_rows),
        "prediction_rows_checked": int(len(bound_targets)),
        "reference_target_keys_checked": int(
            bound_targets[["_source_timestamp", "_source_building"]]
            .drop_duplicates()
            .shape[0]
        ),
        "reference_training_rows": int(len(reference_train)),
        "seasonal_period_hours": period,
        "seasonal_scales_recomputed": True,
    }


def verify_result_directory(
    result_dir: str | Path,
    input_dir: str | Path | None = None,
    expected_model_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate that a result directory is internally complete and consistent."""
    directory = Path(result_dir)
    missing_core = sorted(
        name for name in LEGACY_REQUIRED_RESULT_FILES if not (directory / name).is_file()
    )
    if missing_core:
        raise ValueError(f"Result directory is missing files: {missing_core}")
    manifest = _load_json(directory / "run_manifest.json", "run manifest")
    enhanced_bundle = "metric_policy" in manifest
    required_result_files = (
        REQUIRED_RESULT_FILES if enhanced_bundle else LEGACY_REQUIRED_RESULT_FILES
    )
    missing_files = sorted(
        name for name in required_result_files if not (directory / name).is_file()
    )
    if missing_files:
        raise ValueError(f"Result directory is missing files: {missing_files}")
    if not str(manifest.get("state_policy", "")).startswith("stateless"):
        raise ValueError("Run manifest lacks the stateless/no-memory policy")

    metrics = pd.read_csv(directory / "metrics.csv")
    metric_columns = {
        "condition",
        "model",
        "n",
        "mae",
        "rmse",
        "wmape",
        "mase",
        "r2",
        "mae_ci_low",
        "mae_ci_high",
        "training_rows",
        "data_retained",
    }
    if enhanced_bundle:
        metric_columns |= {
            "mase_macro_building",
            "mase_buildings_evaluated",
            "mase_buildings_undefined",
        }
    if missing := metric_columns - set(metrics.columns):
        raise ValueError(f"metrics.csv is missing columns: {sorted(missing)}")
    if metrics.duplicated(["condition", "model"]).any():
        raise ValueError("metrics.csv contains duplicate condition/model rows")

    predictions = pd.read_csv(directory / "predictions.csv.gz", parse_dates=["timestamp"])
    prediction_columns = {
        "timestamp",
        "building_id",
        "site_id",
        "primary_use",
        "condition",
        "model",
        "prediction",
        "target",
    }
    if missing := prediction_columns - set(predictions.columns):
        raise ValueError(f"predictions.csv.gz is missing columns: {sorted(missing)}")
    if predictions.duplicated(
        ["timestamp", "building_id", "condition", "model"]
    ).any():
        raise ValueError("predictions.csv.gz contains duplicate prediction keys")

    pairs = set(zip(metrics["condition"], metrics["model"], strict=True))
    prediction_pairs = set(
        zip(predictions["condition"], predictions["model"], strict=True)
    )
    if pairs != prediction_pairs:
        raise ValueError("Metric and prediction condition/model pairs do not match")
    manifest_models = expected_models(manifest, expected_model_names)
    if pairs != expected_condition_model_pairs(manifest, expected_model_names):
        raise ValueError(
            "Results must contain exactly the expected condition/model pairs "
            f"({len(EXPECTED_CONDITIONS) * len(manifest_models)} pairs for "
            f"{sorted(manifest_models)})"
        )
    pair_count = len(pairs)
    key_counts = predictions.groupby(["timestamp", "building_id"], observed=True).size()
    if key_counts.empty or not (key_counts == pair_count).all():
        raise ValueError("Predictions are not aligned on one common timestamp/building key set")
    target_counts = predictions.groupby(
        ["timestamp", "building_id"], observed=True
    )["target"].nunique(dropna=False)
    if not (target_counts == 1).all():
        raise ValueError("Predictions do not use one shared reference target per key")

    common_count = int(len(key_counts))
    if int(manifest.get("common_prediction_count", -1)) != common_count:
        raise ValueError("Run-manifest common_prediction_count does not match predictions")
    if int(manifest.get("prediction_rows", -1)) != len(predictions):
        raise ValueError("Run-manifest prediction_rows does not match predictions")

    source_binding = None
    if input_dir is not None:
        source_binding = verify_result_source_binding(
            manifest,
            predictions,
            input_dir,
            target_timestamp_column="timestamp",
        )

    config_values = manifest.get("config", {}).get("values", {})
    repetitions = int(config_values.get("bootstrap_repetitions", 0))
    seed = int(manifest.get("seed", -1))
    scale = float(manifest.get("seasonal_scale", float("nan")))
    building_scales: dict[str, float] = {}
    if enhanced_bundle:
        building_scales_value = manifest.get("reference_training_scales_by_building")
        if not isinstance(building_scales_value, dict) or not building_scales_value:
            raise ValueError(
                "Run manifest lacks reference_training_scales_by_building"
            )
        try:
            building_scales = {
                str(name): float(value) for name, value in building_scales_value.items()
            }
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid reference-training building scale map") from error
        prediction_buildings = set(predictions["building_id"].astype(str).unique())
        if set(building_scales) != prediction_buildings:
            raise ValueError(
                "Reference-training scale map does not exactly cover prediction buildings"
            )
        if manifest.get("reference_training_scale_coverage") != building_scale_coverage(
            building_scales
        ):
            raise ValueError("Reference-training building scale coverage is inconsistent")
        metric_policy = manifest.get("metric_policy", {})
        if (
            metric_policy.get("primary_column") != "mase"
            or metric_policy.get("primary_status") != "unchanged"
            or metric_policy.get("scale_source") != "reference_training_only"
        ):
            raise ValueError("Run manifest does not preserve the canonical MASE policy")
    metric_names = [
        "n",
        "mae",
        "rmse",
        "wmape",
        "mase",
        "r2",
        "mae_ci_low",
        "mae_ci_high",
    ]
    expected_building_parts: list[pd.DataFrame] = []
    for row in metrics.itertuples(index=False):
        group = predictions[
            (predictions["condition"] == row.condition)
            & (predictions["model"] == row.model)
        ]
        calculated = calculate_metrics(
            group["target"], group["prediction"], scale, repetitions, seed
        )
        if enhanced_bundle:
            building_table, building_summary = per_building_mase(
                group["target"],
                group["prediction"],
                group["building_id"],
                building_scales,
            )
            building_table.insert(0, "model", str(row.model))
            building_table.insert(0, "condition", str(row.condition))
            expected_building_parts.append(building_table)
            for name, expected in {
                "mase_macro_building": building_summary.macro_mase,
                "mase_buildings_evaluated": building_summary.buildings_evaluated,
                "mase_buildings_undefined": building_summary.buildings_undefined,
            }.items():
                stored = getattr(row, name)
                matches = (
                    int(stored) == int(expected)
                    if name != "mase_macro_building"
                    else bool(
                        np.isclose(
                            float(stored),
                            float(expected),
                            rtol=1e-9,
                            atol=1e-12,
                            equal_nan=True,
                        )
                    )
                )
                if not matches:
                    raise ValueError(
                        f"metrics.csv {name} mismatch for {row.condition}/{row.model}"
                    )
        for name in metric_names:
            stored = getattr(row, name)
            expected = getattr(calculated, name)
            if name == "n":
                matches = int(stored) == int(expected)
            else:
                matches = bool(
                    np.isclose(
                        float(stored),
                        float(expected),
                        rtol=1e-9,
                        atol=1e-12,
                        equal_nan=True,
                    )
                )
            if not matches:
                raise ValueError(
                    f"metrics.csv {name} mismatch for {row.condition}/{row.model}"
                )

    if enhanced_bundle:
        building_mase = pd.read_csv(directory / "building_mase.csv")
        building_columns = {
        "condition",
        "model",
        "building_id",
        "n",
        "mae",
        "reference_training_scale",
        "mase",
        }
        if missing := building_columns - set(building_mase.columns):
            raise ValueError(f"building_mase.csv is missing columns: {sorted(missing)}")
        if building_mase.duplicated(["condition", "model", "building_id"]).any():
            raise ValueError("building_mase.csv contains duplicate condition/model/building rows")
        expected_building = pd.concat(
            expected_building_parts, ignore_index=True
        ).sort_values(["model", "condition", "building_id"]).reset_index(drop=True)
        observed_building = building_mase.sort_values(
            ["model", "condition", "building_id"]
        ).reset_index(drop=True)
        if not expected_building[["condition", "model", "building_id"]].equals(
            observed_building[["condition", "model", "building_id"]].astype(
                {"condition": str, "model": str, "building_id": str}
            )
        ):
            raise ValueError("building_mase.csv keys do not match recomputed building rows")
        for column in ("n", "mae", "reference_training_scale", "mase"):
            if not np.allclose(
                expected_building[column].to_numpy(dtype=float),
                observed_building[column].to_numpy(dtype=float),
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(f"building_mase.csv {column} does not match recomputation")

    subgroup = pd.read_csv(directory / "subgroup_metrics.csv")
    expected_dimensions = {"site_id", "primary_use", "building_id", "quarter"}
    actual_dimensions = set(subgroup.get("group_dimension", pd.Series(dtype=str)))
    if not expected_dimensions.issubset(actual_dimensions):
        raise ValueError(
            "subgroup_metrics.csv lacks required dimensions: "
            f"{sorted(expected_dimensions - actual_dimensions)}"
        )
    paired = pd.read_csv(directory / "paired_differences.csv")
    paired_columns = {
        "model",
        "n",
        "corrupted_mae",
        "remediated_mae",
        "mean_difference",
        "difference_ci_low",
        "difference_ci_high",
    }
    if missing := paired_columns - set(paired.columns):
        raise ValueError(f"paired_differences.csv is missing columns: {sorted(missing)}")
    if (
        len(paired) != len(manifest_models)
        or paired.duplicated(["model"]).any()
        or set(paired["model"]) != set(manifest_models)
    ):
        raise ValueError(
            "paired_differences.csv must contain exactly one row for each expected model"
        )
    paired_metric_names = [
        "n",
        "corrupted_mae",
        "remediated_mae",
        "mean_difference",
        "difference_ci_low",
        "difference_ci_high",
    ]
    for row in paired.itertuples(index=False):
        corrupted = predictions[
            (predictions["condition"] == "corrupted")
            & (predictions["model"] == row.model)
        ][["timestamp", "building_id", "prediction", "target"]]
        remediated = predictions[
            (predictions["condition"] == "remediated")
            & (predictions["model"] == row.model)
        ][["timestamp", "building_id", "prediction", "target"]]
        joined = corrupted.merge(
            remediated,
            on=["timestamp", "building_id"],
            suffixes=("_corrupted", "_remediated"),
            validate="one_to_one",
        ).sort_values(["timestamp", "building_id"])
        if not np.allclose(
            joined["target_corrupted"], joined["target_remediated"], equal_nan=True
        ):
            raise ValueError(f"Paired targets differ for model {row.model}")
        calculated = paired_absolute_error_difference(
            joined["target_corrupted"],
            joined["prediction_corrupted"],
            joined["prediction_remediated"],
            repetitions,
            seed,
        )
        for name in paired_metric_names:
            stored = getattr(row, name)
            expected = getattr(calculated, name)
            if name == "n":
                matches = int(stored) == int(expected)
            else:
                matches = bool(
                    np.isclose(
                        float(stored),
                        float(expected),
                        rtol=1e-9,
                        atol=1e-12,
                        equal_nan=True,
                    )
                )
            if not matches:
                raise ValueError(
                    f"paired_differences.csv {name} mismatch for {row.model}"
                )

    result_files = manifest.get("result_files")
    if not isinstance(result_files, dict):
        raise ValueError("Run manifest must contain result_files hashes")
    expected_hashed_files = required_result_files - {"run_manifest.json"}
    if set(result_files) != expected_hashed_files:
        raise ValueError(
            "Run-manifest result_files does not exactly cover required outputs: "
            f"expected={sorted(expected_hashed_files)}, actual={sorted(result_files)}"
        )
    for filename, metadata in result_files.items():
        path = directory / filename
        if filename == "run_manifest.json" or not path.is_file():
            raise ValueError(f"Invalid result_files entry: {filename}")
        if not isinstance(metadata, dict) or metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"Result hash mismatch for {filename}")
        try:
            expected_size = _json_integer(
                metadata["size_bytes"], f"result size_bytes for {filename}"
            )
        except KeyError as error:
            raise ValueError(f"Invalid result size_bytes for {filename}") from error
        if expected_size != path.stat().st_size:
            raise ValueError(f"Result size mismatch for {filename}")

    if enhanced_bundle:
        expected_figures = {
            "figures/mae_by_condition.svg": render_metric_bars_svg(
                metrics,
                value_column="mae",
                title="Test MAE by condition and model",
                axis_label="MAE (reference-only test targets)",
            ),
            "figures/macro_building_mase.svg": render_metric_bars_svg(
                metrics,
                value_column="mase_macro_building",
                title="Supplementary macro building MASE",
                axis_label="Unweighted mean of finite per-building MASE",
            ),
        }
        for filename, expected in expected_figures.items():
            if (directory / filename).read_text(encoding="utf-8") != expected:
                raise ValueError(f"Deterministic figure content mismatch for {filename}")

    return {
        "schema_version": "1.0",
        "checks": {
            "required_files": True,
            "result_hashes": True,
            "result_sizes": True,
            "metric_schema": True,
            "metrics_recomputed": True,
            "building_mase_recomputed": enhanced_bundle,
            "deterministic_figures_recomputed": enhanced_bundle,
            "unique_predictions": True,
            "common_key_alignment": True,
            "shared_reference_targets": True,
            "subgroup_coverage": True,
            "paired_differences": True,
            "paired_differences_recomputed": True,
            "source_binding": source_binding is not None,
            "stateless_policy": True,
        },
        "condition_model_pairs": pair_count,
        "common_prediction_count": common_count,
        "prediction_rows": int(len(predictions)),
        "enhanced_supplementary_bundle": enhanced_bundle,
        "source_binding_verified": source_binding is not None,
        "source_binding": source_binding,
    }
