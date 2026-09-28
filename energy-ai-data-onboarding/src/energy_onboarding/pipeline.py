"""End-to-end auditable onboarding pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .checks import (
    QualityConfig,
    coverage_summary,
    detector_calibration_table,
    run_quality_checks,
    split_boundary_checks,
    validate_fault_cutoff,
)
from .contracts import (
    OUTPUT_COLUMNS,
    IssueRecord,
    condition_frame,
    issue_records_frame,
    validate_condition_frame,
)
from .downloader import SourceVerificationError, verify_file
from .faults import (
    detector_metrics,
    fault_event_summaries,
    inject_detector_validation_faults,
    inject_downstream_faults,
    validate_fault_record_metadata,
)
from .gates import apply_check_thresholds, gate_summary, hard_structural_failures
from .ingest import SelectionConfig, ingest_selected_dataset
from .io_utils import (
    canonical_json_hash,
    file_entry,
    read_json,
    runtime_versions,
    sha256_file,
    source_tree_revision,
    utc_now,
    write_csv_gzip,
    write_json,
)
from .remediation import remediate
from .reporting import write_reports
from .weather import prepare_weather, read_weather_csv, validate_weather_frame


TIMESTAMP_SEMANTICS = (
    "BDG2 raw meter timestamps are source-local wall-clock readings. For this "
    "prototype they are represented with a UTC marker for stable parsing without "
    "performing an offset conversion. The committed subset is restricted to sites "
    "with the same source timezone (US/Eastern). This must not be described as "
    "cross-timezone UTC harmonization; DST interpretation remains a limitation."
)
TIMESTAMP_SEMANTICS_MACHINE = {
    "basis": "source_local_wall_clock",
    "canonical_marker": "UTC",
    "offset_conversion_applied": False,
    "dst_disambiguated": False,
}


@dataclass(frozen=True)
class PipelineConfig:
    values: dict[str, Any]
    path: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        config_path = Path(path)
        return cls(read_json(config_path), config_path.resolve())


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    dataset_manifest: Path
    quality_summary: dict[str, Any]


class ReferenceQualityGateError(RuntimeError):
    """Raised when natural/reference observations are unsafe to publish."""


def _quality_config(values: dict[str, Any]) -> QualityConfig:
    quality = values.get("quality", {})
    quantile = quality.get("stuck_calibration_quantile")
    zero_ratio = quality.get("zero_profile_min_ratio")
    return QualityConfig(
        frequency=str(values.get("dataset", {}).get("expected_frequency", "1h")),
        long_zero_threshold=int(quality.get("long_zero_threshold", 6)),
        stuck_threshold=int(quality.get("stuck_threshold", 8)),
        outlier_mad_z=float(quality.get("outlier_mad_z", 12.0)),
        level_shift_ratio=float(quality.get("level_shift_ratio", 8.0)),
        causal_window=int(quality.get("causal_window_hours", 168)),
        rare_group_fraction=float(quality.get("rare_group_fraction", 0.01)),
        run_flagging=str(quality.get("run_flagging", "tail")),
        stuck_calibration_quantile=float(quantile) if quantile is not None else None,
        zero_profile_min_ratio=float(zero_ratio) if zero_ratio is not None else None,
        profile_weeks=int(quality.get("profile_weeks", 8)),
    )


def _remediation_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    remediation = values.get("remediation", {})
    return {
        "maximum_forward_fill_hours": int(remediation.get("maximum_forward_fill_hours", 3)),
        "repair_strategy": str(remediation.get("repair_strategy", "forward_fill")),
        "maximum_profile_fill_hours": int(remediation.get("maximum_profile_fill_hours", 48)),
        "profile_weeks": int(remediation.get("profile_weeks", 8)),
        "profile_fill_ambiguous_scale": bool(
            remediation.get("profile_fill_ambiguous_scale", False)
        ),
    }


def _selection_config(
    values: dict[str, Any], building_ids: list[str] | None = None
) -> SelectionConfig:
    selection = values.get("selection", {})
    configured_ids = building_ids if building_ids is not None else selection.get("building_ids", [])
    return SelectionConfig(
        site_ids=tuple(str(value) for value in selection.get("site_ids", [])),
        building_ids=tuple(str(value) for value in configured_ids),
        maximum_building_count=(
            int(selection["maximum_building_count"])
            if selection.get("maximum_building_count") is not None
            else None
        ),
        maximum_source_rows=(
            int(selection["maximum_source_rows"])
            if selection.get("maximum_source_rows") is not None
            else None
        ),
        full_run=bool(selection.get("full_run", False)),
        strategy=str(
            selection.get(
                "strategy", "metadata_round_robin_by_site_then_primary_use"
            )
        ),
        read_chunk_rows=int(selection.get("read_chunk_rows", 4096)),
    )


def _resolve_source_manifest(config: PipelineConfig) -> Path:
    declared = config.values.get("dataset", {}).get("source_manifest")
    if not declared:
        raise ValueError("dataset.source_manifest is required.")
    candidate = Path(str(declared))
    if candidate.is_absolute():
        return candidate
    if config.path is not None:
        repository_root = config.path.parent.parent
        resolved = repository_root / candidate
        if resolved.exists():
            return resolved
    return candidate.resolve()


def _verify_authoritative_inputs(
    source_manifest_path: Path,
    electricity_csv: Path,
    metadata_csv: Path,
    weather_csv: Path | None = None,
) -> dict[str, Any]:
    source = read_json(source_manifest_path)
    declarations = {
        str(record["name"]): record for record in source.get("files", [])
    }
    result: dict[str, Any] = {}
    roles = [
        ("electricity", electricity_csv, "electricity.csv"),
        ("metadata", metadata_csv, "metadata.csv"),
    ]
    if weather_csv is not None:
        roles.append(("weather", weather_csv, "weather.csv"))
    for role, path, expected_name in roles:
        record = declarations.get(expected_name)
        if record is None:
            raise ValueError(f"Source manifest has no {expected_name} declaration.")
        expected_size = record.get("expected_size_bytes")
        expected_sha256 = str(record.get("expected_sha256", ""))
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise SourceVerificationError(
                f"Source manifest has an invalid expected size for {expected_name}."
            )
        if len(expected_sha256) != 64:
            raise SourceVerificationError(
                f"Source manifest has an invalid SHA-256 for {expected_name}."
            )
        try:
            int(expected_sha256, 16)
        except ValueError as exc:
            raise SourceVerificationError(
                f"Source manifest has a non-hex SHA-256 for {expected_name}."
            ) from exc
        observed = verify_file(
            path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        result[role] = observed
    return result


def _canonicalize_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.duplicated(["timestamp", "building_id"]).any():
        return frame
    aggregation: dict[str, str] = {
        "load": "mean",
        "site_id": "first",
        "primary_use": "first",
    }
    for optional in ("timezone", "source_version", "_raw_timestamp"):
        if optional in frame:
            aggregation[optional] = "first"
    return (
        frame.groupby(["timestamp", "building_id"], as_index=False, dropna=False)
        .agg(aggregation)
        .sort_values(["timestamp", "building_id"], kind="stable", na_position="first")
        .reset_index(drop=True)
    )


def _enrich_remediated_for_checks(
    remediated: pd.DataFrame, reference_internal: pd.DataFrame
) -> pd.DataFrame:
    metadata_columns = ["timestamp", "building_id"]
    for column in ("timezone", "source_version"):
        if column in reference_internal:
            metadata_columns.append(column)
    metadata = reference_internal[metadata_columns].drop_duplicates(
        ["timestamp", "building_id"], keep="first"
    )
    return remediated.merge(
        metadata, on=["timestamp", "building_id"], how="left", validate="1:1"
    )


def _condition_keys(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        frame[["timestamp", "building_id"]].assign(
            timestamp=lambda values: pd.to_datetime(
                values["timestamp"], errors="coerce", utc=True
            ),
            building_id=lambda values: values["building_id"].astype(str),
        )
    )


def _write_plain_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", na_rep="")


def _prepare_empty_target(path: Path) -> None:
    """Refuse to mix a new run with any prior success or failed-run artifacts."""

    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Choose a new run directory."
        )
    path.mkdir(parents=True, exist_ok=True)


def _write_blocked_diagnostics(
    target: Path,
    *,
    created_at: str,
    reason_code: str,
    message: str,
    records: list[IssueRecord] | None = None,
) -> None:
    """Write clearly named diagnostics without creating publishable conditions."""

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": created_at,
        "publication_status": "blocked",
        "reason_code": reason_code,
        "message": message,
        "published_condition_files": [],
    }
    if records is not None:
        payload["reference_gate"] = gate_summary(records)
        write_csv_gzip(
            target / "blocked_quality_issues.csv.gz", issue_records_frame(records)
        )
    write_json(target / "blocked_run.json", payload)


def _mark_seeded_issue_samples(
    issue_frame: pd.DataFrame, faults: list[dict[str, Any]]
) -> pd.DataFrame:
    result = issue_frame.copy()
    fault_keys = {
        (str(record.get("timestamp")), str(record.get("building_id")))
        for record in faults
    }
    result["seeded_fault"] = [
        (str(timestamp), str(building)) in fault_keys
        if pd.notna(timestamp) and pd.notna(building)
        else False
        for timestamp, building in zip(result["timestamp"], result["building_id"])
    ]
    return result


def run_pipeline(
    config: PipelineConfig | str | Path,
    electricity_csv: str | Path,
    metadata_csv: str | Path,
    output_dir: str | Path,
    *,
    weather_csv: str | Path | None = None,
    fixture_mode: bool = False,
    fault_cutoff: str | None = None,
    seed: int | None = None,
    severity: str | None = None,
    building_ids: list[str] | None = None,
) -> PipelineResult:
    """Run ingestion, checks, both fault suites, remediation, and publication."""

    resolved_config = config if isinstance(config, PipelineConfig) else PipelineConfig.from_file(config)
    values = resolved_config.values
    electricity_path = Path(electricity_csv)
    metadata_path = Path(metadata_csv)
    weather_path = Path(weather_csv) if weather_csv is not None else None
    weather_values = values.get("weather", {})
    if not isinstance(weather_values, dict):
        raise ValueError("weather must be an object.")
    if bool(weather_values.get("required", False)) and weather_path is None:
        raise ValueError("This configuration requires --weather-csv (weather.required is true).")
    target = Path(output_dir)
    created_at = utc_now()
    _prepare_empty_target(target)

    try:
        source_manifest_path = _resolve_source_manifest(resolved_config)
        if not source_manifest_path.is_file():
            raise FileNotFoundError(source_manifest_path.name)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _write_blocked_diagnostics(
            target,
            created_at=created_at,
            reason_code="SOURCE_MANIFEST_UNAVAILABLE",
            message="The authoritative source manifest could not be resolved.",
        )
        raise SourceVerificationError(
            "Authoritative source manifest unavailable; publication stopped."
        ) from exc

    if fixture_mode:
        source_verification: dict[str, Any] = {
            "status": "fixture_test_only_unverified_against_bdg2_release"
        }
    else:
        try:
            source_verification = _verify_authoritative_inputs(
                source_manifest_path, electricity_path, metadata_path, weather_path
            )
        except (OSError, ValueError, SourceVerificationError, json.JSONDecodeError) as exc:
            _write_blocked_diagnostics(
                target,
                created_at=created_at,
                reason_code="SOURCE_IDENTITY_VERIFICATION_FAILED",
                message=(
                    "At least one authoritative source file did not match its "
                    "pinned size and SHA-256 declaration."
                ),
            )
            raise SourceVerificationError(
                "Authoritative source verification failed; publication stopped."
            ) from exc

    selection = _selection_config(values, building_ids)
    dataset = values.get("dataset", {})
    ingestion = ingest_selected_dataset(
        electricity_path,
        metadata_path,
        selection=selection,
        timestamp_column=dataset.get("timestamp_column"),
        source_version=str(dataset.get("version", "v1.0")),
    )
    reference_internal = ingestion.frame.copy()
    fault_values = values.get("faults", {})
    chosen_cutoff = fault_cutoff or str(fault_values.get("fault_cutoff"))
    if not chosen_cutoff or chosen_cutoff == "None":
        raise ValueError("faults.fault_cutoff or --fault-cutoff is required.")
    chosen_seed = int(seed if seed is not None else fault_values.get("seed", 20251206))
    chosen_severity = str(severity or fault_values.get("severity", "medium"))
    events_per_type = int(fault_values.get("events_per_type", 1))
    # Run-detector calibration is derived from the natural reference data only
    # and strictly before the fault cutoff; the same per-building thresholds
    # are then applied to every condition so that seeded faults can never
    # influence their own detection thresholds.
    base_quality_config = replace(_quality_config(values), calibration_end=chosen_cutoff)
    calibration_table = detector_calibration_table(
        reference_internal.assign(
            timestamp=pd.to_datetime(reference_internal["timestamp"], errors="coerce", utc=True)
        ),
        base_quality_config,
    )
    if base_quality_config.stuck_calibration_quantile is not None:
        quality_config = replace(
            base_quality_config,
            stuck_thresholds_by_building={
                str(row.building_id): int(row.stuck_threshold_effective)
                for row in calibration_table.itertuples(index=False)
            },
        )
        # Re-derive the published table with the frozen thresholds so that the
        # artifact records the policy actually applied to every condition.
        calibration_table = detector_calibration_table(
            reference_internal.assign(
                timestamp=pd.to_datetime(
                    reference_internal["timestamp"], errors="coerce", utc=True
                )
            ),
            quality_config,
        )
    else:
        quality_config = base_quality_config
    gate_values = values.get("gate", {})
    if not isinstance(gate_values, dict):
        raise ValueError("gate must be an object.")
    reference_checks = apply_check_thresholds(
        run_quality_checks(
            reference_internal, quality_config, source_condition="reference_natural"
        )
        + split_boundary_checks(
            reference_internal,
            chosen_cutoff,
            source_condition="reference_natural",
        )
        + ingestion.issues,
        gate_values,
    )
    reference_gate = gate_summary(reference_checks)
    reference_hard_failures = hard_structural_failures(reference_checks)
    fail_severities = gate_values.get("fail_severities", ["fail"])
    if not isinstance(fail_severities, list) or not set(fail_severities).issubset(
        {"pass", "warn", "fail"}
    ):
        raise ValueError("gate.fail_severities must be a list of pass/warn/fail values.")
    configured_gate_block = bool(gate_values.get("abort_on_reference_fail", True)) and (
        reference_gate["status"] in set(fail_severities)
    )
    if reference_hard_failures or configured_gate_block:
        hard_failure_codes = sorted({record.code for record in reference_hard_failures})
        _write_blocked_diagnostics(
            target,
            created_at=created_at,
            reason_code=(
                "REFERENCE_HARD_STRUCTURE_FAILED"
                if reference_hard_failures
                else "REFERENCE_QUALITY_GATE_FAILED"
            ),
            message=(
                "Natural/reference structural checks failed and cannot be "
                f"overridden ({', '.join(hard_failure_codes)})."
                if reference_hard_failures
                else "Natural/reference quality checks reached a configured "
                "blocking status; condition publication was not attempted."
            ),
            records=reference_checks,
        )
        raise ReferenceQualityGateError(
            "Natural/reference quality gate failed; publication stopped. "
            f"See {target / 'blocked_run.json'}."
        )
    if reference_internal["timestamp"].isna().any():
        raise ValueError("Reference input has unparseable timestamps; publication stopped.")
    reference_internal = _canonicalize_duplicates(reference_internal)
    reference = condition_frame(reference_internal, "reference")
    contract_errors = validate_condition_frame(reference, expected_condition="reference")
    if contract_errors:
        raise ValueError("Reference output contract failed: " + "; ".join(contract_errors))

    detector_suite = inject_detector_validation_faults(
        reference_internal,
        cutoff=chosen_cutoff,
        seed=chosen_seed,
        severity=chosen_severity,
        events_per_type=events_per_type,
    )
    detector_checks = apply_check_thresholds(
        run_quality_checks(
            detector_suite.frame,
            quality_config,
            source_condition="detector_validation_mixed",
        ),
        gate_values,
    )
    detector_table = detector_metrics(
        detector_suite.frame, detector_suite.faults, quality=quality_config
    )

    downstream_suite = inject_downstream_faults(
        reference_internal,
        cutoff=chosen_cutoff,
        seed=chosen_seed,
        severity=chosen_severity,
        events_per_type=events_per_type,
    )
    corrupted_internal = downstream_suite.frame
    corrupted = condition_frame(corrupted_internal, "corrupted")
    corrupted_checks = apply_check_thresholds(
        run_quality_checks(
            corrupted_internal, quality_config, source_condition="corrupted_mixed"
        ),
        gate_values,
    )
    remediation_kwargs = _remediation_kwargs(values)
    remediation_result = remediate(
        corrupted_internal,
        quality=quality_config,
        repair_before=chosen_cutoff,
        **remediation_kwargs,
    )
    remediated = remediation_result.frame
    remediated_checks = apply_check_thresholds(
        run_quality_checks(
            _enrich_remediated_for_checks(remediated, reference_internal),
            quality_config,
            source_condition="remediated",
        ),
        gate_values,
    )

    expected_keys = _condition_keys(reference)
    for name, frame in (("corrupted", corrupted), ("remediated", remediated)):
        errors = validate_condition_frame(
            frame, expected_condition=name  # type: ignore[arg-type]
        )
        if errors:
            raise ValueError(f"{name} output contract failed: {'; '.join(errors)}")
        if not expected_keys.equals(_condition_keys(frame)):
            raise AssertionError(f"{name} does not preserve the reference key set.")
    cutoff_violations = validate_fault_cutoff(downstream_suite.faults, chosen_cutoff)
    if cutoff_violations:
        raise AssertionError(f"Fault cutoff violations: {cutoff_violations[:3]}")

    # The evaluation interval is identical in all three conditions.
    boundary = pd.Timestamp(chosen_cutoff)
    boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
    post_cutoff = pd.to_datetime(reference["timestamp"], utc=True).ge(boundary)
    for frame in (corrupted, remediated):
        reference_values = reference.loc[post_cutoff, "load"].reset_index(drop=True)
        condition_values = frame.loc[post_cutoff, "load"].reset_index(drop=True)
        if not np.array_equal(
            reference_values.to_numpy(), condition_values.to_numpy(), equal_nan=True
        ):
            raise AssertionError("A condition altered load values at/after fault cutoff.")

    condition_frames = {
        "reference.csv.gz": reference,
        "corrupted.csv.gz": corrupted,
        "remediated.csv.gz": remediated,
    }
    for filename, frame in condition_frames.items():
        write_csv_gzip(target / filename, frame)

    weather_summary: dict[str, Any] | None = None
    weather_frame: pd.DataFrame | None = None
    if weather_path is not None:
        reference_timestamps = pd.to_datetime(reference["timestamp"], utc=True)
        weather_frame, weather_summary = prepare_weather(
            read_weather_csv(weather_path),
            site_ids=ingestion.selected_sites,
            grid_start=reference_timestamps.min(),
            grid_end=reference_timestamps.max(),
            frequency=quality_config.frequency,
        )
        weather_errors = validate_weather_frame(weather_frame)
        if weather_errors:
            raise ValueError("Weather output contract failed: " + "; ".join(weather_errors))
        write_csv_gzip(target / "weather.csv.gz", weather_frame)
    _write_plain_csv(target / "detector_calibration.csv", calibration_table)

    reference_hash = sha256_file(target / "reference.csv.gz")
    fault_manifest = {
        "schema_version": "1.0",
        "fault_event_metadata_version": "1.0",
        "suite": "downstream_forecasting",
        "seed": chosen_seed,
        "severity_level": chosen_severity,
        "fault_cutoff": pd.Timestamp(chosen_cutoff).isoformat(),
        "timestamp_semantics_machine": TIMESTAMP_SEMANTICS_MACHINE,
        "reference_sha256": reference_hash,
        "key_policy": "value_only_faults; timestamp/building_id keys unchanged",
        "test_target_policy": "at/after cutoff values equal reference; Repo B uses reference targets",
        "faults": downstream_suite.faults,
        "fault_events": fault_event_summaries(downstream_suite.faults),
        "detector_validation_faults": detector_suite.faults,
        "detector_validation_fault_events": fault_event_summaries(
            detector_suite.faults
        ),
    }
    write_json(target / "fault_manifest.json", fault_manifest)

    issue_parts = [
        issue_records_frame(reference_checks),
        issue_records_frame(corrupted_checks),
        issue_records_frame(remediated_checks),
        issue_records_frame(detector_checks),
    ]
    quality_issues = pd.DataFrame(
        [row for part in issue_parts for row in part.to_dict(orient="records")],
        columns=issue_parts[0].columns,
    )
    quality_issues = _mark_seeded_issue_samples(
        quality_issues, downstream_suite.faults + detector_suite.faults
    )
    write_csv_gzip(target / "quality_issues.csv.gz", quality_issues)
    write_csv_gzip(target / "remediation_log.csv.gz", remediation_result.log)
    write_csv_gzip(target / "quarantine.csv.gz", remediation_result.quarantine)
    _write_plain_csv(target / "detector_metrics.csv", detector_table)

    repaired_count = int((remediation_result.log["status"] == "repaired").sum())
    quarantined_actions = int(
        (remediation_result.log["status"] == "quarantined").sum()
    )
    coverage = coverage_summary(reference_internal)
    detector_policy = {
        "run_flagging": quality_config.run_flagging,
        "stuck_threshold_configured": int(quality_config.stuck_threshold),
        "stuck_calibration_quantile": quality_config.stuck_calibration_quantile,
        "long_zero_threshold": int(quality_config.long_zero_threshold),
        "zero_profile_min_ratio": quality_config.zero_profile_min_ratio,
        "profile_weeks": int(quality_config.profile_weeks),
        "calibration_source": "reference condition, strictly before the fault cutoff",
        "calibration_artifact": "detector_calibration.csv",
    }
    remediation_policy = {
        **remediation_kwargs,
        "future_values_used": False,
        "backward_fill_or_interpolation": False,
        "unit_scale_division": False,
    }
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": created_at,
        "execution_scope": "fixture_test_only" if fixture_mode else "verified_bdg2_subset",
        "detector_policy": detector_policy,
        "remediation_policy": remediation_policy,
        "fault_events_per_type": events_per_type,
        "weather": weather_summary,
        "dataset": {
            "title": dataset.get("title", "Building Data Genome Project 2"),
            "version": dataset.get("version", "v1.0"),
            "doi": dataset.get("doi", "10.5281/zenodo.3887306"),
        },
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "timestamp_semantics_machine": TIMESTAMP_SEMANTICS_MACHINE,
        "fault_cutoff": pd.Timestamp(chosen_cutoff).isoformat(),
        "coverage": coverage,
        "quality_gates": {
            "reference": reference_gate,
            "corrupted": gate_summary(corrupted_checks),
            "remediated": gate_summary(remediated_checks),
            "detector_validation": gate_summary(detector_checks),
        },
        "downstream_faults": {"fault_records": len(downstream_suite.faults)},
        "detector_validation": {"fault_records": len(detector_suite.faults)},
        "remediation": {
            "log_rows": len(remediation_result.log),
            "repaired_rows": repaired_count,
            "quarantined_rows": quarantined_actions,
        },
        "quarantine": {"rows": len(remediation_result.quarantine)},
        "privacy": {
            "mode": "stateless",
            "data_scope": "public building-energy observations and reproducibility metadata only",
            "external_transmission": False,
            "analytics": False,
        },
    }
    write_json(target / "quality_summary.json", summary)

    run_manifest = {
        "schema_version": "1.0",
        "created_at_utc": created_at,
        "execution_scope": summary["execution_scope"],
        "config_sha256": canonical_json_hash(values),
        "seed": chosen_seed,
        "fault_cutoff": pd.Timestamp(chosen_cutoff).isoformat(),
        "fault_event_metadata_version": "1.0",
        "timestamp_semantics_machine": TIMESTAMP_SEMANTICS_MACHINE,
        "selected_buildings": ingestion.selected_buildings,
        "selected_sites": ingestion.selected_sites,
        "runtime": runtime_versions(),
        "code_revision": source_tree_revision(Path(__file__).resolve().parents[2]),
        "state_policy": "stateless batch run; public data and reproducibility metadata only, no personal data",
    }
    write_json(target / "run_manifest.json", run_manifest)
    write_reports(target, summary)

    source_manifest_hash = sha256_file(source_manifest_path)
    input_files = {
        "electricity": {
            "path": electricity_path.name,
            "sha256": sha256_file(electricity_path),
            "size_bytes": electricity_path.stat().st_size,
        },
        "metadata": {
            "path": metadata_path.name,
            "sha256": sha256_file(metadata_path),
            "size_bytes": metadata_path.stat().st_size,
        },
    }
    if weather_path is not None:
        input_files["weather"] = {
            "path": weather_path.name,
            "sha256": sha256_file(weather_path),
            "size_bytes": weather_path.stat().st_size,
        }
    tabular_rows: dict[str, tuple[int, list[str]]] = {
        **{
            name: (len(frame), list(frame.columns))
            for name, frame in condition_frames.items()
        },
        "quality_issues.csv.gz": (len(quality_issues), list(quality_issues.columns)),
        "remediation_log.csv.gz": (
            len(remediation_result.log),
            list(remediation_result.log.columns),
        ),
        "quarantine.csv.gz": (
            len(remediation_result.quarantine),
            list(remediation_result.quarantine.columns),
        ),
        "detector_metrics.csv": (len(detector_table), list(detector_table.columns)),
        "detector_calibration.csv": (
            len(calibration_table),
            list(calibration_table.columns),
        ),
    }
    if weather_frame is not None:
        tabular_rows["weather.csv.gz"] = (len(weather_frame), list(weather_frame.columns))
    artifact_names = [
        *tabular_rows,
        "fault_manifest.json",
        "quality_summary.json",
        "run_manifest.json",
        "data_card.md",
        "report.md",
        "report.html",
    ]
    files: dict[str, Any] = {}
    for filename in artifact_names:
        row_info = tabular_rows.get(filename)
        files[filename] = file_entry(
            target / filename,
            relative_to=target,
            rows=row_info[0] if row_info else None,
            columns=row_info[1] if row_info else None,
        )

    check_counts = {
        condition: {
            severity: int(
                sum(record.status == severity for record in records)
            )
            for severity in ("pass", "warn", "fail")
        }
        for condition, records in (
            ("reference", reference_checks),
            ("corrupted", corrupted_checks),
            ("remediated", remediated_checks),
            ("detector_validation", detector_checks),
        )
    }
    dataset_manifest = {
        "schema_version": "1.0",
        "created_at_utc": created_at,
        "execution_scope": summary["execution_scope"],
        "source_manifest": source_manifest_path.name,
        "source_manifest_hash": source_manifest_hash,
        "source_verification": source_verification,
        "input_files": input_files,
        "config_hash": canonical_json_hash(values),
        "dataset": summary["dataset"],
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "timestamp_semantics_machine": TIMESTAMP_SEMANTICS_MACHINE,
        "selection": {
            "strategy": selection.strategy,
            "selected_sites": ingestion.selected_sites,
            "selected_buildings": ingestion.selected_buildings,
            "source_rows_read": ingestion.source_rows_read,
            "exclusions": ingestion.exclusions,
        },
        "dataset_summary": coverage,
        "schema": {
            "columns": OUTPUT_COLUMNS,
            "key": ["timestamp", "building_id"],
            "frequency": quality_config.frequency,
            "unit": "as supplied by BDG2; no unit conversion performed",
            "null_policy": "preserve key with null and log/quarantine when unresolved",
        },
        "check_counts": check_counts,
        "detector_policy": detector_policy,
        "remediation_policy": remediation_policy,
        "fault_events_per_type": events_per_type,
        "weather": weather_summary,
        "remediation_counts": summary["remediation"],
        "quarantine_counts": summary["quarantine"],
        "fault_cutoff": pd.Timestamp(chosen_cutoff).isoformat(),
        "fault_event_metadata_version": "1.0",
        "seed": chosen_seed,
        "runtime": runtime_versions(),
        "code_revision": source_tree_revision(Path(__file__).resolve().parents[2]),
        "files": files,
        "outputs": files,
        "state_policy": {
            "mode": "stateless",
            "data_scope": "public energy data plus reproducibility artifacts",
            "external_transmission": False,
            "analytics": False,
        },
    }
    manifest_path = target / "dataset_manifest.json"
    write_json(manifest_path, dataset_manifest)
    return PipelineResult(target, manifest_path, summary)


def verify_result_dir(result_dir: str | Path) -> list[str]:
    """Fail-closed verification of a completed producer directory."""

    directory = Path(result_dir)
    errors: list[str] = []
    required = {
        "reference.csv.gz",
        "corrupted.csv.gz",
        "remediated.csv.gz",
        "fault_manifest.json",
        "dataset_manifest.json",
        "quality_summary.json",
        "quality_issues.csv.gz",
        "remediation_log.csv.gz",
        "quarantine.csv.gz",
        "detector_metrics.csv",
        "data_card.md",
        "report.md",
        "report.html",
    }
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        return [f"missing required artifacts: {missing}"]
    try:
        manifest = read_json(directory / "dataset_manifest.json")
        if manifest.get("execution_scope") != "fixture_test_only":
            source_verification = manifest.get("source_verification")
            if not isinstance(source_verification, dict):
                errors.append("dataset_manifest.source_verification must be an object")
            else:
                for role in ("electricity", "metadata"):
                    observed = source_verification.get(role)
                    if not isinstance(observed, dict):
                        errors.append(f"source verification missing: {role}")
                        continue
                    if observed.get("verification_status") != "verified":
                        errors.append(f"source verification not verified: {role}")
                    digest = str(observed.get("observed_sha256", ""))
                    if len(digest) != 64:
                        errors.append(f"source verification SHA-256 invalid: {role}")
                    observed_size = observed.get("observed_size_bytes")
                    if not isinstance(observed_size, int) or observed_size <= 0:
                        errors.append(f"source verification size invalid: {role}")
        files = manifest.get("files")
        if not isinstance(files, dict):
            return ["dataset_manifest.files must be an object"]
        for filename, record in files.items():
            path = directory / filename
            if not path.is_file():
                errors.append(f"manifest file missing: {filename}")
                continue
            observed = sha256_file(path)
            if observed != record.get("sha256"):
                errors.append(f"hash mismatch: {filename}")

        conditions: dict[str, pd.DataFrame] = {}
        for condition in ("reference", "corrupted", "remediated"):
            frame = pd.read_csv(directory / f"{condition}.csv.gz")
            condition_errors = validate_condition_frame(
                frame, expected_condition=condition  # type: ignore[arg-type]
            )
            errors.extend(f"{condition}: {message}" for message in condition_errors)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
            frame["building_id"] = frame["building_id"].astype(str)
            frame["load"] = pd.to_numeric(frame["load"], errors="coerce")
            conditions[condition] = frame
        keys = _condition_keys(conditions["reference"])
        for condition in ("corrupted", "remediated"):
            if not keys.equals(_condition_keys(conditions[condition])):
                errors.append(f"{condition}: key set/order differs from reference")

        if manifest.get("weather") is not None:
            if "weather.csv.gz" not in files:
                errors.append("manifest declares weather but weather.csv.gz is not registered")
            elif (directory / "weather.csv.gz").is_file():
                weather = pd.read_csv(directory / "weather.csv.gz")
                errors.extend(f"weather: {message}" for message in validate_weather_frame(weather))
                weather_sites = set(weather["site_id"].astype(str))
                reference_sites = set(conditions["reference"]["site_id"].astype(str))
                if not reference_sites.issubset(weather_sites):
                    errors.append("weather: reference sites are not all covered")
        if "detector_calibration.csv" in files:
            calibration = pd.read_csv(directory / "detector_calibration.csv")
            calibrated = set(calibration["building_id"].astype(str))
            reference_buildings = set(conditions["reference"]["building_id"].astype(str))
            if calibrated != reference_buildings:
                errors.append("detector_calibration.csv does not cover the reference buildings exactly")

        fault_manifest = read_json(directory / "fault_manifest.json")
        if fault_manifest.get("reference_sha256") != sha256_file(
            directory / "reference.csv.gz"
        ):
            errors.append("fault_manifest reference_sha256 mismatch")
        faults = fault_manifest.get("faults")
        if not isinstance(faults, list):
            errors.append("fault_manifest.faults must be a list")
        else:
            errors.extend(
                f"fault cutoff violation: {value}"
                for value in validate_fault_cutoff(
                    faults, str(fault_manifest.get("fault_cutoff"))
                )
            )
            if fault_manifest.get("fault_event_metadata_version") == "1.0":
                errors.extend(validate_fault_record_metadata(faults))
                if fault_manifest.get("fault_events") != fault_event_summaries(faults):
                    errors.append("fault_manifest.fault_events does not match fault records")
        detector_faults = fault_manifest.get("detector_validation_faults", [])
        if fault_manifest.get("fault_event_metadata_version") == "1.0":
            if not isinstance(detector_faults, list):
                errors.append("fault_manifest.detector_validation_faults must be a list")
            else:
                errors.extend(validate_fault_record_metadata(detector_faults))
                if fault_manifest.get(
                    "detector_validation_fault_events"
                ) != fault_event_summaries(detector_faults):
                    errors.append(
                        "fault_manifest.detector_validation_fault_events does not "
                        "match fault records"
                    )
        cutoff = pd.to_datetime(fault_manifest.get("fault_cutoff"), utc=True)
        post_cutoff = conditions["reference"]["timestamp"].ge(cutoff)
        for condition in ("corrupted", "remediated"):
            if not np.array_equal(
                conditions["reference"].loc[post_cutoff, "load"].to_numpy(),
                conditions[condition].loc[post_cutoff, "load"].to_numpy(),
                equal_nan=True,
            ):
                errors.append(f"{condition}: values changed at/after fault cutoff")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"verification exception: {exc}")
    return errors
