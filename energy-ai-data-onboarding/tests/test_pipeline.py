from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from energy_onboarding.cli import main
from energy_onboarding.contracts import OUTPUT_COLUMNS
from energy_onboarding.io_utils import sha256_file
from energy_onboarding.pipeline import (
    ReferenceQualityGateError,
    run_pipeline,
    verify_result_dir,
)


def test_fixture_pipeline_outputs_verify_and_match_contract(
    sample_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    meter, metadata, config = sample_inputs
    output = tmp_path / "result"
    result = run_pipeline(
        config, meter, metadata, output, fixture_mode=True
    )
    assert result.dataset_manifest.is_file()
    assert verify_result_dir(output) == []

    conditions = {
        name: pd.read_csv(output / f"{name}.csv.gz")
        for name in ("reference", "corrupted", "remediated")
    }
    for name, frame in conditions.items():
        assert list(frame.columns) == OUTPUT_COLUMNS
        assert set(frame["condition"]) == {name}
    reference_keys = conditions["reference"][["timestamp", "building_id"]]
    for name in ("corrupted", "remediated"):
        pd.testing.assert_frame_equal(
            reference_keys,
            conditions[name][["timestamp", "building_id"]],
        )

    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert manifest["execution_scope"] == "fixture_test_only"
    assert manifest["schema"]["columns"] == OUTPUT_COLUMNS
    assert manifest["code_revision"]["kind"] == "source_tree_sha256"
    assert len(manifest["code_revision"]["value"]) == 64
    for filename, record in manifest["files"].items():
        assert sha256_file(output / filename) == record["sha256"]
        assert not Path(record["path"]).is_absolute()

    fault_manifest = json.loads((output / "fault_manifest.json").read_text())
    assert fault_manifest["schema_version"] == "1.0"
    assert fault_manifest["fault_event_metadata_version"] == "1.0"
    assert fault_manifest["reference_sha256"] == sha256_file(output / "reference.csv.gz")
    assert fault_manifest["faults"]
    assert all(record["condition"] == "corrupted" for record in fault_manifest["faults"])
    assert fault_manifest["fault_events"]
    assert all(
        {
            "fault_id",
            "severity",
            "start_timestamp",
            "end_timestamp",
            "affected_key_count",
            "affected_keys_sha256",
            "original_value",
            "corrupted_value",
        }
        <= set(record)
        for record in fault_manifest["faults"]
    )
    summary = json.loads((output / "quality_summary.json").read_text())
    assert summary["timestamp_semantics_machine"] == {
        "basis": "source_local_wall_clock",
        "canonical_marker": "UTC",
        "offset_conversion_applied": False,
        "dst_disambiguated": False,
    }
    assert (output / "report.md").read_text().startswith("# Energy-data onboarding")
    assert (output / "data_card.md").is_file()


def test_verify_detects_tampering_and_cli_returns_nonzero(
    sample_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    meter, metadata, config = sample_inputs
    output = tmp_path / "result"
    run_pipeline(config, meter, metadata, output, fixture_mode=True)
    with (output / "report.md").open("a", encoding="utf-8") as handle:
        handle.write("tamper")
    errors = verify_result_dir(output)
    assert "hash mismatch: report.md" in errors
    assert main(["verify", "--result-dir", str(output)]) == 2


def test_schema_keys_do_not_create_conversational_or_personal_stores(
    sample_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    meter, metadata, config = sample_inputs
    output = tmp_path / "result"
    run_pipeline(config, meter, metadata, output, fixture_mode=True)
    values = [
        json.loads((output / filename).read_text())
        for filename in (
            "dataset_manifest.json",
            "fault_manifest.json",
            "quality_summary.json",
            "run_manifest.json",
        )
    ]
    banned_key_fragments = (
        "prompt",
        "chat",
        "conversation",
        "embedding",
        "vector_database",
        "user_profile",
        "personal_data",
    )

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                assert not any(fragment in key.lower() for fragment in banned_key_fragments)
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    for value in values:
        walk(value)


def test_reference_fail_gate_blocks_publication_and_cli_is_nonzero(
    sample_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    meter, metadata, config_path = sample_inputs
    meter_frame = pd.read_csv(meter)
    meter_frame.loc[5, "b1"] = None
    meter_frame.to_csv(meter, index=False)
    config = json.loads(config_path.read_text())
    config["gate"] = {
        "abort_on_reference_fail": True,
        "fail_severities": ["fail"],
        "check_thresholds": {
            "LOAD_MISSING": {"basis": "count", "fail_above": 0}
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    direct_output = tmp_path / "blocked-direct"
    with pytest.raises(ReferenceQualityGateError):
        run_pipeline(
            config_path, meter, metadata, direct_output, fixture_mode=True
        )
    blocked = json.loads((direct_output / "blocked_run.json").read_text())
    assert blocked["publication_status"] == "blocked"
    result = next(
        item
        for item in blocked["reference_gate"]["check_results"]
        if item["issue_code"] == "LOAD_MISSING"
    )
    assert result["count"] == 1
    assert result["denominator"] == 900
    assert result["threshold"]["basis"] == "count"
    assert result["threshold"]["comparison"] == "strictly_greater_than"
    assert not (direct_output / "reference.csv.gz").exists()
    assert not (direct_output / "dataset_manifest.json").exists()

    cli_output = tmp_path / "blocked-cli"
    assert main(
        [
            "run",
            "--config",
            str(config_path),
            "--electricity-csv",
            str(meter),
            "--metadata-csv",
            str(metadata),
            "--output-dir",
            str(cli_output),
            "--fixture-mode",
        ]
    ) == 2
    assert not (cli_output / "reference.csv.gz").exists()


def test_source_manifest_and_identity_fail_closed_without_condition_files(
    sample_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    meter, metadata, config_path = sample_inputs
    config = json.loads(config_path.read_text())
    config["dataset"]["source_manifest"] = str(tmp_path / "missing.json")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    missing_output = tmp_path / "missing-source"
    assert main(
        [
            "run",
            "--config",
            str(config_path),
            "--electricity-csv",
            str(meter),
            "--metadata-csv",
            str(metadata),
            "--output-dir",
            str(missing_output),
            "--fixture-mode",
        ]
    ) == 2
    assert json.loads((missing_output / "blocked_run.json").read_text())[
        "reason_code"
    ] == "SOURCE_MANIFEST_UNAVAILABLE"
    assert not (missing_output / "reference.csv.gz").exists()

    authoritative = Path(__file__).parents[1] / "provenance" / "bdg2_v1.0.json"
    config["dataset"]["source_manifest"] = str(authoritative)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    mismatch_output = tmp_path / "mismatched-source"
    assert main(
        [
            "run",
            "--config",
            str(config_path),
            "--electricity-csv",
            str(meter),
            "--metadata-csv",
            str(metadata),
            "--output-dir",
            str(mismatch_output),
        ]
    ) == 2
    assert json.loads((mismatch_output / "blocked_run.json").read_text())[
        "reason_code"
    ] == "SOURCE_IDENTITY_VERIFICATION_FAILED"
    assert not (mismatch_output / "dataset_manifest.json").exists()


def test_hard_structural_failure_blocks_even_when_config_disables_abort(
    sample_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    meter, metadata, config_path = sample_inputs
    meter_frame = pd.read_csv(meter)
    meter_frame = pd.concat([meter_frame, meter_frame.iloc[[0]]], ignore_index=True)
    meter_frame.to_csv(meter, index=False)
    config = json.loads(config_path.read_text())
    config["gate"] = {
        "abort_on_reference_fail": False,
        "fail_severities": [],
        "check_thresholds": {
            "DUPLICATE_OBSERVATION_KEY": {
                "basis": "count",
                "warn_above": 1000,
                "fail_above": 2000,
            }
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "hard-structural-block"
    with pytest.raises(ReferenceQualityGateError):
        run_pipeline(config_path, meter, metadata, output, fixture_mode=True)
    blocked = json.loads((output / "blocked_run.json").read_text())
    assert blocked["reason_code"] == "REFERENCE_HARD_STRUCTURE_FAILED"
    assert blocked["reference_gate"]["hard_structural_failures"] == [
        "DUPLICATE_OBSERVATION_KEY"
    ]
    decision = next(
        item
        for item in blocked["reference_gate"]["check_results"]
        if item["issue_code"] == "DUPLICATE_OBSERVATION_KEY"
    )
    assert decision["status"] == "fail"
    assert decision["enforcement"] == "hard_structural_count_greater_than_zero"
    assert decision["threshold"]["fail_above"] == 2000.0
    assert not (output / "reference.csv.gz").exists()
    assert not (output / "dataset_manifest.json").exists()


def test_fault_group_metadata_is_recomputed_even_if_outer_hash_is_updated(
    sample_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    meter, metadata, config = sample_inputs
    output = tmp_path / "result"
    run_pipeline(config, meter, metadata, output, fixture_mode=True)
    fault_path = output / "fault_manifest.json"
    fault_manifest = json.loads(fault_path.read_text())
    fault_manifest["faults"][0]["affected_keys_sha256"] = "0" * 64
    fault_path.write_text(
        json.dumps(fault_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dataset_path = output / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_path.read_text())
    dataset_manifest["files"]["fault_manifest.json"]["sha256"] = sha256_file(
        fault_path
    )
    dataset_path.write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    errors = verify_result_dir(output)
    assert any("invalid affected_keys_sha256" in error for error in errors)


def test_pipeline_rejects_nonempty_output_directory(
    sample_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    meter, metadata, config = sample_inputs
    output = tmp_path / "occupied"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        run_pipeline(config, meter, metadata, output, fixture_mode=True)
    assert marker.read_text() == "existing"
