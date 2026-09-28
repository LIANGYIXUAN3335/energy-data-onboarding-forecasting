from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from energy_forecasting.cli import main
from energy_forecasting.dataset import sha256_file
from energy_forecasting.experiment import run_experiment
from energy_forecasting.verification import (
    verify_input_directory,
    verify_result_directory,
)


def _run(input_bundle: dict[str, Path], output: Path) -> None:
    run_experiment(
        input_bundle["reference"],
        input_bundle["corrupted"],
        input_bundle["remediated"],
        input_bundle["config"],
        output,
        producer_manifest_path=input_bundle["producer_manifest"],
        fault_manifest_path=input_bundle["fault_manifest"],
    )


def test_verify_input_and_result_cli(
    input_bundle: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    verified = verify_input_directory(input_bundle["input_dir"])
    assert verified.summary["checks"]["equal_condition_keys"] is True
    assert verified.summary["documented_affected_keys"] > 0
    assert main(["verify-inputs", "--input-dir", str(input_bundle["input_dir"])]) == 0
    assert '"producer_hashes": true' in capsys.readouterr().out

    output = tmp_path / "results"
    _run(input_bundle, output)
    unbound_summary = verify_result_directory(output)
    assert unbound_summary["source_binding_verified"] is False
    summary = verify_result_directory(output, input_dir=input_bundle["input_dir"])
    assert summary["checks"]["common_key_alignment"] is True
    assert summary["source_binding_verified"] is True
    assert main(
        [
            "verify-results",
            "--result-dir",
            str(output),
            "--input-dir",
            str(input_bundle["input_dir"]),
        ]
    ) == 0
    output_text = capsys.readouterr().out
    assert '"paired_differences": true' in output_text
    assert '"source_binding_verified": true' in output_text


def test_producer_hash_mismatch_fails_closed(input_bundle: dict[str, Path]) -> None:
    corrupted = pd.read_csv(input_bundle["corrupted"])
    corrupted.loc[0, "load"] = corrupted.loc[0, "load"] + 1
    corrupted.to_csv(input_bundle["corrupted"], index=False, compression="gzip")
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify_input_directory(input_bundle["input_dir"])


def test_equal_key_contract_fails_after_valid_hash_update(
    input_bundle: dict[str, Path],
) -> None:
    remediated = pd.read_csv(input_bundle["remediated"])
    remediated = remediated.iloc[:-1]
    remediated.to_csv(input_bundle["remediated"], index=False, compression="gzip")
    manifest = json.loads(input_bundle["producer_manifest"].read_text(encoding="utf-8"))
    entry = manifest["files"]["remediated.csv.gz"]
    entry["sha256"] = sha256_file(input_bundle["remediated"])
    entry["rows"] = len(remediated)
    input_bundle["producer_manifest"].write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="key set differs"):
        verify_input_directory(input_bundle["input_dir"])


def test_condition_metadata_change_fails_after_valid_hash_update(
    input_bundle: dict[str, Path],
) -> None:
    remediated = pd.read_csv(input_bundle["remediated"])
    remediated.loc[0, "site_id"] = "unexpected_site"
    remediated.to_csv(input_bundle["remediated"], index=False, compression="gzip")
    manifest = json.loads(input_bundle["producer_manifest"].read_text(encoding="utf-8"))
    entry = manifest["files"]["remediated.csv.gz"]
    entry["sha256"] = sha256_file(input_bundle["remediated"])
    input_bundle["producer_manifest"].write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="metadata differs from reference"):
        verify_input_directory(input_bundle["input_dir"])


def test_fault_at_declared_cutoff_is_rejected(input_bundle: dict[str, Path]) -> None:
    manifest = json.loads(input_bundle["fault_manifest"].read_text(encoding="utf-8"))
    manifest["faults"][0]["timestamp"] = manifest["fault_cutoff"]
    input_bundle["fault_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="is not before fault_cutoff"):
        verify_input_directory(input_bundle["input_dir"])


def test_post_cutoff_condition_change_fails_after_hash_update(
    input_bundle: dict[str, Path],
) -> None:
    corrupted = pd.read_csv(input_bundle["corrupted"])
    fault_manifest = json.loads(
        input_bundle["fault_manifest"].read_text(encoding="utf-8")
    )
    cutoff = pd.Timestamp(fault_manifest["fault_cutoff"])
    timestamps = pd.to_datetime(corrupted["timestamp"])
    index = corrupted.index[timestamps >= cutoff][0]
    corrupted.loc[index, "load"] += 1.0
    corrupted.to_csv(input_bundle["corrupted"], index=False, compression="gzip")
    producer = json.loads(
        input_bundle["producer_manifest"].read_text(encoding="utf-8")
    )
    producer["files"]["corrupted.csv.gz"]["sha256"] = sha256_file(
        input_bundle["corrupted"]
    )
    input_bundle["producer_manifest"].write_text(
        json.dumps(producer), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="post-cutoff parity"):
        verify_input_directory(input_bundle["input_dir"])


def test_fault_outside_experiment_training_partition_is_rejected(
    input_bundle: dict[str, Path], tmp_path: Path
) -> None:
    config = json.loads(input_bundle["config"].read_text(encoding="utf-8"))
    config["train_fraction"] = 0.2
    config["validation_fraction"] = 0.2
    input_bundle["config"].write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="training-only"):
        _run(input_bundle, tmp_path / "results")


def test_result_hash_tamper_is_rejected(
    input_bundle: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "results"
    _run(input_bundle, output)
    metrics = pd.read_csv(output / "metrics.csv")
    metrics.loc[0, "mae"] = metrics.loc[0, "mae"] + 0.5
    metrics.to_csv(output / "metrics.csv", index=False)
    with pytest.raises(ValueError, match="mismatch"):
        verify_result_directory(output)


def test_result_contract_and_source_binding_tamper_are_rejected(
    input_bundle: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "results"
    _run(input_bundle, output)
    manifest_path = output / "run_manifest.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tampered = deepcopy(original_manifest)
    tampered["producer_manifest"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="producer-manifest hash"):
        verify_result_directory(output, input_dir=input_bundle["input_dir"])

    tampered = deepcopy(original_manifest)
    tampered["input_files"]["reference"]["rows"] += 1
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="row-count mismatch"):
        verify_result_directory(output, input_dir=input_bundle["input_dir"])

    tampered = deepcopy(original_manifest)
    tampered["seasonal_scale"] = 123.456
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="pooled seasonal scale"):
        verify_result_directory(output, input_dir=input_bundle["input_dir"])

    tampered = deepcopy(original_manifest)
    tampered["result_files"]["metrics.csv"]["size_bytes"] += 1
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_result_directory(output)


def test_source_binding_checks_every_prediction_target(
    input_bundle: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "results"
    _run(input_bundle, output)
    predictions = pd.read_csv(output / "predictions.csv.gz", parse_dates=["timestamp"])
    first = predictions.iloc[0]
    same_source_key = (
        predictions["timestamp"].eq(first["timestamp"])
        & predictions["building_id"].eq(first["building_id"])
    )
    assert int(same_source_key.sum()) == 6
    predictions.loc[same_source_key, "target"] += 1.0
    predictions.to_csv(
        output / "predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    with pytest.raises(ValueError, match="do not match reference input"):
        verify_result_directory(output, input_dir=input_bundle["input_dir"])


def test_exact_six_condition_model_pairs_are_required(
    input_bundle: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "results"
    _run(input_bundle, output)
    metrics = pd.read_csv(output / "metrics.csv")
    predictions = pd.read_csv(output / "predictions.csv.gz")
    removed = metrics.iloc[0][["condition", "model"]].tolist()
    metrics = metrics.loc[
        ~(
            metrics["condition"].eq(removed[0])
            & metrics["model"].eq(removed[1])
        )
    ]
    predictions = predictions.loc[
        ~(
            predictions["condition"].eq(removed[0])
            & predictions["model"].eq(removed[1])
        )
    ]
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(
        output / "predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    with pytest.raises(ValueError, match="exactly the expected condition/model pairs"):
        verify_result_directory(output)


def test_prediction_gzip_header_has_zero_mtime(
    input_bundle: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "results"
    _run(input_bundle, output)
    payload = (output / "predictions.csv.gz").read_bytes()
    assert payload[:2] == b"\x1f\x8b"
    assert payload[4:8] == b"\x00\x00\x00\x00"


def test_deterministic_svg_tamper_is_rejected(
    input_bundle: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "results"
    _run(input_bundle, output)
    figure = output / "figures" / "mae_by_condition.svg"
    figure.write_text(
        figure.read_text(encoding="utf-8").replace("lower is better", "changed"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch|figure content mismatch"):
        verify_result_directory(output)
