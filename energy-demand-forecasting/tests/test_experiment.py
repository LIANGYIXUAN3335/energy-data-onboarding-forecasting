from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from energy_forecasting.experiment import run_experiment


def _schema_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key).lower() for key in value} | set().union(
            *(_schema_keys(item) for item in value.values()), set()
        )
    if isinstance(value, list):
        return set().union(*(_schema_keys(item) for item in value), set())
    return set()


def test_end_to_end_experiment(input_bundle: dict[str, Path], tmp_path: Path) -> None:
    output = tmp_path / "results"

    manifest = run_experiment(
        input_bundle["reference"],
        input_bundle["corrupted"],
        input_bundle["remediated"],
        input_bundle["config"],
        output,
        producer_manifest_path=input_bundle["producer_manifest"],
        fault_manifest_path=input_bundle["fault_manifest"],
    )

    metrics = pd.read_csv(output / "metrics.csv")
    assert len(metrics) == 6
    assert set(metrics["condition"]) == {"reference", "corrupted", "remediated"}
    assert set(metrics["model"]) == {"seasonal_naive", "hist_gradient_boosting"}
    assert metrics["r2"].notna().all()
    assert {
        "mase_macro_building",
        "mase_buildings_evaluated",
        "mase_buildings_undefined",
    }.issubset(metrics.columns)
    building_mase = pd.read_csv(output / "building_mase.csv")
    assert set(building_mase["building_id"]) == {
        "SiteA_office_A",
        "SiteA_school_B",
    }
    assert set(pd.read_csv(output / "subgroup_metrics.csv")["group_dimension"]) == {
        "site_id",
        "primary_use",
        "building_id",
        "quarter",
    }
    paired = pd.read_csv(output / "paired_differences.csv")
    assert set(paired["model"]) == {"seasonal_naive", "hist_gradient_boosting"}
    assert manifest["state_policy"].startswith("stateless")
    assert manifest["target_source"] == "reference"
    assert manifest["producer_manifest"]["sha256"]
    assert manifest["code_revision"]["kind"] == "source_tree_sha256"
    assert len(manifest["code_revision"]["value"]) == 64
    assert (output / "report.html").exists()
    assert (output / "predictions.csv.gz").exists()
    assert (output / "model_card.md").exists()
    assert (output / "figures" / "mae_by_condition.svg").exists()
    assert (output / "figures" / "macro_building_mase.svg").exists()
    stored_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    keys = _schema_keys(stored_manifest)
    prohibited = {"prompt", "chat", "memory", "embedding", "vector_database", "personal_data"}
    assert keys.isdisjoint(prohibited)
    assert stored_manifest["metric_policy"]["primary_status"] == "unchanged"
    assert stored_manifest["metric_policy"]["scale_source"] == "reference_training_only"


def test_canonical_run_rejects_nonempty_output(
    input_bundle: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "old-result.txt").write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        run_experiment(
            input_bundle["reference"],
            input_bundle["corrupted"],
            input_bundle["remediated"],
            input_bundle["config"],
            output,
            producer_manifest_path=input_bundle["producer_manifest"],
            fault_manifest_path=input_bundle["fault_manifest"],
        )
