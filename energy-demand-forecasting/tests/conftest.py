from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.dataset import sha256_file


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=24 * 35, freq="h")
    rows = []
    for building_index, building in enumerate(["SiteA_office_A", "SiteA_school_B"]):
        hour = timestamps.hour.to_numpy()
        day = timestamps.dayofweek.to_numpy()
        load = (
            100
            + building_index * 25
            + 15 * np.sin(2 * np.pi * hour / 24)
            + 5 * np.cos(2 * np.pi * day / 7)
        )
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "building_id": building,
                    "load": load,
                    "site_id": "SiteA",
                    "primary_use": "Office" if building_index == 0 else "Education",
                    "condition": "reference",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def input_bundle(hourly_frame: pd.DataFrame, tmp_path: Path) -> dict[str, Path]:
    bundle = tmp_path / "input"
    bundle.mkdir()
    reference = hourly_frame.copy()
    corrupted = reference.copy()
    remediated = reference.copy()
    corrupted["condition"] = "corrupted"
    remediated["condition"] = "remediated"

    unique_times = reference["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    fault_cutoff = unique_times.iloc[int(len(unique_times) * 0.5)]
    affected = (corrupted["timestamp"] < fault_cutoff) & (corrupted.index % 113 == 0)
    corrupted.loc[affected, "load"] = corrupted.loc[affected, "load"] * 4.0

    paths: dict[str, Path] = {}
    for name, frame in {
        "reference": reference,
        "corrupted": corrupted,
        "remediated": remediated,
    }.items():
        path = bundle / f"{name}.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
        paths[name] = path

    files = {
        path.name: {
            "path": path.name,
            "sha256": sha256_file(path),
            "rows": len(reference),
            "columns": list(reference.columns),
        }
        for path in paths.values()
    }
    producer_manifest = bundle / "dataset_manifest.json"
    producer_manifest.write_text(
        json.dumps({"schema_version": "1.0", "files": files}, indent=2),
        encoding="utf-8",
    )

    faults = []
    for index in corrupted.index[affected]:
        faults.append(
            {
                "timestamp": reference.loc[index, "timestamp"].isoformat(),
                "building_id": reference.loc[index, "building_id"],
                "original_value": float(reference.loc[index, "load"]),
                "corrupted_value": float(corrupted.loc[index, "load"]),
                "fault_type": "scale",
                "severity": "test",
                "condition": "corrupted",
            }
        )
    fault_manifest = bundle / "fault_manifest.json"
    fault_manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "suite": "downstream_forecasting",
                "fault_cutoff": fault_cutoff.isoformat(),
                "reference_sha256": sha256_file(paths["reference"]),
                "faults": faults,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    config = {
        "seed": 42,
        "forecast_horizon_hours": 24,
        "seasonal_period_hours": 168,
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "max_model_iterations": 8,
        "learning_rate": 0.1,
        "max_leaf_nodes": 15,
        "l2_regularization": 0.1,
        "bootstrap_repetitions": 10,
        "condition_order": ["reference", "corrupted", "remediated"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return {
        **paths,
        "producer_manifest": producer_manifest,
        "fault_manifest": fault_manifest,
        "config": config_path,
        "input_dir": bundle,
    }
