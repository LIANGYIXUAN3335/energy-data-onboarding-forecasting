from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    timestamps = pd.date_range("2016-01-01", periods=300, freq="h")
    hour = np.arange(len(timestamps), dtype=float)
    meter = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "b1": 30 + 3 * np.sin(hour / 24 * 2 * np.pi) + hour * 0.01,
            "b2": 45 + 4 * np.cos(hour / 24 * 2 * np.pi) + hour * 0.02,
            "b3": 20 + 2 * np.sin(hour / 168 * 2 * np.pi) + hour * 0.005,
        }
    )
    metadata = pd.DataFrame(
        {
            "building_id": ["b1", "b2", "b3"],
            "site_id": ["site_a", "site_b", "site_c"],
            "primaryspaceusage": ["Education", "Office", "Education"],
            "timezone": ["US/Eastern", "US/Eastern", "US/Eastern"],
        }
    )
    meter_path = tmp_path / "fixture_electricity.csv"
    metadata_path = tmp_path / "fixture_metadata.csv"
    meter.to_csv(meter_path, index=False)
    metadata.to_csv(metadata_path, index=False)

    source_manifest = (
        Path(__file__).parents[1] / "provenance" / "bdg2_v1.0.json"
    ).resolve()
    config = {
        "schema_version": "1.0",
        "dataset": {
            "id": "fixture",
            "title": "Deterministic software fixture",
            "version": "test-only",
            "doi": "not-applicable",
            "expected_frequency": "1h",
            "timestamp_column": "timestamp",
            "source_manifest": str(source_manifest),
        },
        "selection": {
            "site_ids": [],
            "building_ids": ["b1", "b2", "b3"],
            "maximum_building_count": 3,
            "maximum_source_rows": None,
            "full_run": False,
            "strategy": "metadata_round_robin_by_site_then_primary_use",
            "read_chunk_rows": 37,
        },
        "quality": {
            "long_zero_threshold": 4,
            "stuck_threshold": 5,
            "outlier_mad_z": 10.0,
            "level_shift_ratio": 8.0,
            "causal_window_hours": 48,
            "rare_group_fraction": 0.0,
        },
        "faults": {
            "seed": 12345,
            "severity": "low",
            "fault_cutoff": "2016-01-10T00:00:00Z",
        },
        "remediation": {"maximum_forward_fill_hours": 3},
    }
    config_path = tmp_path / "fixture_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return meter_path, metadata_path, config_path
