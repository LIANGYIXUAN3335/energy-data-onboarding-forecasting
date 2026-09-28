from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from energy_onboarding.checks import QualityConfig, validate_fault_cutoff
from energy_onboarding.faults import (
    detector_metrics,
    inject_detector_validation_faults,
    inject_downstream_faults,
)
from energy_onboarding.ingest import SelectionConfig, ingest_selected_dataset
from energy_onboarding.remediation import remediate


def _reference(sample_inputs: tuple[Path, Path, Path]) -> pd.DataFrame:
    meter, metadata, _ = sample_inputs
    return ingest_selected_dataset(
        meter,
        metadata,
        selection=SelectionConfig(
            building_ids=("b1", "b2", "b3"), maximum_building_count=3
        ),
    ).frame


def test_downstream_faults_are_deterministic_key_preserving_and_pre_cutoff(
    sample_inputs: tuple[Path, Path, Path],
) -> None:
    reference = _reference(sample_inputs)
    cutoff = "2016-01-10T00:00:00Z"
    first = inject_downstream_faults(reference, cutoff=cutoff, seed=44, severity="low")
    second = inject_downstream_faults(reference, cutoff=cutoff, seed=44, severity="low")
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.faults == second.faults
    assert set(record["fault_type"] for record in first.faults) == {
        "missing_value",
        "negative_value",
        "positive_spike",
        "long_zero_block",
        "stuck_segment",
        "unit_scale_segment",
    }
    assert validate_fault_cutoff(first.faults, cutoff) == []
    pd.testing.assert_frame_equal(
        reference[["timestamp", "building_id"]].reset_index(drop=True),
        first.frame[["timestamp", "building_id"]].reset_index(drop=True),
    )


def test_detector_suite_is_separate_and_metrics_are_reported(
    sample_inputs: tuple[Path, Path, Path],
) -> None:
    reference = _reference(sample_inputs)
    result = inject_detector_validation_faults(
        reference, cutoff="2016-01-10T00:00:00Z", seed=9, severity="low"
    )
    types = {record["fault_type"] for record in result.faults}
    assert {"missing_interval", "duplicate_key", "timestamp_shift", "metadata_mismatch"} <= types
    metrics = detector_metrics(
        result.frame,
        result.faults,
        quality=QualityConfig(long_zero_threshold=4, stuck_threshold=5),
    )
    assert set(["precision", "recall", "false_positive_rate"]) <= set(metrics.columns)
    assert set(metrics["fault_type"]) == types


def test_remediation_never_reads_future_and_protects_evaluation_window() -> None:
    timestamps = pd.date_range("2016-01-01", periods=6, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "building_id": "b1",
            "load": [np.nan, 5.0, np.nan, 7.0, -9.0, 1000.0],
            "site_id": "s1",
            "primary_use": "Office",
            "timezone": "US/Eastern",
        }
    )
    result = remediate(
        frame,
        quality=QualityConfig(causal_window=3, level_shift_ratio=1000),
        maximum_forward_fill_hours=2,
        repair_before="2016-01-01T04:00:00Z",
    )
    assert pd.isna(result.frame.loc[0, "load"])
    assert result.frame.loc[2, "load"] == 5.0
    assert result.frame.loc[4, "load"] == -9.0
    assert result.frame.loc[5, "load"] == 1000.0
