from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_onboarding.checks import run_quality_checks, split_boundary_checks
from energy_onboarding.gates import (
    apply_check_thresholds,
    gate_summary,
    hard_structural_failures,
)


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range("2016-01-01", periods=20, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "building_id": "b1",
            "load": np.arange(20, dtype=float),
            "site_id": "s1",
            "primary_use": "Office",
            "timezone": "US/Eastern",
        }
    )


def test_per_check_threshold_preserves_observed_count_and_denominator() -> None:
    frame = _frame()
    frame.loc[[1, 2, 3], "load"] = np.nan
    records = run_quality_checks(frame)
    evaluated = apply_check_thresholds(
        records,
        {
            "check_thresholds": {
                "LOAD_MISSING": {
                    "basis": "rate",
                    "warn_above": 0.1,
                    "fail_above": 0.2,
                }
            }
        },
    )
    summary = gate_summary(evaluated)
    decision = next(
        result
        for result in summary["check_results"]
        if result["issue_code"] == "LOAD_MISSING"
    )
    assert decision["status"] == "warn"
    assert decision["count"] == 3
    assert decision["denominator"] == 20
    assert decision["rate"] == pytest.approx(0.15)
    assert decision["threshold"] == {
        "basis": "rate",
        "warn_above": 0.1,
        "fail_above": 0.2,
        "comparison": "strictly_greater_than",
    }


def test_unknown_gate_issue_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown issue codes"):
        apply_check_thresholds(
            run_quality_checks(_frame()),
            {"check_thresholds": {"LOAD_MISNG": {"fail_above": 0}}},
        )


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_thresholds_are_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="finite number"):
        apply_check_thresholds(
            run_quality_checks(_frame()),
            {
                "check_thresholds": {
                    "SCHEMA_REQUIRED_COLUMN_MISSING": {
                        "basis": "count",
                        "fail_above": threshold,
                    }
                }
            },
        )


@pytest.mark.parametrize("field", ["warn_above", "fail_above"])
def test_rate_thresholds_must_be_within_zero_and_one(field: str) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_check_thresholds(
            run_quality_checks(_frame()),
            {
                "check_thresholds": {
                    "LOAD_MISSING": {"basis": "rate", field: 1.01}
                }
            },
        )


def test_hard_structural_failure_cannot_be_downgraded_by_threshold() -> None:
    frame = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)
    records = apply_check_thresholds(
        run_quality_checks(frame),
        {
            "check_thresholds": {
                "DUPLICATE_OBSERVATION_KEY": {
                    "basis": "count",
                    "warn_above": 1000,
                    "fail_above": 2000,
                }
            }
        },
    )
    duplicate = next(
        record for record in records if record.code == "DUPLICATE_OBSERVATION_KEY"
    )
    assert duplicate.status == "fail"
    assert duplicate.threshold_comparison == "hard_structural_count_greater_than_zero"
    assert hard_structural_failures(records) == [duplicate]


def test_split_checks_report_building_and_subgroup_denominators() -> None:
    frame = _frame()
    second = frame.iloc[:5].copy()
    second["building_id"] = "b2"
    second["site_id"] = "s2"
    second["primary_use"] = "Education"
    combined = pd.concat([frame, second], ignore_index=True)
    outcomes = {
        record.code: record
        for record in split_boundary_checks(
            combined, "2016-01-01T10:00:00Z", source_condition="reference_natural"
        )
    }
    assert outcomes["SPLIT_BUILDING_COVERAGE"].count == 1
    assert outcomes["SPLIT_BUILDING_COVERAGE"].denominator == 2
    assert outcomes["SPLIT_BUILDING_COVERAGE"].status == "fail"
    assert outcomes["SUBGROUP_SPLIT_COVERAGE"].count == 1
    assert outcomes["SUBGROUP_SPLIT_COVERAGE"].denominator == 2
    assert "stored source-local wall-clock" in outcomes[
        "SUBGROUP_SPLIT_COVERAGE"
    ].expected_rule
