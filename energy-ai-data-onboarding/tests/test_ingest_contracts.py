from __future__ import annotations

from pathlib import Path

import pandas as pd

from energy_onboarding.contracts import OUTPUT_COLUMNS, condition_frame, validate_condition_frame
from energy_onboarding.ingest import (
    SelectionConfig,
    ingest_selected_dataset,
    metadata_quality_issues,
    read_metadata_csv,
)


def test_wide_ingest_is_column_pruned_and_long(sample_inputs: tuple[Path, Path, Path]) -> None:
    meter, metadata, _ = sample_inputs
    result = ingest_selected_dataset(
        meter,
        metadata,
        selection=SelectionConfig(
            building_ids=("b1", "b3"),
            maximum_building_count=2,
            maximum_source_rows=25,
            read_chunk_rows=7,
        ),
        source_version="fixture",
    )
    assert result.selected_buildings == ["b1", "b3"]
    assert result.source_rows_read == 25
    assert len(result.frame) == 50
    assert set(result.frame["building_id"]) == {"b1", "b3"}
    assert set(result.frame["timezone"]) == {"US/Eastern"}
    published = condition_frame(result.frame, "reference")
    assert list(published.columns) == OUTPUT_COLUMNS
    assert validate_condition_frame(published, expected_condition="reference") == []


def test_metadata_duplicate_and_orphan_contract(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "building_id": ["b1", "b1", None],
            "site_id": ["a", "a", "z"],
            "primaryspaceusage": ["Office", "Office", "Office"],
        }
    ).to_csv(path, index=False)
    metadata = read_metadata_csv(path)
    issues = {record.code: record for record in metadata_quality_issues(metadata)}
    assert issues["METADATA_DUPLICATE_KEY"].status == "fail"
    assert issues["METADATA_MISSING_KEY"].status == "fail"


def test_condition_contract_rejects_duplicate_and_bad_columns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2016-01-01", "2016-01-01"],
            "building_id": ["b1", "b1"],
            "load": [1.0, 2.0],
            "site_id": ["s", "s"],
            "primary_use": ["Office", "Office"],
            "condition": ["reference", "reference"],
        }
    )
    assert any("unique" in error for error in validate_condition_frame(frame))
    assert validate_condition_frame(frame.drop(columns="condition"))[0].startswith("columns")


def test_selected_ingestion_gate_ignores_unselected_metadata_defects(
    tmp_path: Path,
) -> None:
    meter = tmp_path / "electricity.csv"
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "timestamp": ["2016-01-01 00:00:00", "2016-01-01 01:00:00"],
            "b1": [1.0, 2.0],
            "b2": [3.0, 4.0],
        }
    ).to_csv(meter, index=False)
    pd.DataFrame(
        {
            "building_id": ["b1", "b2", "b2"],
            "site_id": ["s1", "s2", "s2-conflict"],
            "primaryspaceusage": ["Office", "Office", "Education"],
            "timezone": ["US/Eastern", "US/Eastern", "US/Eastern"],
        }
    ).to_csv(metadata, index=False)
    result = ingest_selected_dataset(
        meter,
        metadata,
        selection=SelectionConfig(building_ids=("b1",), maximum_building_count=1),
    )
    issues = {record.code: record for record in result.issues}
    assert issues["METADATA_DUPLICATE_KEY"].count == 0
    assert set(result.frame["building_id"]) == {"b1"}
