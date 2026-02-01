from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.cli import main
from energy_forecasting.dataset import SplitBoundary
from energy_forecasting.sensitivity import (
    DIRECT_NUMERIC_FEATURES,
    FORECAST_KEYS,
    HORIZONS,
    _common_predictions,
    assign_origin_partitions,
    build_direct_examples,
    run_sensitivity_experiment,
    validate_exclusive_fault_boundary,
    verify_post_cutoff_condition_parity,
    verify_sensitivity_result_directory,
)


def test_exclusive_fault_cutoff_allows_exact_next_hour_only() -> None:
    train_end = pd.Timestamp("2016-12-31T23:00:00Z")
    validate_exclusive_fault_boundary(
        pd.Timestamp("2017-01-01T00:00:00Z"), train_end
    )
    with pytest.raises(ValueError, match="first hourly timestamp"):
        validate_exclusive_fault_boundary(
            pd.Timestamp("2017-01-01T01:00:00Z"), train_end
        )


def test_common_alignment_drops_an_entire_partial_origin_vector() -> None:
    pairs = [
        (condition, model)
        for condition in ("reference", "corrupted", "remediated")
        for model in ("seasonal_naive", "hist_gradient_boosting")
    ]
    rows: list[dict[str, object]] = []
    for origin in pd.to_datetime(
        ["2017-04-01T23:00:00Z", "2017-04-02T23:00:00Z"], utc=True
    ):
        for horizon in HORIZONS:
            for condition, model in pairs:
                rows.append(
                    {
                        "forecast_origin": origin,
                        "horizon_hours": horizon,
                        "target_timestamp": origin + pd.Timedelta(hours=horizon),
                        "building_id": "b1",
                        "condition": condition,
                        "model": model,
                    }
                )
    frame = pd.DataFrame(rows)
    second_origin = pd.Timestamp("2017-04-02T23:00:00Z")
    frame = frame.loc[
        ~(
            frame["forecast_origin"].eq(second_origin)
            & frame["horizon_hours"].eq(24)
        )
    ]
    retained, summary = _common_predictions(frame, expected_pairs=len(pairs))
    assert set(retained["forecast_origin"]) == {
        pd.Timestamp("2017-04-01T23:00:00Z")
    }
    assert summary["retained_complete_origin_building_groups"] == 1
    assert summary["excluded_incomplete_origin_building_groups"] == 1


def _sensitivity_config(path: Path) -> Path:
    config = {
        "analysis_kind": "direct_horizons_1_to_24_sensitivity",
        "seed": 42,
        "horizons_hours": list(HORIZONS),
        "origin_hour": 23,
        "origin_stride_hours": 24,
        "seasonal_period_hours": 168,
        "train_end": "2024-01-20T23:00:00Z",
        "validation_end": "2024-01-28T23:00:00Z",
        "max_model_iterations": 8,
        "learning_rate": 0.1,
        "max_leaf_nodes": 15,
        "l2_regularization": 0.1,
        "bootstrap_repetitions": 10,
        "bootstrap_block": "building_week",
        "condition_order": ["reference", "corrupted", "remediated"],
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_direct_features_ignore_all_values_after_origin(
    hourly_frame: pd.DataFrame,
) -> None:
    origin = pd.Timestamp("2024-01-15 23:00:00", tz="UTC")
    original = hourly_frame.copy()
    changed = hourly_frame.copy()
    timestamp_utc = pd.to_datetime(changed["timestamp"], utc=True)
    changed.loc[
        (changed["building_id"] == "SiteA_office_A")
        & (timestamp_utc > origin)
        & (timestamp_utc <= origin + pd.Timedelta(hours=24)),
        "load",
    ] = 9_999_999.0
    original_examples = build_direct_examples(original)
    changed_examples = build_direct_examples(changed)
    selector = (
        (original_examples["building_id"] == "SiteA_office_A")
        & (original_examples["forecast_origin"] == origin)
    )
    left = original_examples.loc[
        selector, DIRECT_NUMERIC_FEATURES + ["direct_lag_168", "direct_lag_24"]
    ]
    right = changed_examples.loc[
        selector, DIRECT_NUMERIC_FEATURES + ["direct_lag_168", "direct_lag_24"]
    ]
    assert np.allclose(
        left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True
    )
    assert not np.allclose(
        original_examples.loc[selector, "condition_target"],
        changed_examples.loc[selector, "condition_target"],
    )


def test_horizon_above_24_is_rejected(hourly_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="1 through 24"):
        build_direct_examples(hourly_frame, horizons=tuple(range(1, 26)))


def test_target_time_partition_embargoes_boundary_straddling_origin(
    hourly_frame: pd.DataFrame,
) -> None:
    examples = build_direct_examples(hourly_frame)
    boundary = SplitBoundary(
        train_end=pd.Timestamp("2024-01-11 12:00:00", tz="UTC"),
        validation_end=pd.Timestamp("2024-01-20 12:00:00", tz="UTC"),
        test_end=pd.Timestamp("2024-02-04 23:00:00", tz="UTC"),
    )
    labels = assign_origin_partitions(examples, boundary)
    origin = pd.Timestamp("2024-01-10 23:00:00", tz="UTC")
    assert set(labels[examples["forecast_origin"] == origin]) == {
        "excluded_boundary_straddle"
    }
    for forecast_origin, group in examples.assign(partition=labels).groupby(
        "forecast_origin", observed=True
    ):
        if group["partition"].iloc[0] == "train":
            assert group["target_timestamp"].max() <= boundary.train_end
        if group["partition"].iloc[0] == "validation":
            assert group["target_timestamp"].min() > boundary.train_end
            assert group["target_timestamp"].max() <= boundary.validation_end


def test_post_cutoff_condition_parity_fails_closed(
    hourly_frame: pd.DataFrame,
) -> None:
    reference = hourly_frame.copy()
    reference["timestamp"] = pd.to_datetime(reference["timestamp"], utc=True)
    corrupted = reference.copy()
    remediated = reference.copy()
    cutoff = pd.Timestamp("2024-01-10 00:00:00", tz="UTC")
    changed_index = corrupted.index[corrupted["timestamp"] == cutoff][0]
    corrupted.loc[changed_index, "load"] += 1.0
    with pytest.raises(ValueError, match="post-cutoff parity"):
        verify_post_cutoff_condition_parity(
            {
                "reference": reference,
                "corrupted": corrupted,
                "remediated": remediated,
            },
            cutoff,
        )


def test_sensitivity_end_to_end_alignment_and_tamper_detection(
    input_bundle: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _sensitivity_config(tmp_path / "sensitivity.json")
    output = tmp_path / "sensitivity-results"
    manifest = run_sensitivity_experiment(
        input_bundle["reference"],
        input_bundle["corrupted"],
        input_bundle["remediated"],
        config,
        output,
        producer_manifest_path=input_bundle["producer_manifest"],
        fault_manifest_path=input_bundle["fault_manifest"],
    )
    assert manifest["canonical_run_unchanged"] is True
    assert manifest["post_cutoff_condition_parity"]["passed"] is True
    predictions = pd.read_csv(
        output / "predictions.csv.gz",
        parse_dates=["forecast_origin", "cutoff_timestamp", "target_timestamp"],
    )
    assert not predictions.duplicated(
        FORECAST_KEYS + ["condition", "model"]
    ).any()
    assert set(predictions["horizon_hours"]) == set(HORIZONS)
    assert (predictions["forecast_origin"] == predictions["cutoff_timestamp"]).all()
    assert (
        predictions["target_timestamp"]
        == predictions["forecast_origin"]
        + pd.to_timedelta(predictions["horizon_hours"], unit="h")
    ).all()
    assert len(pd.read_csv(output / "horizon_metrics.csv")) == 6 * 24
    assert len(pd.read_csv(output / "paired_differences.csv")) == 2
    assert len(pd.read_csv(output / "horizon_paired_differences.csv")) == 2 * 24
    unbound_summary = verify_sensitivity_result_directory(output)
    assert unbound_summary["source_binding_verified"] is False
    summary = verify_sensitivity_result_directory(
        output, input_dir=input_bundle["input_dir"]
    )
    assert summary["checks"]["horizon_metrics_recomputed"] is True
    assert summary["checks"]["shared_reference_targets"] is True
    assert summary["checks"]["paired_block_differences_recomputed"] is True
    assert summary["source_binding_verified"] is True
    assert main(
        [
            "verify-sensitivity-results",
            "--result-dir",
            str(output),
            "--input-dir",
            str(input_bundle["input_dir"]),
        ]
    ) == 0
    assert '"source_binding_verified": true' in capsys.readouterr().out

    manifest_path = output / "run_manifest.json"
    original_manifest_text = manifest_path.read_text(encoding="utf-8")
    changed_manifest = json.loads(original_manifest_text)
    changed_manifest["result_files"]["metrics.csv"]["size_bytes"] += 1
    manifest_path.write_text(json.dumps(changed_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_sensitivity_result_directory(output)
    manifest_path.write_text(original_manifest_text, encoding="utf-8")

    horizon_path = output / "horizon_metrics.csv"
    original_horizon_payload = horizon_path.read_bytes()
    changed_horizons = pd.read_csv(horizon_path)
    changed_horizons["horizon_hours"] = changed_horizons["horizon_hours"].astype(float)
    changed_horizons.loc[0, "horizon_hours"] = 1.5
    changed_horizons.to_csv(horizon_path, index=False)
    with pytest.raises(ValueError, match="exact integers"):
        verify_sensitivity_result_directory(output)
    horizon_path.write_bytes(original_horizon_payload)

    paired_path = output / "paired_differences.csv"
    original_paired_payload = paired_path.read_bytes()
    changed_paired = pd.read_csv(paired_path)
    changed_paired.loc[0, "model"] = "unexpected_model"
    changed_paired.to_csv(paired_path, index=False)
    with pytest.raises(ValueError, match="exactly one aggregate row per model"):
        verify_sensitivity_result_directory(output)
    paired_path.write_bytes(original_paired_payload)

    horizon_paired_path = output / "horizon_paired_differences.csv"
    original_horizon_paired_payload = horizon_paired_path.read_bytes()
    changed_horizon_paired = pd.read_csv(horizon_paired_path)
    changed_horizon_paired.loc[0, "horizon_hours"] = changed_horizon_paired.loc[
        1, "horizon_hours"
    ]
    changed_horizon_paired.to_csv(horizon_paired_path, index=False)
    with pytest.raises(ValueError, match="exactly the expected models times horizons"):
        verify_sensitivity_result_directory(output)
    horizon_paired_path.write_bytes(original_horizon_paired_payload)

    metrics_path = output / "metrics.csv"
    predictions_path = output / "predictions.csv.gz"
    original_metrics_payload = metrics_path.read_bytes()
    original_predictions_payload = predictions_path.read_bytes()
    changed_metrics = pd.read_csv(metrics_path)
    removed_condition = str(changed_metrics.loc[0, "condition"])
    removed_model = str(changed_metrics.loc[0, "model"])
    changed_metrics = changed_metrics.loc[
        ~(
            changed_metrics["condition"].eq(removed_condition)
            & changed_metrics["model"].eq(removed_model)
        )
    ]
    changed_predictions = predictions.loc[
        ~(
            predictions["condition"].eq(removed_condition)
            & predictions["model"].eq(removed_model)
        )
    ]
    changed_metrics.to_csv(metrics_path, index=False)
    changed_predictions.to_csv(
        predictions_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    with pytest.raises(ValueError, match="exactly the expected condition/model pairs"):
        verify_sensitivity_result_directory(output)
    metrics_path.write_bytes(original_metrics_payload)
    predictions_path.write_bytes(original_predictions_payload)

    first = predictions.iloc[0]
    same_source_key = (
        predictions["target_timestamp"].eq(first["target_timestamp"])
        & predictions["building_id"].eq(first["building_id"])
    )
    assert int(same_source_key.sum()) == 6
    predictions.loc[same_source_key, "target"] += 1.0
    predictions.to_csv(
        predictions_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    with pytest.raises(ValueError, match="do not match reference input"):
        verify_sensitivity_result_directory(
            output, input_dir=input_bundle["input_dir"]
        )


def test_sensitivity_rejects_nonempty_output_directory(
    input_bundle: dict[str, Path], tmp_path: Path
) -> None:
    config = _sensitivity_config(tmp_path / "sensitivity.json")
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        run_sensitivity_experiment(
            input_bundle["reference"],
            input_bundle["corrupted"],
            input_bundle["remediated"],
            config,
            output,
            producer_manifest_path=input_bundle["producer_manifest"],
            fault_manifest_path=input_bundle["fault_manifest"],
        )
