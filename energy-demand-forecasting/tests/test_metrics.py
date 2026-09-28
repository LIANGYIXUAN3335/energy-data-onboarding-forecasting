from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.metrics import (
    block_bootstrap_mean_interval,
    calculate_metrics,
    paired_absolute_error_difference,
    paired_absolute_error_difference_blocked,
    per_building_mase,
    seasonal_scales_by_building,
)


def test_metrics_known_values() -> None:
    metric = calculate_metrics(
        np.array([10.0, 20.0, 30.0]),
        np.array([11.0, 18.0, 33.0]),
        scale=2.0,
        bootstrap_repetitions=20,
        seed=7,
    )
    assert metric.n == 3
    assert metric.mae == 2.0
    assert metric.mase == 1.0
    assert np.isclose(metric.rmse, np.sqrt(14 / 3))
    assert np.isclose(metric.wmape, 6 / 60)
    assert np.isclose(metric.r2, 0.93)


def test_paired_difference_known_values() -> None:
    difference = paired_absolute_error_difference(
        np.array([10.0, 20.0, 30.0]),
        np.array([14.0, 18.0, 35.0]),
        np.array([11.0, 18.0, 32.0]),
        bootstrap_repetitions=1,
        seed=3,
    )
    assert difference.n == 3
    assert difference.corrupted_mae == 11 / 3
    assert difference.remediated_mae == 5 / 3
    assert difference.mean_difference == -2.0
    assert difference.difference_ci_low == -2.0
    assert difference.difference_ci_high == -2.0


def test_per_building_mase_uses_reference_training_scales() -> None:
    training = pd.DataFrame(
        {
            "timestamp": list(pd.date_range("2024-01-01", periods=4, freq="h")) * 2,
            "building_id": ["A"] * 4 + ["B"] * 4,
            "load": [1.0, 2.0, 3.0, 4.0, 10.0, 14.0, 18.0, 22.0],
        }
    )
    scales = seasonal_scales_by_building(training, period=2)
    assert scales == {"A": 2.0, "B": 8.0}
    table, summary = per_building_mase(
        target=np.array([10.0, 12.0, 30.0, 32.0]),
        prediction=np.array([12.0, 14.0, 34.0, 36.0]),
        building_id=np.array(["A", "A", "B", "B"]),
        reference_training_scales=scales,
    )
    assert table.set_index("building_id").loc["A", "mase"] == 1.0
    assert table.set_index("building_id").loc["B", "mase"] == 0.5
    assert summary.macro_mase == 0.75
    assert summary.buildings_evaluated == 2
    assert summary.buildings_undefined == 0


def test_block_bootstrap_is_deterministic_and_streamed_by_block() -> None:
    values = np.array([1.0, 3.0, 10.0, 14.0])
    blocks = np.array(["A-week", "A-week", "B-week", "B-week"])
    first = block_bootstrap_mean_interval(values, blocks, repetitions=50, seed=11)
    second = block_bootstrap_mean_interval(values, blocks, repetitions=50, seed=11)
    assert first == second
    assert first[0] <= np.mean(values) <= first[1]


def test_paired_block_difference_preserves_pairing() -> None:
    result = paired_absolute_error_difference_blocked(
        target=np.array([10.0, 20.0, 30.0, 40.0]),
        corrupted_prediction=np.array([14.0, 25.0, 36.0, 48.0]),
        remediated_prediction=np.array([11.0, 22.0, 33.0, 44.0]),
        block_ids=np.array(["A-week", "A-week", "B-week", "B-week"]),
        bootstrap_repetitions=50,
        seed=9,
    )
    assert result.corrupted_mae == 5.75
    assert result.remediated_mae == 2.5
    assert result.mean_difference == -3.25
    assert result.difference_ci_low <= result.mean_difference
    assert result.difference_ci_high >= result.mean_difference
