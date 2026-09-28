from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSet:
    n: int
    mae: float
    rmse: float
    wmape: float
    mase: float
    r2: float
    mae_ci_low: float
    mae_ci_high: float


@dataclass(frozen=True)
class PairedDifference:
    n: int
    corrupted_mae: float
    remediated_mae: float
    mean_difference: float
    difference_ci_low: float
    difference_ci_high: float


@dataclass(frozen=True)
class BuildingMaseSummary:
    """Supplementary macro average over building-specific MASE values."""

    macro_mase: float
    buildings_evaluated: int
    buildings_undefined: int


def seasonal_scale(train_reference: pd.DataFrame, period: int = 168) -> float:
    differences: list[np.ndarray] = []
    for _, group in train_reference.groupby("building_id", observed=True):
        values = group.sort_values("timestamp")["load"].to_numpy(dtype=float)
        if len(values) <= period:
            continue
        diff = np.abs(values[period:] - values[:-period])
        differences.append(diff[np.isfinite(diff)])
    if not differences:
        return float("nan")
    joined = np.concatenate(differences)
    if joined.size == 0:
        return float("nan")
    scale = float(np.mean(joined))
    return scale if scale > 0 else float("nan")


def seasonal_scales_by_building(
    train_reference: pd.DataFrame,
    period: int = 168,
) -> dict[str, float]:
    """Return one scale per building, always from reference training data.

    A building whose reference-training history is too short, non-finite, or
    seasonally constant receives ``NaN``.  Keeping those buildings explicit is
    important: the supplementary macro result must not silently imply complete
    building coverage.
    """

    scales: dict[str, float] = {}
    for building_id, group in train_reference.groupby("building_id", observed=True):
        values = group.sort_values("timestamp")["load"].to_numpy(dtype=float)
        if len(values) <= period:
            scales[str(building_id)] = float("nan")
            continue
        differences = np.abs(values[period:] - values[:-period])
        finite = differences[np.isfinite(differences)]
        if finite.size == 0:
            scales[str(building_id)] = float("nan")
            continue
        scale = float(np.mean(finite))
        scales[str(building_id)] = scale if scale > 0 else float("nan")
    return scales


def building_scale_coverage(scales: dict[str, float]) -> dict[str, object]:
    """Describe finite and undefined building denominators deterministically."""

    valid = sorted(
        str(name)
        for name, value in scales.items()
        if np.isfinite(float(value)) and float(value) > 0
    )
    undefined = sorted(str(name) for name in scales if str(name) not in set(valid))
    return {
        "building_count": len(scales),
        "valid_scale_buildings": valid,
        "undefined_scale_buildings": undefined,
        "valid_scale_count": len(valid),
        "undefined_scale_count": len(undefined),
    }


def per_building_mase(
    target: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
    building_id: np.ndarray | pd.Series,
    reference_training_scales: dict[str, float],
) -> tuple[pd.DataFrame, BuildingMaseSummary]:
    """Calculate auditable per-building MASE and its unweighted macro mean.

    The returned rows retain buildings with an undefined scale, while the macro
    average uses only finite building MASE values and reports both coverage
    counts.  This is supplementary to the existing pooled MASE and does not
    change the canonical primary metric.
    """

    frame = pd.DataFrame(
        {
            "target": np.asarray(target, dtype=float),
            "prediction": np.asarray(prediction, dtype=float),
            "building_id": pd.Series(building_id, copy=False).astype(str).to_numpy(),
        }
    )
    rows: list[dict[str, float | int | str]] = []
    for name, group in frame.groupby("building_id", observed=True, sort=True):
        actual = group["target"].to_numpy(dtype=float)
        predicted = group["prediction"].to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(predicted)
        count = int(valid.sum())
        mae = (
            float(np.mean(np.abs(predicted[valid] - actual[valid])))
            if count
            else float("nan")
        )
        scale = float(reference_training_scales.get(str(name), float("nan")))
        mase = (
            float(mae / scale)
            if np.isfinite(mae) and np.isfinite(scale) and scale > 0
            else float("nan")
        )
        rows.append(
            {
                "building_id": str(name),
                "n": count,
                "mae": mae,
                "reference_training_scale": scale,
                "mase": mase,
            }
        )
    table = pd.DataFrame(
        rows,
        columns=[
            "building_id",
            "n",
            "mae",
            "reference_training_scale",
            "mase",
        ],
    )
    finite = table["mase"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    summary = BuildingMaseSummary(
        macro_mase=float(np.mean(finite)) if finite.size else float("nan"),
        buildings_evaluated=int(finite.size),
        buildings_undefined=int(len(table) - finite.size),
    )
    return table, summary


def bootstrap_mae_interval(
    absolute_error: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    values = absolute_error[np.isfinite(absolute_error)]
    if values.size == 0:
        return float("nan"), float("nan")
    if repetitions <= 1:
        mean = float(np.mean(values))
        return mean, mean
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = rng.choice(values, size=values.size, replace=True)
        estimates[index] = np.mean(sample)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_mean_interval(
    values: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for a paired mean."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    if repetitions <= 1:
        mean = float(np.mean(finite))
        return mean, mean
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample_indices = rng.integers(0, finite.size, size=finite.size)
        estimates[index] = np.mean(finite[sample_indices])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def block_bootstrap_mean_interval(
    values: np.ndarray | pd.Series,
    block_ids: np.ndarray | pd.Series,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap a row-weighted mean by whole blocks with bounded memory.

    Each repetition samples only ``number_of_blocks`` integer indices and then
    combines pre-aggregated block sums/counts.  It never allocates an
    ``N rows x repetitions`` matrix.  This is used by the multi-horizon
    sensitivity analysis, where within-building/week errors are correlated.
    """

    frame = pd.DataFrame(
        {
            "value": np.asarray(values, dtype=float),
            "block": pd.Series(block_ids, copy=False).astype(str).to_numpy(),
        }
    )
    frame = frame[np.isfinite(frame["value"])].copy()
    if frame.empty:
        return float("nan"), float("nan")
    aggregate = frame.groupby("block", observed=True, sort=True)["value"].agg(
        ["sum", "count"]
    )
    sums = aggregate["sum"].to_numpy(dtype=float)
    counts = aggregate["count"].to_numpy(dtype=float)
    if repetitions <= 1:
        mean = float(sums.sum() / counts.sum())
        return mean, mean
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    block_count = len(aggregate)
    for index in range(repetitions):
        sampled = rng.integers(0, block_count, size=block_count)
        estimates[index] = float(sums[sampled].sum() / counts[sampled].sum())
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def calculate_metrics_blocked(
    target: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
    scale: float,
    block_ids: np.ndarray | pd.Series,
    bootstrap_repetitions: int,
    seed: int,
) -> MetricSet:
    """Calculate the standard metric set with a whole-block MAE interval."""

    actual = np.asarray(target, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    blocks = pd.Series(block_ids, copy=False).astype(str).to_numpy()
    valid = np.isfinite(actual) & np.isfinite(predicted)
    base = calculate_metrics(actual, predicted, scale, 1, seed)
    low, high = block_bootstrap_mean_interval(
        np.abs(predicted[valid] - actual[valid]),
        blocks[valid],
        bootstrap_repetitions,
        seed,
    )
    return MetricSet(
        n=base.n,
        mae=base.mae,
        rmse=base.rmse,
        wmape=base.wmape,
        mase=base.mase,
        r2=base.r2,
        mae_ci_low=low,
        mae_ci_high=high,
    )


def calculate_metrics(
    target: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
    scale: float,
    bootstrap_repetitions: int,
    seed: int,
) -> MetricSet:
    actual = np.asarray(target, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    if actual.size == 0:
        raise ValueError("No finite target/prediction pairs are available")
    errors = predicted - actual
    absolute = np.abs(errors)
    mae = float(np.mean(absolute))
    denominator = float(np.sum(np.abs(actual)))
    wmape = float(np.sum(absolute) / denominator) if denominator > 0 else float("nan")
    mase = float(mae / scale) if np.isfinite(scale) and scale > 0 else float("nan")
    centered = actual - np.mean(actual)
    total_sum_squares = float(np.sum(centered**2))
    r2 = (
        float(1.0 - np.sum(errors**2) / total_sum_squares)
        if actual.size >= 2 and total_sum_squares > 0
        else float("nan")
    )
    low, high = bootstrap_mae_interval(absolute, bootstrap_repetitions, seed)
    return MetricSet(
        n=int(actual.size),
        mae=mae,
        rmse=float(np.sqrt(np.mean(errors**2))),
        wmape=wmape,
        mase=mase,
        r2=r2,
        mae_ci_low=low,
        mae_ci_high=high,
    )


def paired_absolute_error_difference(
    target: np.ndarray | pd.Series,
    corrupted_prediction: np.ndarray | pd.Series,
    remediated_prediction: np.ndarray | pd.Series,
    bootstrap_repetitions: int,
    seed: int,
) -> PairedDifference:
    """Compare remediated and corrupted absolute errors on identical rows.

    The signed difference is ``remediated - corrupted``; negative values favor
    remediation. Bootstrap samples preserve row pairing.
    """
    actual = np.asarray(target, dtype=float)
    corrupted = np.asarray(corrupted_prediction, dtype=float)
    remediated = np.asarray(remediated_prediction, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(corrupted) & np.isfinite(remediated)
    if not valid.any():
        raise ValueError("No finite paired predictions are available")
    corrupted_error = np.abs(corrupted[valid] - actual[valid])
    remediated_error = np.abs(remediated[valid] - actual[valid])
    differences = remediated_error - corrupted_error
    low, high = bootstrap_mean_interval(differences, bootstrap_repetitions, seed)
    return PairedDifference(
        n=int(differences.size),
        corrupted_mae=float(np.mean(corrupted_error)),
        remediated_mae=float(np.mean(remediated_error)),
        mean_difference=float(np.mean(differences)),
        difference_ci_low=low,
        difference_ci_high=high,
    )


def paired_absolute_error_difference_blocked(
    target: np.ndarray | pd.Series,
    corrupted_prediction: np.ndarray | pd.Series,
    remediated_prediction: np.ndarray | pd.Series,
    block_ids: np.ndarray | pd.Series,
    bootstrap_repetitions: int,
    seed: int,
) -> PairedDifference:
    """Paired remediation-minus-corruption difference with block resampling.

    Pairing is preserved row by row before whole building-week blocks are
    sampled. Negative values favor remediation. The block bootstrap uses the
    same bounded-memory implementation as sensitivity MAE intervals.
    """

    actual = np.asarray(target, dtype=float)
    corrupted = np.asarray(corrupted_prediction, dtype=float)
    remediated = np.asarray(remediated_prediction, dtype=float)
    blocks = pd.Series(block_ids, copy=False).astype(str).to_numpy()
    valid = np.isfinite(actual) & np.isfinite(corrupted) & np.isfinite(remediated)
    if not valid.any():
        raise ValueError("No finite block-paired predictions are available")
    corrupted_error = np.abs(corrupted[valid] - actual[valid])
    remediated_error = np.abs(remediated[valid] - actual[valid])
    differences = remediated_error - corrupted_error
    low, high = block_bootstrap_mean_interval(
        differences,
        blocks[valid],
        bootstrap_repetitions,
        seed,
    )
    return PairedDifference(
        n=int(differences.size),
        corrupted_mae=float(np.mean(corrupted_error)),
        remediated_mae=float(np.mean(remediated_error)),
        mean_difference=float(np.mean(differences)),
        difference_ci_low=low,
        difference_ci_high=high,
    )
