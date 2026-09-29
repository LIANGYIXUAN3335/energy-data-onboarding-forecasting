"""Seeded detector-validation and downstream fault suites."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .checks import QualityConfig, causal_anomaly_masks
from .contracts import json_value


SEVERITY_PARAMETERS: dict[str, dict[str, float | int]] = {
    "low": {"block": 4, "spike": 12.0, "scale": 10.0},
    "medium": {"block": 8, "spike": 30.0, "scale": 100.0},
    "high": {"block": 16, "spike": 75.0, "scale": 1000.0},
}


@dataclass
class FaultSuiteResult:
    frame: pd.DataFrame
    faults: list[dict[str, Any]]


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _fault_record(
    row: pd.Series,
    *,
    fault_type: str,
    original_value: Any,
    corrupted_value: Any,
    severity: str,
    suite: str,
    fault_id: str,
    **extra: Any,
) -> dict[str, Any]:
    record = {
        "fault_id": fault_id,
        "timestamp": json_value(row["timestamp"]),
        "building_id": str(row["building_id"]),
        "original_value": json_value(original_value),
        "corrupted_value": json_value(corrupted_value),
        "fault_type": fault_type,
        "severity": severity,
        "condition": "corrupted",
        "suite": suite,
    }
    record.update({key: json_value(value) for key, value in extra.items()})
    return record


FAULT_FAMILIES = (
    "missing_value",
    "negative_value",
    "positive_spike",
    "long_zero_block",
    "stuck_segment",
    "unit_scale_segment",
)


def normalize_event_counts(events_per_type: int | dict[str, int]) -> dict[str, int]:
    """Validate an integer or per-family mapping of seeded event counts."""

    if isinstance(events_per_type, dict):
        unknown = sorted(set(map(str, events_per_type)).difference(FAULT_FAMILIES))
        if unknown:
            raise ValueError(f"events_per_type has unknown fault families: {unknown}")
        counts = {family: int(events_per_type.get(family, 1)) for family in FAULT_FAMILIES}
    else:
        counts = {family: int(events_per_type) for family in FAULT_FAMILIES}
    if any(count < 1 for count in counts.values()):
        raise ValueError("events_per_type must be at least 1 for every fault family.")
    return counts


def _candidate_groups(frame: pd.DataFrame, cutoff: pd.Timestamp) -> list[pd.Index]:
    eligible = frame.loc[
        frame["timestamp"].notna()
        & frame["timestamp"].lt(cutoff)
        & pd.to_numeric(frame["load"], errors="coerce").notna()
        & np.isfinite(pd.to_numeric(frame["load"], errors="coerce"))
    ].sort_values(["building_id", "timestamp"], kind="stable")
    return [group.index for _, group in eligible.groupby("building_id", sort=True)]


def _choose_single(
    groups: list[pd.Index], rng: np.random.Generator, reserved: set[int]
) -> int:
    """Pick one eligible row uniformly, in building-then-time order.

    The candidate order and the single ``rng.integers`` draw are the same as
    in the original list-based implementation, so suites are reproducible
    across versions for the same seed.
    """

    ordered = np.concatenate([np.asarray(group, dtype=np.int64) for group in groups]) if groups else np.empty(0, dtype=np.int64)
    if reserved:
        blocked = np.fromiter(reserved, dtype=np.int64, count=len(reserved))
        keep = ~np.isin(ordered, blocked)
        candidates = ordered[keep]
    else:
        candidates = ordered
    if candidates.size == 0:
        raise ValueError("Not enough pre-cutoff observations for configured fault suite.")
    return int(candidates[int(rng.integers(0, candidates.size))])


def _choose_block(
    frame: pd.DataFrame,
    groups: list[pd.Index],
    length: int,
    rng: np.random.Generator,
    reserved: set[int],
    *,
    require_positive: bool = False,
) -> list[int]:
    """Pick one contiguous hourly block of ``length`` rows not touching ``reserved``.

    Candidate enumeration is vectorized per building: a start position is
    eligible when every step inside the window is exactly one hour, no row in
    the window is reserved and, when ``require_positive`` is set, every load in
    the window is strictly positive (a zero block, a stuck value or a unit
    scale written over zeros would be an invisible, unfalsifiable fault).
    Sampling is uniform over all eligible starts across buildings, so the
    choice is fully determined by ``rng``.
    """

    candidates: list[tuple[list[int], int, np.ndarray]] = []
    total = 0
    for group in groups:
        indices = np.asarray([int(value) for value in group], dtype=np.int64)
        count = len(indices)
        if count < length:
            continue
        timestamps = pd.DatetimeIndex(frame.loc[indices, "timestamp"])
        steps = np.ones(max(count - 1, 0), dtype=bool)
        if count > 1:
            steps = (timestamps[1:] - timestamps[:-1]) == pd.Timedelta(hours=1)
        reserved_mask = np.fromiter((int(value) in reserved for value in indices), dtype=bool, count=count)
        if require_positive:
            loads = pd.to_numeric(frame.loc[indices, "load"], errors="coerce").to_numpy(dtype=float)
            reserved_mask = reserved_mask | ~(np.isfinite(loads) & (loads > 0))
        window_steps = length - 1
        if window_steps:
            step_ok = np.convolve(steps.astype(np.int64), np.ones(window_steps, dtype=np.int64), mode="valid") == window_steps
        else:
            step_ok = np.ones(count, dtype=bool)
        reserved_hits = np.convolve(reserved_mask.astype(np.int64), np.ones(length, dtype=np.int64), mode="valid")
        eligible = step_ok[: count - length + 1] & (reserved_hits == 0)
        starts = np.flatnonzero(eligible)
        if starts.size:
            candidates.append((indices.tolist(), int(starts.size), starts))
            total += int(starts.size)
    if not total:
        raise ValueError(
            f"Not enough contiguous pre-cutoff observations for a {length}-row fault block."
        )
    choice = int(rng.integers(0, total))
    for indices, size, starts in candidates:
        if choice < size:
            start = int(starts[choice])
            return indices[start : start + length]
        choice -= size
    raise AssertionError("Block selection fell through; this should be unreachable.")


def _set_fault(
    frame: pd.DataFrame,
    indices: Iterable[int],
    *,
    fault_type: str,
    severity: str,
    suite: str,
    fault_id: str,
    transform: Any,
    faults: list[dict[str, Any]],
) -> None:
    for position, index in enumerate(indices):
        original = frame.at[index, "load"]
        corrupted = transform(original, position)
        frame.at[index, "load"] = corrupted
        faults.append(
            _fault_record(
                frame.loc[index],
                fault_type=fault_type,
                original_value=original,
                corrupted_value=corrupted,
                severity=severity,
                suite=suite,
                fault_id=fault_id,
            )
        )


def _enrich_fault_records(
    faults: list[dict[str, Any]], *, suite: str
) -> list[dict[str, Any]]:
    """Attach deterministic event ranges and affected-key digests to each row."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in faults:
        grouped.setdefault(str(record["fault_id"]), []).append(record)
    enriched: list[dict[str, Any]] = []
    for records in grouped.values():
        timestamps = sorted(str(record["timestamp"]) for record in records)
        keys = sorted(
            {
                (str(record["timestamp"]), str(record["building_id"]))
                for record in records
            }
        )
        digest_payload = json.dumps(
            [{"timestamp": timestamp, "building_id": building_id} for timestamp, building_id in keys],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        details = {
            "suite": suite,
            "start_timestamp": timestamps[0],
            "end_timestamp": timestamps[-1],
            "affected_key_count": len(keys),
            "affected_keys_sha256": hashlib.sha256(digest_payload).hexdigest(),
        }
        enriched.extend({**record, **details} for record in records)
    return enriched


def fault_event_summaries(faults: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse enriched per-key records into auditable event-level summaries."""

    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in faults:
        fault_id = str(record["fault_id"])
        if fault_id in seen:
            continue
        seen.add(fault_id)
        summaries.append(
            {
                key: record[key]
                for key in (
                    "fault_id",
                    "fault_type",
                    "suite",
                    "severity",
                    "start_timestamp",
                    "end_timestamp",
                    "affected_key_count",
                    "affected_keys_sha256",
                )
            }
        )
    return summaries


def validate_fault_record_metadata(faults: list[dict[str, Any]]) -> list[str]:
    """Recompute and validate event metadata without trusting the manifest rollup."""

    errors: list[str] = []
    required = {
        "fault_id",
        "fault_type",
        "suite",
        "severity",
        "timestamp",
        "building_id",
        "original_value",
        "corrupted_value",
        "start_timestamp",
        "end_timestamp",
        "affected_key_count",
        "affected_keys_sha256",
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(faults):
        missing = sorted(required.difference(record))
        if missing:
            errors.append(f"fault record {index} missing fields: {missing}")
            continue
        grouped.setdefault(str(record["fault_id"]), []).append(record)

    for fault_id, records in grouped.items():
        suites = {str(record["suite"]) for record in records}
        types = {str(record["fault_type"]) for record in records}
        severities = {str(record["severity"]) for record in records}
        if len(suites) != 1 or len(types) != 1 or len(severities) != 1:
            errors.append(f"fault event {fault_id} has inconsistent event attributes")
            continue
        expected = _enrich_fault_records(
            [dict(record) for record in records], suite=next(iter(suites))
        )
        for index, (observed, recomputed) in enumerate(zip(records, expected)):
            for field in (
                "start_timestamp",
                "end_timestamp",
                "affected_key_count",
                "affected_keys_sha256",
            ):
                if observed.get(field) != recomputed.get(field):
                    errors.append(
                        f"fault event {fault_id} record {index} has invalid {field}"
                    )
    return errors


def inject_downstream_faults(
    reference: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp,
    seed: int,
    severity: str = "medium",
    events_per_type: int | dict[str, int] = 1,
) -> FaultSuiteResult:
    """Inject value-only faults while preserving every observation key.

    ``events_per_type`` seeds that many independent events of every fault
    family (six families), either one integer for all families or a mapping
    ``{family: count}`` (missing families default to 1). Event ``k`` of a
    family has ``fault_id`` ``downstream_forecasting:<family>:00k``. With the
    default of one event per family the suite is identical to the v2 suite
    for the same seed. Events are seeded family by family so that a mapping
    with the same counts as an integer yields the same suite.
    """

    if severity not in SEVERITY_PARAMETERS:
        raise ValueError(f"Unknown severity {severity!r}; choose {sorted(SEVERITY_PARAMETERS)}")
    counts = normalize_event_counts(events_per_type)
    boundary = _utc_timestamp(cutoff)
    corrupted = reference.copy(deep=True).reset_index(drop=True)
    corrupted["timestamp"] = pd.to_datetime(corrupted["timestamp"], errors="coerce", utc=True)
    groups = _candidate_groups(corrupted, boundary)
    rng = np.random.default_rng(seed)
    reserved: set[int] = set()
    faults: list[dict[str, Any]] = []
    parameters = SEVERITY_PARAMETERS[severity]

    single_specs = (
        ("missing_value", lambda value, _: np.nan),
        ("negative_value", lambda value, _: -abs(float(value)) - 1.0),
        ("positive_spike", lambda value, _: max(abs(float(value)), 1.0) * float(parameters["spike"])),
    )
    block_length = int(parameters["block"])
    max_events = max(counts.values())
    # Round-robin over families so that the first event of every family is
    # placed before the second of any family; a mapping with uniform counts
    # therefore reproduces the integer form exactly.
    for event in range(1, max_events + 1):
        for fault_type, transform in single_specs:
            if event > counts[fault_type]:
                continue
            index = _choose_single(groups, rng, reserved)
            reserved.add(index)
            _set_fault(
                corrupted,
                [index],
                fault_type=fault_type,
                severity=severity,
                suite="downstream_forecasting",
                fault_id=f"downstream_forecasting:{fault_type}:{event:03d}",
                transform=transform,
                faults=faults,
            )

        if event <= counts["long_zero_block"]:
            zero_block = _choose_block(
                corrupted, groups, block_length, rng, reserved, require_positive=True
            )
            reserved.update(zero_block)
            _set_fault(
                corrupted,
                zero_block,
                fault_type="long_zero_block",
                severity=severity,
                suite="downstream_forecasting",
                fault_id=f"downstream_forecasting:long_zero_block:{event:03d}",
                transform=lambda _value, _position: 0.0,
                faults=faults,
            )

        if event <= counts["stuck_segment"]:
            stuck_block = _choose_block(
                corrupted, groups, block_length, rng, reserved, require_positive=True
            )
            reserved.update(stuck_block)
            stuck_value = float(corrupted.loc[stuck_block[0], "load"]) * 0.731 + 0.123
            _set_fault(
                corrupted,
                stuck_block,
                fault_type="stuck_segment",
                severity=severity,
                suite="downstream_forecasting",
                fault_id=f"downstream_forecasting:stuck_segment:{event:03d}",
                transform=lambda _value, _position, stuck_value=stuck_value: stuck_value,
                faults=faults,
            )

        if event <= counts["unit_scale_segment"]:
            scale_block = _choose_block(
                corrupted, groups, block_length, rng, reserved, require_positive=True
            )
            reserved.update(scale_block)
            _set_fault(
                corrupted,
                scale_block,
                fault_type="unit_scale_segment",
                severity=severity,
                suite="downstream_forecasting",
                fault_id=f"downstream_forecasting:unit_scale_segment:{event:03d}",
                transform=lambda value, _position: float(value) * float(parameters["scale"]),
                faults=faults,
            )

    original_keys = reference[["timestamp", "building_id"]].astype(str)
    corrupted_keys = corrupted[["timestamp", "building_id"]].astype(str)
    if not original_keys.equals(corrupted_keys):
        raise AssertionError("Downstream suite changed timestamp/building keys.")
    if any(pd.Timestamp(record["timestamp"]) >= boundary for record in faults):
        raise AssertionError("Downstream suite wrote a fault at or after cutoff.")
    return FaultSuiteResult(
        corrupted,
        _enrich_fault_records(faults, suite="downstream_forecasting"),
    )


def inject_detector_validation_faults(
    reference: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp,
    seed: int,
    severity: str = "medium",
    events_per_type: int | dict[str, int] = 1,
) -> FaultSuiteResult:
    """Inject a separate structural/value suite used only to validate detectors."""

    base = inject_downstream_faults(
        reference,
        cutoff=cutoff,
        seed=seed + 101,
        severity=severity,
        events_per_type=events_per_type,
    )
    detector = base.frame.copy(deep=True)
    faults = [
        dict(
            record,
            suite="detector_validation",
            fault_id=str(record["fault_id"]).replace(
                "downstream_forecasting:", "detector_validation:", 1
            ),
        )
        for record in base.faults
    ]
    boundary = _utc_timestamp(cutoff)
    rng = np.random.default_rng(seed + 202)
    groups = _candidate_groups(detector, boundary)
    fault_keys = {(fault["building_id"], fault["timestamp"]) for fault in faults}
    keys = zip(
        detector["building_id"].astype(str),
        (json_value(value) for value in detector["timestamp"]),
    )
    reserved = {int(index) for index, key in zip(detector.index, keys) if key in fault_keys}

    missing_index = _choose_single(groups, rng, reserved)
    reserved.add(missing_index)
    missing_row = detector.loc[missing_index].copy()
    faults.append(
        _fault_record(
            missing_row,
            fault_type="missing_interval",
            original_value=missing_row["load"],
            corrupted_value=None,
            severity=severity,
            suite="detector_validation",
            fault_id="detector_validation:missing_interval:001",
            action="row_removed",
        )
    )
    detector = detector.drop(index=missing_index)

    detector = detector.reset_index(drop=True)
    groups = _candidate_groups(detector, boundary)
    duplicate_index = _choose_single(groups, rng, set())
    duplicate_row = detector.loc[duplicate_index].copy()
    faults.append(
        _fault_record(
            duplicate_row,
            fault_type="duplicate_key",
            original_value=duplicate_row["load"],
            corrupted_value=duplicate_row["load"],
            severity=severity,
            suite="detector_validation",
            fault_id="detector_validation:duplicate_key:001",
            action="row_duplicated",
        )
    )
    detector = pd.concat([detector, duplicate_row.to_frame().T], ignore_index=True)

    groups = _candidate_groups(detector, boundary)
    shift_index = _choose_single(groups, rng, set())
    original_timestamp = detector.at[shift_index, "timestamp"]
    shifted = original_timestamp + pd.Timedelta(hours=2)
    if shifted >= boundary:
        shifted = original_timestamp - pd.Timedelta(hours=2)
    shift_row = detector.loc[shift_index].copy()
    detector.at[shift_index, "timestamp"] = shifted
    faults.append(
        _fault_record(
            shift_row,
            fault_type="timestamp_shift",
            original_value=original_timestamp,
            corrupted_value=shifted,
            severity=severity,
            suite="detector_validation",
            fault_id="detector_validation:timestamp_shift:001",
            corrupted_timestamp=json_value(shifted),
        )
    )

    groups = _candidate_groups(detector, boundary)
    metadata_index = _choose_single(groups, rng, set())
    metadata_row = detector.loc[metadata_index].copy()
    original_site = metadata_row["site_id"]
    detector.at[metadata_index, "site_id"] = pd.NA
    faults.append(
        _fault_record(
            metadata_row,
            fault_type="metadata_mismatch",
            original_value=original_site,
            corrupted_value=None,
            severity=severity,
            suite="detector_validation",
            fault_id="detector_validation:metadata_mismatch:001",
            field="site_id",
        )
    )
    return FaultSuiteResult(
        detector.reset_index(drop=True),
        _enrich_fault_records(faults, suite="detector_validation"),
    )


def detector_metrics(
    corrupted: pd.DataFrame,
    faults: list[dict[str, Any]],
    *,
    quality: QualityConfig | None = None,
) -> pd.DataFrame:
    """Compare detector flags with seeded ground truth only."""

    quality = quality or QualityConfig()
    frame = corrupted.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    masks = causal_anomaly_masks(frame, quality)

    def keys(mask: pd.Series) -> set[tuple[str, str]]:
        selected = frame.loc[mask, ["timestamp", "building_id"]]
        return {
            (json_value(row.timestamp), str(row.building_id))
            for row in selected.itertuples(index=False)
        }

    duplicate = frame.duplicated(["timestamp", "building_id"], keep=False)
    metadata_missing = frame[["site_id", "primary_use"]].isna().any(axis=1)
    detected_by_type = {
        "missing_value": keys(masks["missing"]),
        "negative_value": keys(masks["negative"]),
        "positive_spike": keys(masks["outlier"] | masks["level_shift"]),
        "long_zero_block": keys(masks["long_zero"]),
        "stuck_segment": keys(masks["stuck"]),
        "unit_scale_segment": keys(masks["outlier"] | masks["level_shift"]),
        "duplicate_key": keys(duplicate),
        "metadata_mismatch": keys(metadata_missing),
    }
    universe = {
        (json_value(row.timestamp), str(row.building_id))
        for row in frame[["timestamp", "building_id"]].itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for fault_type in sorted({str(record["fault_type"]) for record in faults}):
        truth = {
            (str(record["timestamp"]), str(record["building_id"]))
            for record in faults
            if record["fault_type"] == fault_type
        }
        if fault_type in {"missing_interval", "timestamp_shift"}:
            # Continuity is an aggregate detector; a removed/shifted interval is
            # detected when its original key is absent or a duplicate was made.
            detected = {key for key in truth if key not in universe}
            if fault_type == "timestamp_shift" and not detected:
                detected = truth if duplicate.any() else set()
            flagged = detected
        else:
            flagged = detected_by_type.get(fault_type, set())
            detected = truth & flagged
        tp = len(detected)
        fn = len(truth - detected)
        fp = len(flagged - truth)
        negatives = max(len(universe - truth), 0)
        tn = max(negatives - fp, 0)
        rows.append(
            {
                "fault_type": fault_type,
                "true_positive": tp,
                "false_negative": fn,
                "false_positive": fp,
                "true_negative": tn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            }
        )
    return pd.DataFrame(rows)
