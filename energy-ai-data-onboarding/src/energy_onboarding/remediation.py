"""Deterministic, past-only remediation and explicit quarantine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .checks import QualityConfig, causal_anomaly_masks
from .contracts import condition_frame, json_value


@dataclass
class RemediationResult:
    frame: pd.DataFrame
    log: pd.DataFrame
    quarantine: pd.DataFrame


LOG_COLUMNS = [
    "timestamp",
    "building_id",
    "original_value",
    "remediated_value",
    "rule",
    "status",
    "reason",
]

QUARANTINE_COLUMNS = [
    "timestamp",
    "building_id",
    "load",
    "site_id",
    "primary_use",
    "condition",
    "reason",
]

REPAIR_STRATEGIES = ("forward_fill", "forward_then_profile")


def remediate(
    corrupted: pd.DataFrame,
    *,
    quality: QualityConfig | None = None,
    maximum_forward_fill_hours: int = 3,
    repair_before: str | pd.Timestamp | None = None,
    repair_strategy: str = "forward_fill",
    maximum_profile_fill_hours: int = 48,
    profile_weeks: int = 8,
    profile_fill_ambiguous_scale: bool = False,
) -> RemediationResult:
    """Repair defects from earlier rows only; quarantine the remainder.

    Two past-only repair rules exist:

    * ``forward_fill``: an invalid row inside a gap of at most
      ``maximum_forward_fill_hours`` consecutive invalid rows takes the last
      valid value observed before the gap.
    * ``forward_then_profile`` (v3): rows that the forward fill cannot cover,
      up to ``maximum_profile_fill_hours`` consecutive invalid rows, take the
      median of the same weekday/hour over the previous ``profile_weeks`` weeks,
      computed only from originally valid observations that are strictly
      earlier than the row being repaired.

    The algorithm never performs backward fill, centered smoothing, or linear
    interpolation, and never divides a suspected unit shift by an inferred
    factor. Suspected unit/level shifts are quarantined unless
    ``profile_fill_ambiguous_scale`` explicitly allows the profile rule to
    replace them with the building's typical value for that hour.
    """

    if maximum_forward_fill_hours < 0:
        raise ValueError("maximum_forward_fill_hours must be non-negative.")
    if repair_strategy not in REPAIR_STRATEGIES:
        raise ValueError(f"repair_strategy must be one of {REPAIR_STRATEGIES}.")
    if maximum_profile_fill_hours < 0:
        raise ValueError("maximum_profile_fill_hours must be non-negative.")
    if profile_weeks < 1:
        raise ValueError("profile_weeks must be at least 1.")
    config = quality or QualityConfig()
    work = corrupted.copy(deep=True).reset_index(drop=True)
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    work["load"] = pd.to_numeric(work["load"], errors="coerce")
    masks = causal_anomaly_masks(work, config)
    metadata_invalid = work[["site_id", "primary_use"]].isna().any(axis=1)
    timestamp_invalid = work["timestamp"].isna()
    ambiguous_scale = masks["level_shift"]
    value_invalid = (
        masks["missing"]
        | masks["infinite"]
        | masks["negative"]
        | masks["outlier"]
        | masks["long_zero"]
        | masks["stuck"]
        | ambiguous_scale
    )
    protected = pd.Series(False, index=work.index, dtype=bool)
    if repair_before is not None:
        boundary = pd.Timestamp(repair_before)
        boundary = (
            boundary.tz_localize("UTC")
            if boundary.tzinfo is None
            else boundary.tz_convert("UTC")
        )
        in_training_window = work["timestamp"].lt(boundary)
        protected = ~in_training_window
        value_invalid &= in_training_window
        metadata_invalid &= in_training_window
        timestamp_invalid &= in_training_window
        ambiguous_scale &= in_training_window
    reasons: dict[int, list[str]] = {int(index): [] for index in work.index}
    for name, mask in (
        ("missing_load", masks["missing"]),
        ("infinite_load", masks["infinite"]),
        ("negative_load", masks["negative"]),
        ("robust_outlier", masks["outlier"]),
        ("long_zero_run", masks["long_zero"]),
        ("stuck_sensor", masks["stuck"]),
        ("ambiguous_level_or_unit_shift", ambiguous_scale),
        ("missing_metadata", metadata_invalid),
        ("invalid_timestamp", timestamp_invalid),
    ):
        for index in work.index[mask]:
            reasons[int(index)].append(name)

    original = work["load"].copy()
    work.loc[value_invalid, "load"] = np.nan
    log_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    use_profile = repair_strategy == "forward_then_profile"
    forward_rule = f"past_only_forward_fill_limit_{maximum_forward_fill_hours}h"
    profile_rule = (
        f"past_only_hour_of_week_median_{profile_weeks}w_limit_{maximum_profile_fill_hours}h"
    )

    # Stable ordering preserves same-timestamp order but normal inputs have one key.
    ordered = work.sort_values(["building_id", "timestamp"], kind="stable", na_position="first")
    for _, group in ordered.groupby("building_id", sort=False, dropna=False):
        last_valid: float | None = None
        gap_length = 0
        # Causal hour-of-week history of originally valid values only.
        history: dict[int, deque[float]] = {}
        for index in group.index:
            timestamp = work.at[index, "timestamp"]
            hour_of_week = (
                int(timestamp.dayofweek) * 24 + int(timestamp.hour)
                if pd.notna(timestamp)
                else None
            )
            if protected.loc[index]:
                current_protected = work.at[index, "load"]
                if pd.notna(current_protected) and np.isfinite(current_protected):
                    last_valid = float(current_protected)
                    gap_length = 0
                continue
            invalid_structural = bool(metadata_invalid.loc[index] or timestamp_invalid.loc[index])
            current = work.at[index, "load"]
            was_invalid = bool(value_invalid.loc[index])
            if not was_invalid and not invalid_structural and pd.notna(current) and np.isfinite(current):
                last_valid = float(current)
                gap_length = 0
                if hour_of_week is not None:
                    history.setdefault(hour_of_week, deque(maxlen=profile_weeks)).append(float(current))
                continue
            gap_length += 1
            reason = ";".join(reasons[int(index)]) or "unresolved_quality_issue"
            is_ambiguous = bool(ambiguous_scale.loc[index])
            # Structural problems are never auto-repaired. Ambiguous scale is
            # forward-filled never, and profile-filled only when explicitly allowed.
            can_forward_fill = (
                was_invalid
                and not invalid_structural
                and not is_ambiguous
                and last_valid is not None
                and gap_length <= maximum_forward_fill_hours
            )
            profile_value: float | None = None
            if (
                not can_forward_fill
                and use_profile
                and was_invalid
                and not invalid_structural
                and (not is_ambiguous or profile_fill_ambiguous_scale)
                and gap_length <= maximum_profile_fill_hours
                and hour_of_week is not None
            ):
                values = history.get(hour_of_week)
                if values and len(values) >= 2:
                    profile_value = float(np.median(np.fromiter(values, dtype=float)))
            if can_forward_fill:
                work.at[index, "load"] = last_valid
                status = "repaired"
                rule = forward_rule
                remediated_value: Any = last_valid
            elif profile_value is not None:
                work.at[index, "load"] = profile_value
                status = "repaired"
                rule = profile_rule
                remediated_value = profile_value
            else:
                work.at[index, "load"] = np.nan
                status = "quarantined"
                rule = "preserve_key_with_null"
                remediated_value = None
                quarantine_rows.append(
                    {
                        "timestamp": json_value(work.at[index, "timestamp"]),
                        "building_id": str(work.at[index, "building_id"]),
                        "load": json_value(original.loc[index]),
                        "site_id": json_value(work.at[index, "site_id"]),
                        "primary_use": json_value(work.at[index, "primary_use"]),
                        "condition": "remediated",
                        "reason": reason,
                    }
                )
            log_rows.append(
                {
                    "timestamp": json_value(work.at[index, "timestamp"]),
                    "building_id": str(work.at[index, "building_id"]),
                    "original_value": json_value(original.loc[index]),
                    "remediated_value": json_value(remediated_value),
                    "rule": rule,
                    "status": status,
                    "reason": reason,
                }
            )

    return RemediationResult(
        frame=condition_frame(work, "remediated"),
        log=pd.DataFrame(log_rows, columns=LOG_COLUMNS),
        quarantine=pd.DataFrame(quarantine_rows, columns=QUARANTINE_COLUMNS),
    )
