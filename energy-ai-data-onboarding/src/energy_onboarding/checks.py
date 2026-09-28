"""Configuration-driven data-quality and integrity checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import IssueRecord


@dataclass(frozen=True)
class QualityConfig:
    frequency: str = "1h"
    long_zero_threshold: int = 6
    stuck_threshold: int = 8
    outlier_mad_z: float = 12.0
    level_shift_ratio: float = 8.0
    causal_window: int = 168
    rare_group_fraction: float = 0.01
    # Run-detector policy. The defaults reproduce the v2 behaviour exactly.
    #
    # ``run_flagging``:
    #   "tail"      - only run positions at or beyond the threshold are flagged
    #                 (v2). The head of a long run stays "valid", which lets a
    #                 forward fill copy the run's own value back into it.
    #   "whole_run" - once a run reaches the threshold every member of that
    #                 run is flagged (v3). A run of length L is established by
    #                 hour L; no value later than the run is consulted.
    # ``stuck_calibration_quantile``: when set, each building's stuck-sensor
    #   threshold becomes max(stuck_threshold, ceil(q-quantile of that
    #   building's natural constant-run lengths before ``calibration_end``) + 1).
    #   Coarsely quantized meters, whose readings repeat for hours as a matter
    #   of resolution, therefore stop being flagged as stuck; high-resolution
    #   meters keep the configured threshold.
    # ``zero_profile_min_ratio``: when set, a zero run is flagged only during
    #   hours when the building is normally active, i.e. when the causal
    #   hour-of-week median over the previous ``profile_weeks`` weeks exceeds
    #   this ratio times the building's causal 168-hour median. Buildings that
    #   are genuinely idle at night or at weekends are no longer quarantined.
    # ``calibration_end``: rows at or after this timestamp never contribute to
    #   a building's calibration statistics (the fault cutoff is used).
    run_flagging: str = "tail"
    stuck_calibration_quantile: float | None = None
    zero_profile_min_ratio: float | None = None
    profile_weeks: int = 8
    calibration_end: str | None = None
    # Per-building stuck thresholds already calibrated on the reference
    # condition. When present they override the quantile computation so that
    # corrupted/remediated inspections never calibrate on injected faults.
    stuck_thresholds_by_building: Mapping[str, int] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.run_flagging not in {"tail", "whole_run"}:
            raise ValueError("run_flagging must be 'tail' or 'whole_run'.")
        if self.stuck_calibration_quantile is not None and not (
            0 < float(self.stuck_calibration_quantile) <= 1
        ):
            raise ValueError("stuck_calibration_quantile must be in (0, 1].")
        if self.zero_profile_min_ratio is not None and float(self.zero_profile_min_ratio) < 0:
            raise ValueError("zero_profile_min_ratio must be non-negative.")
        if int(self.profile_weeks) < 1:
            raise ValueError("profile_weeks must be at least 1.")


def _sample(frame: pd.DataFrame, mask: pd.Series, columns: list[str]) -> tuple[dict, ...]:
    available = [column for column in columns if column in frame.columns]
    return tuple(frame.loc[mask, available].head(5).to_dict(orient="records"))


def _record(
    *,
    code: str,
    check: str,
    count: int,
    denominator: int,
    nonzero_status: str,
    repairable: bool,
    columns: tuple[str, ...],
    sample: tuple[dict, ...] = (),
    noun: str = "rows",
    expected_rule: str = "",
    action: str = "none",
    source_condition: str = "natural",
) -> IssueRecord:
    status = nonzero_status if count else "pass"
    return IssueRecord(
        code=code,
        check=check,
        status=status,  # type: ignore[arg-type]
        count=int(count),
        denominator=int(denominator),
        rate=float(count / max(denominator, 1)),
        message=(
            f"{count} {noun} violate {check}."
            if count
            else f"No {noun} violate {check}."
        ),
        repairable=repairable,
        columns=columns,
        sample=sample,
        expected_rule=expected_rule,
        action=action,
        source_condition=source_condition,
    )


def _causal_run_position(predicate: pd.Series) -> pd.Series:
    run_id = (~predicate).cumsum()
    return predicate.groupby(run_id, sort=False).cumsum()


def _run_lengths(predicate: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (run_id, run_length) where run_length is the size of the True run."""

    run_id = (~predicate).cumsum()
    length = predicate.groupby(run_id, sort=False).transform("sum")
    return run_id, length.where(predicate, 0).astype(int)


def _calibration_boundary(config: QualityConfig) -> pd.Timestamp | None:
    if config.calibration_end is None:
        return None
    boundary = pd.Timestamp(config.calibration_end)
    return boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")


def hour_of_week_profile(
    loads: pd.Series, timestamps: pd.Series, weeks: int
) -> pd.Series:
    """Causal hour-of-week median: same weekday/hour over the previous ``weeks``.

    Only observations strictly earlier than each row contribute, so the profile
    is available at issuance time and never reads a later value.
    """

    hour_of_week = timestamps.dt.dayofweek * 24 + timestamps.dt.hour
    profile = pd.Series(np.nan, index=loads.index, dtype=float)
    for _, positions in hour_of_week.groupby(hour_of_week, sort=False).groups.items():
        history = loads.loc[positions]
        profile.loc[positions] = (
            history.shift(1).rolling(weeks, min_periods=2).median().to_numpy()
        )
    return profile


def _anomaly_masks_with_calibration(
    frame: pd.DataFrame, config: QualityConfig
) -> tuple[dict[str, pd.Series], list[dict[str, Any]]]:
    names = (
        "missing",
        "infinite",
        "negative",
        "outlier",
        "long_zero",
        "stuck",
        "level_shift",
    )
    masks = {name: pd.Series(False, index=frame.index, dtype=bool) for name in names}
    calibration: list[dict[str, Any]] = []
    if "load" not in frame or "building_id" not in frame or "timestamp" not in frame:
        return masks, calibration

    numeric = pd.to_numeric(frame["load"], errors="coerce")
    masks["missing"] = numeric.isna()
    masks["infinite"] = pd.Series(np.isinf(numeric), index=frame.index)
    masks["negative"] = numeric.lt(0).fillna(False)

    boundary = _calibration_boundary(config)
    valid = frame.loc[frame["timestamp"].notna()].copy()
    valid["_load_numeric"] = numeric.loc[valid.index]
    valid = valid.sort_values(["building_id", "timestamp"], kind="stable")
    for building_id, group in valid.groupby("building_id", sort=False, dropna=False):
        loads = group["_load_numeric"].astype(float)
        timestamps = pd.to_datetime(group["timestamp"], utc=True)
        prior = loads.shift(1)
        minimum = min(6, max(1, config.causal_window))
        median = prior.rolling(config.causal_window, min_periods=minimum).median()
        absolute_deviation = prior.sub(median).abs()
        mad = absolute_deviation.rolling(
            config.causal_window, min_periods=minimum
        ).median()
        robust_z = loads.sub(median).abs().divide(1.4826 * mad.replace(0, np.nan))
        ratio_outlier = median.gt(0) & (
            loads.gt(median * config.level_shift_ratio)
            | loads.lt(median / config.level_shift_ratio)
        )
        masks["outlier"].loc[group.index] = (
            robust_z.gt(config.outlier_mad_z) | ratio_outlier
        ).fillna(False)

        in_calibration = (
            timestamps.lt(boundary) if boundary is not None else pd.Series(True, index=group.index)
        )
        is_zero = loads.eq(0)
        same_as_prior = loads.eq(loads.shift(1)) & loads.notna() & ~is_zero
        # A constant run starts whenever the value changes; its length counts
        # every member including the first reading. Only finite, non-zero runs
        # of at least two readings are candidates for the stuck-sensor rule.
        value_changed = ~loads.eq(loads.shift(1)) | loads.isna()
        stuck_run_id = value_changed.cumsum()
        run_size = loads.groupby(stuck_run_id, sort=False).transform("size")
        stuck_member = loads.notna() & ~is_zero & run_size.ge(2)
        stuck_length = run_size.where(stuck_member, 0).astype(int)

        effective_stuck = int(config.stuck_threshold)
        run_count = 0
        run_quantile = None
        calibration_source = "configured"
        if config.stuck_calibration_quantile is not None:
            natural = stuck_member & in_calibration
            run_sizes = natural.groupby(stuck_run_id, sort=False).sum()
            run_sizes = run_sizes[run_sizes.ge(2)]
            run_count = int(len(run_sizes))
            if run_count:
                run_quantile = float(
                    np.quantile(run_sizes.to_numpy(dtype=float), float(config.stuck_calibration_quantile))
                )
        fixed = (
            config.stuck_thresholds_by_building.get(str(building_id))
            if config.stuck_thresholds_by_building is not None
            else None
        )
        if fixed is not None:
            effective_stuck = max(effective_stuck, int(fixed))
            calibration_source = "reference_calibration"
        elif run_quantile is not None:
            effective_stuck = max(effective_stuck, int(np.ceil(run_quantile)) + 1)
            calibration_source = "quantile_on_inspected_frame"

        if config.run_flagging == "whole_run":
            stuck = stuck_member & stuck_length.ge(effective_stuck)
        else:
            same_position = _causal_run_position(same_as_prior) + 1
            stuck = same_as_prior & same_position.ge(effective_stuck) & ~is_zero
        masks["stuck"].loc[group.index] = stuck.fillna(False)

        zero_run_id, zero_length = _run_lengths(is_zero)
        if config.run_flagging == "whole_run":
            long_zero = is_zero & zero_length.ge(config.long_zero_threshold)
        else:
            zero_position = _causal_run_position(is_zero)
            long_zero = is_zero & zero_position.ge(config.long_zero_threshold)
        profile_available = None
        if config.zero_profile_min_ratio is not None:
            profile = hour_of_week_profile(loads, timestamps, int(config.profile_weeks))
            normally_active = profile.gt(float(config.zero_profile_min_ratio) * median)
            profile_available = profile.notna() & median.notna()
            # A run is exempt only when every member hour with a profile is a
            # normally idle hour; if any member hour is normally active the whole
            # run stays flagged, so a run is never split into flagged and
            # unflagged members (which would let a forward fill copy the run's
            # own zero back into it). Hours without a profile follow the plain
            # rule for the run.
            # zero_run_id also covers the non-zero reading that precedes each
            # run, so restrict the vote to zero members only.
            row_keep = (~profile_available | normally_active) & is_zero
            run_keep = row_keep.groupby(zero_run_id, sort=False).transform("max")
            long_zero = long_zero & run_keep
        masks["long_zero"].loc[group.index] = long_zero.fillna(False)

        short_prior = prior.rolling(24, min_periods=6).median()
        shift = short_prior.gt(0) & (
            loads.gt(short_prior * config.level_shift_ratio)
            | loads.lt(short_prior / config.level_shift_ratio)
        )
        masks["level_shift"].loc[group.index] = shift.fillna(False)

        calibrated_rows = int(in_calibration.sum())
        calibration.append(
            {
                "building_id": str(building_id),
                "calibration_rows": calibrated_rows,
                "repeat_rate": (
                    float(same_as_prior.loc[in_calibration].mean()) if calibrated_rows else None
                ),
                "zero_rate": float(is_zero.loc[in_calibration].mean()) if calibrated_rows else None,
                "constant_run_count": run_count,
                "constant_run_length_quantile": run_quantile,
                "stuck_threshold_configured": int(config.stuck_threshold),
                "stuck_threshold_effective": int(effective_stuck),
                "calibration_source": calibration_source,
                "long_zero_threshold": int(config.long_zero_threshold),
                "zero_profile_min_ratio": config.zero_profile_min_ratio,
                "profile_weeks": int(config.profile_weeks),
                "run_flagging": config.run_flagging,
                "zero_rows_exempted_by_profile": (
                    int(
                        (
                            is_zero
                            & zero_length.ge(config.long_zero_threshold)
                            & ~long_zero
                        ).sum()
                    )
                    if profile_available is not None
                    else 0
                ),
            }
        )
    return masks, calibration


def causal_anomaly_masks(
    frame: pd.DataFrame, config: QualityConfig
) -> dict[str, pd.Series]:
    """Return anomaly masks computed without future observations.

    Run-based flags (long zero runs, stuck readings) are assigned to a run once
    its length is known; no reading later than the run itself is consulted.
    """

    masks, _ = _anomaly_masks_with_calibration(frame, config)
    return masks


def detector_calibration_table(
    frame: pd.DataFrame, config: QualityConfig
) -> pd.DataFrame:
    """Per-building run-detector calibration actually applied to ``frame``."""

    _, calibration = _anomaly_masks_with_calibration(frame, config)
    columns = [
        "building_id",
        "calibration_rows",
        "repeat_rate",
        "zero_rate",
        "constant_run_count",
        "constant_run_length_quantile",
        "stuck_threshold_configured",
        "stuck_threshold_effective",
        "calibration_source",
        "long_zero_threshold",
        "zero_profile_min_ratio",
        "profile_weeks",
        "run_flagging",
        "zero_rows_exempted_by_profile",
    ]
    return pd.DataFrame(calibration, columns=columns)


def _continuity_gaps(
    frame: pd.DataFrame, frequency: str
) -> tuple[int, tuple[dict[str, Any], ...]]:
    gap_count = 0
    examples: list[dict[str, Any]] = []
    valid = frame.loc[frame["timestamp"].notna(), ["building_id", "timestamp"]]
    valid = valid.drop_duplicates()
    for building_id, group in valid.groupby("building_id", sort=True, dropna=False):
        timestamps = pd.DatetimeIndex(group["timestamp"].sort_values().unique())
        if timestamps.empty:
            continue
        expected = pd.date_range(timestamps.min(), timestamps.max(), freq=frequency)
        missing = expected.difference(timestamps)
        gap_count += len(missing)
        for timestamp in missing[: max(0, 5 - len(examples))]:
            examples.append(
                {"building_id": str(building_id), "timestamp": timestamp.isoformat()}
            )
    return gap_count, tuple(examples)


def _ordering_violations(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for _, group in frame.groupby("building_id", sort=False, dropna=False):
        valid = group["timestamp"].notna()
        prior = group.loc[valid, "timestamp"].shift(1)
        current = group.loc[valid, "timestamp"]
        mask.loc[current.index] = current.lt(prior).fillna(False)
    return mask


def run_quality_checks(
    frame: pd.DataFrame,
    config: QualityConfig | None = None,
    *,
    source_condition: str = "natural",
) -> list[IssueRecord]:
    """Run all checks and retain passing as well as adverse outcomes."""

    config = config or QualityConfig()
    required = {"timestamp", "building_id", "load", "site_id", "primary_use"}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        return [
            IssueRecord(
                code="SCHEMA_REQUIRED_COLUMN_MISSING",
                check="required output schema",
                status="fail",
                count=len(missing_columns),
                denominator=len(required),
                rate=len(missing_columns) / len(required),
                message=f"Missing required columns: {missing_columns}",
                repairable=False,
                columns=tuple(missing_columns),
                expected_rule="All required contract columns are present.",
                action="stop_publication",
                source_condition=source_condition,
            )
        ]

    rows = len(frame)
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    normalized = frame.assign(timestamp=timestamps)
    timestamp_missing = timestamps.isna()
    duplicate = normalized.duplicated(["timestamp", "building_id"], keep=False) & ~timestamp_missing
    ordering = _ordering_violations(normalized)
    metadata_missing = normalized[["site_id", "primary_use"]].isna().any(axis=1)
    timezone_missing = (
        normalized["timezone"].isna()
        if "timezone" in normalized
        else pd.Series(True, index=normalized.index)
    )
    anomalies = causal_anomaly_masks(normalized, config)
    gaps, gap_sample = _continuity_gaps(normalized, config.frequency)

    specifications = [
        ("TIMESTAMP_PARSE_ERROR", "timestamp parseability", timestamp_missing, "fail", False, ("timestamp",), "RFC 3339-compatible timestamp", "quarantine"),
        ("TIMESTAMP_ORDER_ERROR", "monotonic per-building timestamps", ordering, "fail", True, ("timestamp",), "nondecreasing per-building timestamps", "stable_sort"),
        ("DUPLICATE_OBSERVATION_KEY", "unique timestamp/building key", duplicate, "fail", True, ("timestamp", "building_id"), "one row per timestamp/building_id", "aggregate_duplicate"),
        ("LOAD_MISSING", "non-missing load", anomalies["missing"], "warn", True, ("load",), "finite numeric load or documented null", "past_only_fill_or_quarantine"),
        ("LOAD_INFINITE", "finite load", anomalies["infinite"], "fail", True, ("load",), "finite numeric load", "convert_to_null_then_repair"),
        ("NEGATIVE_LOAD", "non-negative load", anomalies["negative"], "fail", True, ("load",), "load >= 0 unless source exception is documented", "convert_to_null_then_repair"),
        ("ROBUST_OUTLIER", "robust causal load range", anomalies["outlier"], "warn", True, ("load",), f"causal median/MAD z <= {config.outlier_mad_z:g}", "review_or_past_only_repair"),
        ("LONG_ZERO_RUN", "short zero run", anomalies["long_zero"], "warn", True, ("load",), f"fewer than {config.long_zero_threshold} consecutive zeros", "past_only_repair_or_quarantine"),
        ("STUCK_SENSOR", "changing sensor values", anomalies["stuck"], "warn", True, ("load",), f"fewer than {config.stuck_threshold} repeated nonzero values", "past_only_repair_or_quarantine"),
        ("LEVEL_OR_UNIT_SHIFT", "stable causal level", anomalies["level_shift"], "warn", False, ("load",), f"within {config.level_shift_ratio:g}x recent median", "quarantine_ambiguous_scale"),
        ("METADATA_JOIN_MISSING", "complete metadata join", metadata_missing, "fail", False, ("site_id", "primary_use"), "one known site and primary use per building", "quarantine"),
        ("TIMEZONE_MISSING", "documented source timezone", timezone_missing, "warn", False, ("timezone",), "timezone supplied by source metadata", "document_utc_normalization_caveat"),
    ]
    records: list[IssueRecord] = []
    for code, name, mask, severity, repairable, columns, rule, action in specifications:
        records.append(
            _record(
                code=code,
                check=name,
                count=int(mask.sum()),
                denominator=rows,
                nonzero_status=severity,
                repairable=repairable,
                columns=columns,
                sample=_sample(
                    normalized,
                    mask,
                    ["building_id", "timestamp", "load", "site_id", "primary_use"],
                ),
                expected_rule=rule,
                action=action,
                source_condition=source_condition,
            )
        )
    records.append(
        _record(
            code="INTERVAL_CONTINUITY_GAP",
            check=f"continuous {config.frequency} observations",
            count=gaps,
            denominator=rows + gaps,
            nonzero_status="warn",
            repairable=True,
            columns=("timestamp",),
            sample=gap_sample,
            noun="expected intervals",
            expected_rule=f"one observation every {config.frequency} per building",
            action="preserve_key_as_null_when_materialized",
            source_condition=source_condition,
        )
    )

    rare_row_mask = pd.Series(False, index=normalized.index, dtype=bool)
    if rows:
        for column in ("site_id", "primary_use"):
            shares = normalized[column].value_counts(dropna=True) / rows
            rare_values = shares.loc[shares.lt(config.rare_group_fraction)].index
            rare_row_mask |= normalized[column].isin(rare_values)
    records.append(
        _record(
            code="COVERAGE_RARE_GROUP",
            check=f"group coverage at least {config.rare_group_fraction:.1%}",
            count=int(rare_row_mask.sum()),
            denominator=rows,
            nonzero_status="warn",
            repairable=False,
            columns=("site_id", "primary_use"),
            sample=_sample(normalized, rare_row_mask, ["building_id", "site_id", "primary_use"]),
            expected_rule="report representation gaps without fairness claims",
            action="report",
            source_condition=source_condition,
        )
    )

    building_count = int(normalized["building_id"].nunique(dropna=True))
    unstable_buildings: set[str] = set()
    for column in ("site_id", "primary_use", "timezone"):
        if column not in normalized:
            continue
        cardinality = normalized.groupby("building_id", dropna=True)[column].nunique(
            dropna=True
        )
        unstable_buildings.update(cardinality.loc[cardinality.gt(1)].index.astype(str))
    unstable_mask = normalized["building_id"].astype(str).isin(unstable_buildings)
    records.append(
        _record(
            code="SUBGROUP_METADATA_INSTABILITY",
            check="stable subgroup metadata per building",
            count=len(unstable_buildings),
            denominator=building_count,
            nonzero_status="fail",
            repairable=False,
            columns=("building_id", "site_id", "primary_use", "timezone"),
            sample=_sample(
                normalized,
                unstable_mask,
                ["building_id", "site_id", "primary_use", "timezone"],
            ),
            noun="buildings",
            expected_rule=(
                "each building maps to one site, primary-use category, and source timezone"
            ),
            action="stop_publication_and_review_metadata",
            source_condition=source_condition,
        )
    )
    return records


def split_boundary_checks(
    frame: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    *,
    source_condition: str = "natural",
) -> list[IssueRecord]:
    """Check that buildings and reported subgroups occur on both sides of a split.

    BDG2 timestamps are source-local wall-clock values represented with a UTC
    marker by this prototype.  These checks therefore compare values only in
    that stored representation; they do not claim timezone harmonization.
    """

    required = {"timestamp", "building_id", "site_id", "primary_use"}
    if not required.issubset(frame.columns):
        return []
    boundary = pd.Timestamp(cutoff)
    boundary = (
        boundary.tz_localize("UTC")
        if boundary.tzinfo is None
        else boundary.tz_convert("UTC")
    )
    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(
        normalized["timestamp"], errors="coerce", utc=True
    )
    normalized["building_id"] = normalized["building_id"].astype("string")
    valid = normalized.loc[
        normalized["timestamp"].notna() & normalized["building_id"].notna()
    ].copy()
    before = valid.loc[valid["timestamp"].lt(boundary)]
    after = valid.loc[valid["timestamp"].ge(boundary)]

    all_buildings = set(valid["building_id"].astype(str))
    before_buildings = set(before["building_id"].astype(str))
    after_buildings = set(after["building_id"].astype(str))
    missing_buildings = sorted(
        all_buildings.difference(before_buildings.intersection(after_buildings))
    )
    building_sample = tuple(
        {
            "building_id": building_id,
            "observed_value": (
                "missing_before_boundary"
                if building_id not in before_buildings
                else "missing_at_or_after_boundary"
            ),
        }
        for building_id in missing_buildings[:5]
    )

    subgroup_columns = ["site_id", "primary_use"]
    if "timezone" in valid:
        subgroup_columns.append("timezone")

    def subgroup_keys(values: pd.DataFrame) -> set[tuple[str, ...]]:
        if values.empty:
            return set()
        selected = values[subgroup_columns].fillna("<missing>").astype(str)
        return {tuple(row) for row in selected.drop_duplicates().itertuples(index=False, name=None)}

    all_groups = subgroup_keys(valid)
    before_groups = subgroup_keys(before)
    after_groups = subgroup_keys(after)
    missing_groups = sorted(all_groups.difference(before_groups.intersection(after_groups)))
    group_sample = tuple(
        {
            "observed_value": dict(zip(subgroup_columns, group)),
        }
        for group in missing_groups[:5]
    )
    semantics = (
        f"both sides of {boundary.isoformat()} in the stored source-local "
        "wall-clock representation"
    )
    return [
        IssueRecord(
            code="SPLIT_BUILDING_COVERAGE",
            check="building coverage across split boundary",
            status="fail" if missing_buildings else "pass",
            count=len(missing_buildings),
            denominator=len(all_buildings),
            rate=len(missing_buildings) / max(len(all_buildings), 1),
            message=(
                f"{len(missing_buildings)} buildings lack observations on both sides "
                "of the configured boundary."
            ),
            repairable=False,
            columns=("timestamp", "building_id"),
            sample=building_sample,
            expected_rule=semantics,
            action="stop_publication_or_change_the_prospectively_defined_scope",
            source_condition=source_condition,
        ),
        IssueRecord(
            code="SUBGROUP_SPLIT_COVERAGE",
            check="reported subgroup coverage across split boundary",
            status="warn" if missing_groups else "pass",
            count=len(missing_groups),
            denominator=len(all_groups),
            rate=len(missing_groups) / max(len(all_groups), 1),
            message=(
                f"{len(missing_groups)} reported site/use/timezone groups lack "
                "observations on both sides of the configured boundary."
            ),
            repairable=False,
            columns=tuple(subgroup_columns),
            sample=group_sample,
            expected_rule=semantics,
            action="report_coverage_limitation",
            source_condition=source_condition,
        ),
    ]


def coverage_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return representation and temporal coverage, not a fairness conclusion."""

    valid_timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    month = valid_timestamps.dt.month
    season = pd.Series("unknown", index=frame.index, dtype="string")
    season.loc[month.isin([12, 1, 2])] = "winter"
    season.loc[month.isin([3, 4, 5])] = "spring"
    season.loc[month.isin([6, 7, 8])] = "summer"
    season.loc[month.isin([9, 10, 11])] = "autumn"

    def counts(series: pd.Series) -> dict[str, int]:
        values = series.fillna("<missing>").astype(str).value_counts().sort_index()
        return {str(key): int(value) for key, value in values.items()}

    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "buildings": int(frame["building_id"].nunique(dropna=True)),
        "timestamp_min": valid_timestamps.min().isoformat() if valid_timestamps.notna().any() else None,
        "timestamp_max": valid_timestamps.max().isoformat() if valid_timestamps.notna().any() else None,
        "by_site": counts(frame["site_id"]),
        "by_primary_use": counts(frame["primary_use"]),
        "by_season": counts(season),
        "load_missing": int(pd.to_numeric(frame["load"], errors="coerce").isna().sum()),
    }
    if "timezone" in frame:
        result["by_timezone"] = counts(frame["timezone"])
    return result


def validate_fault_cutoff(
    faults: list[dict[str, Any]], cutoff: str | pd.Timestamp
) -> list[str]:
    """Return faults that violate the strict pre-cutoff training boundary."""

    boundary = pd.Timestamp(cutoff)
    boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
    violations: list[str] = []
    for fault in faults:
        timestamp = pd.to_datetime(fault.get("timestamp"), errors="coerce", utc=True)
        if pd.isna(timestamp) or timestamp >= boundary:
            violations.append(
                f"{fault.get('fault_type')}:{fault.get('building_id')}:{fault.get('timestamp')}"
            )
    return violations
