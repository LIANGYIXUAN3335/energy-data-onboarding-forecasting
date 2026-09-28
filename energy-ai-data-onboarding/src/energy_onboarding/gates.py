"""Quality-gate aggregation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from typing import Any, Iterable, Mapping

from .contracts import IssueRecord, STATUS_RANK


SUPPORTED_ISSUE_CODES = frozenset(
    {
        "SCHEMA_REQUIRED_COLUMN_MISSING",
        "TIMESTAMP_PARSE_ERROR",
        "TIMESTAMP_ORDER_ERROR",
        "DUPLICATE_OBSERVATION_KEY",
        "LOAD_MISSING",
        "LOAD_INFINITE",
        "NEGATIVE_LOAD",
        "ROBUST_OUTLIER",
        "LONG_ZERO_RUN",
        "STUCK_SENSOR",
        "LEVEL_OR_UNIT_SHIFT",
        "METADATA_JOIN_MISSING",
        "TIMEZONE_MISSING",
        "INTERVAL_CONTINUITY_GAP",
        "COVERAGE_RARE_GROUP",
        "SUBGROUP_METADATA_INSTABILITY",
        "SPLIT_BUILDING_COVERAGE",
        "SUBGROUP_SPLIT_COVERAGE",
        "METADATA_DUPLICATE_KEY",
        "METADATA_MISSING_KEY",
        "METADATA_REQUIRED_VALUE_MISSING",
    }
)

# These checks protect the identity, key structure, referential integrity, and
# usable train/evaluation coverage of a published reference dataset. Unlike
# value-quality heuristics, they have no safe numeric tolerance: one affected
# structural unit makes the reference contract ambiguous or unusable. They are
# therefore hard failures and cannot be downgraded by configurable thresholds.
HARD_STRUCTURAL_FAILURE_CODES = frozenset(
    {
        "SCHEMA_REQUIRED_COLUMN_MISSING",
        "TIMESTAMP_PARSE_ERROR",
        "DUPLICATE_OBSERVATION_KEY",
        "METADATA_JOIN_MISSING",
        "METADATA_DUPLICATE_KEY",
        "METADATA_MISSING_KEY",
        "SUBGROUP_METADATA_INSTABILITY",
        "SPLIT_BUILDING_COVERAGE",
    }
)


def gate_status(records: Iterable[IssueRecord]) -> str:
    records = list(records)
    if not records:
        return "pass"
    return max((record.status for record in records), key=STATUS_RANK.__getitem__)


def gate_summary(records: Iterable[IssueRecord]) -> dict:
    records = list(records)
    status_counts = Counter(record.status for record in records)
    hard_failures = sorted(
        {
            record.code
            for record in records
            if record.code in HARD_STRUCTURAL_FAILURE_CODES and record.count > 0
        }
    )
    return {
        "status": gate_status(records),
        "checks": len(records),
        "status_counts": {
            status: int(status_counts.get(status, 0))
            for status in ("pass", "warn", "fail")
        },
        "affected_observations_sum": int(sum(record.count for record in records)),
        "hard_structural_failures": hard_failures,
        "check_results": [
            {
                "issue_code": record.code,
                "status": record.status,
                "count": int(record.count),
                "denominator": int(record.denominator),
                "rate": float(record.rate),
                "threshold": {
                    "basis": record.threshold_basis,
                    "warn_above": record.warn_above,
                    "fail_above": record.fail_above,
                    "comparison": record.threshold_comparison,
                },
                "enforcement": (
                    "hard_structural_count_greater_than_zero"
                    if record.code in HARD_STRUCTURAL_FAILURE_CODES
                    else "configurable_or_detector_default"
                ),
            }
            for record in records
        ],
    }


def hard_structural_failures(records: Iterable[IssueRecord]) -> list[IssueRecord]:
    """Return nonzero structural failures that must always block publication."""

    return [
        record
        for record in records
        if record.code in HARD_STRUCTURAL_FAILURE_CODES and record.count > 0
    ]


def apply_check_thresholds(
    records: Iterable[IssueRecord], gate_config: Mapping[str, Any] | None
) -> list[IssueRecord]:
    """Apply optional, explicit per-check count/rate thresholds.

    Rules live under ``gate.check_thresholds`` and are keyed by issue code::

        {"LOAD_MISSING": {"basis": "rate", "warn_above": 0, "fail_above": 0.05}}

    A configured rule replaces that check's built-in nonzero severity except
    for ``HARD_STRUCTURAL_FAILURE_CODES``. Those checks remain ``fail`` whenever
    their observed count is nonzero. Checks without a rule retain their
    detector-defined status.
    """

    config = dict(gate_config or {})
    raw_rules = config.get("check_thresholds", {})
    if raw_rules is None:
        raw_rules = {}
    if not isinstance(raw_rules, Mapping):
        raise ValueError("gate.check_thresholds must be an object keyed by issue code.")
    unknown = sorted(set(map(str, raw_rules)).difference(SUPPORTED_ISSUE_CODES))
    if unknown:
        raise ValueError(
            "gate.check_thresholds contains unknown issue codes: " f"{unknown}"
        )

    validated_rules: dict[str, tuple[str, float | None, float | None]] = {}
    for issue_code, raw_rule in raw_rules.items():
        if not isinstance(raw_rule, Mapping):
            raise ValueError(f"Threshold rule for {issue_code} must be an object.")
        basis = str(raw_rule.get("basis", "rate"))
        if basis not in {"count", "rate"}:
            raise ValueError(
                f"Threshold basis for {issue_code} must be 'count' or 'rate'."
            )
        warn_above = raw_rule.get("warn_above")
        fail_above = raw_rule.get("fail_above")
        normalized: dict[str, float | None] = {}
        for label, threshold in (
            ("warn_above", warn_above),
            ("fail_above", fail_above),
        ):
            if threshold is None:
                normalized[label] = None
                continue
            try:
                threshold_value = float(threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{issue_code}.{label} must be a finite number."
                ) from exc
            if not math.isfinite(threshold_value):
                raise ValueError(
                    f"{issue_code}.{label} must be a finite number."
                )
            if threshold_value < 0:
                raise ValueError(f"{issue_code}.{label} must be non-negative.")
            if basis == "rate" and threshold_value > 1:
                raise ValueError(
                    f"{issue_code}.{label} rate threshold must be between 0 and 1."
                )
            normalized[label] = threshold_value
        normalized_warn = normalized["warn_above"]
        normalized_fail = normalized["fail_above"]
        if normalized_warn is not None and normalized_fail is not None:
            if normalized_fail < normalized_warn:
                raise ValueError(
                    f"{issue_code}.fail_above must be >= warn_above."
                )
        validated_rules[str(issue_code)] = (basis, normalized_warn, normalized_fail)

    evaluated: list[IssueRecord] = []
    for record in records:
        validated_rule = validated_rules.get(record.code)
        is_hard_failure = (
            record.code in HARD_STRUCTURAL_FAILURE_CODES and record.count > 0
        )
        if validated_rule is None:
            evaluated.append(
                replace(
                    record,
                    status="fail",  # type: ignore[arg-type]
                    threshold_comparison="hard_structural_count_greater_than_zero",
                )
                if is_hard_failure
                else record
            )
            continue
        basis, normalized_warn, normalized_fail = validated_rule
        value = float(record.count if basis == "count" else record.rate)
        status = "pass"
        if normalized_fail is not None and value > normalized_fail:
            status = "fail"
        elif normalized_warn is not None and value > normalized_warn:
            status = "warn"
        if is_hard_failure:
            status = "fail"
        evaluated.append(
            replace(
                record,
                status=status,  # type: ignore[arg-type]
                threshold_basis=basis,
                warn_above=normalized_warn,
                fail_above=normalized_fail,
                threshold_comparison=(
                    "hard_structural_count_greater_than_zero"
                    if is_hard_failure
                    else "strictly_greater_than"
                ),
            )
        )
    return evaluated
