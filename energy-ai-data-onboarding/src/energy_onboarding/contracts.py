"""Shared data contracts and serializable quality records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd


Condition = Literal["reference", "corrupted", "remediated"]
Status = Literal["pass", "warn", "fail"]

OUTPUT_COLUMNS = [
    "timestamp",
    "building_id",
    "load",
    "site_id",
    "primary_use",
    "condition",
]

STATUS_RANK: dict[str, int] = {"pass": 0, "warn": 1, "fail": 2}


def json_value(value: Any) -> Any:
    """Convert pandas/NumPy values into strict JSON-compatible values."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class IssueRecord:
    """One quality-check outcome, including successful checks."""

    code: str
    check: str
    status: Status
    count: int
    denominator: int
    rate: float
    message: str
    repairable: bool
    columns: tuple[str, ...] = ()
    sample: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    expected_rule: str = ""
    action: str = "none"
    source_condition: str = "natural"
    seeded_fault: bool = False
    threshold_basis: str | None = None
    warn_above: float | None = None
    fail_above: float | None = None
    threshold_comparison: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_value(asdict(self))


def condition_frame(frame: pd.DataFrame, condition: Condition) -> pd.DataFrame:
    """Return a stable, typed frame that satisfies the downstream contract."""

    result = frame.copy()
    result["condition"] = condition
    for column in OUTPUT_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
    result["building_id"] = result["building_id"].astype("string")
    result["load"] = pd.to_numeric(result["load"], errors="coerce").astype(float)
    result["site_id"] = result["site_id"].astype("string")
    result["primary_use"] = result["primary_use"].astype("string")
    result["condition"] = result["condition"].astype("string")
    return result.loc[:, OUTPUT_COLUMNS].sort_values(
        ["timestamp", "building_id"], kind="stable", na_position="first"
    ).reset_index(drop=True)


def validate_condition_frame(
    frame: pd.DataFrame,
    *,
    expected_condition: Condition | None = None,
    require_unique_keys: bool = True,
) -> list[str]:
    """Return contract errors for a published forecasting condition."""

    errors: list[str] = []
    if list(frame.columns) != OUTPUT_COLUMNS:
        errors.append(
            f"columns must be exactly {OUTPUT_COLUMNS}; observed {list(frame.columns)}"
        )
        return errors
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any():
        errors.append(f"{int(timestamps.isna().sum())} timestamps are not parseable")
    if frame["building_id"].isna().any() or frame["building_id"].astype(str).str.strip().eq("").any():
        errors.append("building_id must be non-empty")
    if require_unique_keys and frame.duplicated(["timestamp", "building_id"]).any():
        errors.append("timestamp/building_id keys must be unique")
    loads = pd.to_numeric(frame["load"], errors="coerce")
    non_null = frame["load"].notna()
    if (loads.isna() & non_null).any():
        errors.append("non-null load values must be numeric")
    if np.isinf(loads.dropna()).any():
        errors.append("load values must be finite or null")
    observed_conditions = set(frame["condition"].dropna().astype(str).unique())
    if expected_condition is not None and observed_conditions != {expected_condition}:
        errors.append(
            f"condition must be only {expected_condition!r}; observed "
            f"{sorted(observed_conditions)}"
        )
    return errors


def issue_records_frame(records: list[IssueRecord]) -> pd.DataFrame:
    """Convert check outcomes into the stable machine-readable issue schema."""

    columns = [
        "check_name",
        "issue_code",
        "severity",
        "timestamp",
        "building_id",
        "observed_value",
        "expected_rule",
        "action",
        "source_condition",
        "seeded_fault",
        "count",
        "denominator",
        "rate",
        "repairable",
        "message",
        "threshold_basis",
        "warn_above",
        "fail_above",
        "threshold_comparison",
    ]
    rows: list[dict[str, Any]] = []
    for record in records:
        samples = record.sample or ({},)
        for index, sample in enumerate(samples):
            rows.append(
                {
                    "check_name": record.check,
                    "issue_code": record.code,
                    "severity": record.status,
                    "timestamp": json_value(sample.get("timestamp")),
                    "building_id": json_value(sample.get("building_id")),
                    "observed_value": json_value(
                        sample.get("load", sample.get("observed_value"))
                    ),
                    "expected_rule": record.expected_rule or record.message,
                    "action": record.action,
                    "source_condition": record.source_condition,
                    "seeded_fault": bool(record.seeded_fault),
                    "count": record.count if index == 0 else None,
                    "denominator": record.denominator if index == 0 else None,
                    "rate": record.rate if index == 0 else None,
                    "repairable": record.repairable,
                    "message": record.message,
                    "threshold_basis": record.threshold_basis,
                    "warn_above": record.warn_above,
                    "fail_above": record.fail_above,
                    "threshold_comparison": record.threshold_comparison,
                }
            )
    return pd.DataFrame(rows, columns=columns)
