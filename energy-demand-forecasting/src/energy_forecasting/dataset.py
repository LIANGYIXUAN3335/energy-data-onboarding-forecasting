from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "building_id",
    "load",
    "site_id",
    "primary_use",
}


@dataclass(frozen=True)
class SplitBoundary:
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_condition(path: str | Path, expected_condition: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    # Canonicalize for deterministic key comparison. This does not imply that
    # BDG2 source-local wall-clock observations were offset-converted to true
    # UTC; timestamp semantics remain a producer-manifest responsibility.
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if frame["timestamp"].isna().any():
        count = int(frame["timestamp"].isna().sum())
        raise ValueError(f"{path} contains {count} unparseable timestamps")
    if frame["building_id"].isna().any() or (
        frame["building_id"].astype(str).str.strip() == ""
    ).any():
        raise ValueError(f"{path} contains missing or empty building_id values")
    frame["building_id"] = frame["building_id"].astype("string")
    frame["site_id"] = frame["site_id"].fillna("unknown").astype("string")
    frame["primary_use"] = frame["primary_use"].fillna("unknown").astype("string")
    frame["load"] = pd.to_numeric(frame["load"], errors="coerce")

    duplicate = frame.duplicated(["timestamp", "building_id"], keep=False)
    if duplicate.any():
        raise ValueError(
            f"{path} contains {int(duplicate.sum())} duplicate timestamp/building rows"
        )

    if "condition" not in frame.columns:
        frame["condition"] = expected_condition or Path(path).stem.split(".")[0]
    if expected_condition is not None:
        if frame["condition"].isna().any():
            raise ValueError(f"{path} contains null condition values")
        values = set(frame["condition"].dropna().astype(str).unique())
        if values and values != {expected_condition}:
            raise ValueError(
                f"{path} condition values {sorted(values)} do not match {expected_condition!r}"
            )
        frame["condition"] = expected_condition

    return frame.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def choose_split_boundaries(
    reference: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> SplitBoundary:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train_fraction and validation_fraction must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")

    times = pd.Index(reference["timestamp"].drop_duplicates().sort_values())
    if len(times) < 10:
        raise ValueError("At least 10 unique timestamps are required")
    train_index = max(0, min(len(times) - 3, int(len(times) * train_fraction) - 1))
    validation_index = max(
        train_index + 1,
        min(len(times) - 2, int(len(times) * (train_fraction + validation_fraction)) - 1),
    )
    return SplitBoundary(
        train_end=pd.Timestamp(times[train_index]),
        validation_end=pd.Timestamp(times[validation_index]),
        test_end=pd.Timestamp(times[-1]),
    )


def choose_explicit_split_boundaries(
    reference: pd.DataFrame,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> SplitBoundary:
    """Validate pre-specified calendar boundaries against the reference data."""
    train_timestamp = pd.Timestamp(train_end)
    validation_timestamp = pd.Timestamp(validation_end)
    if train_timestamp.tzinfo is None:
        train_timestamp = train_timestamp.tz_localize("UTC")
    else:
        train_timestamp = train_timestamp.tz_convert("UTC")
    if validation_timestamp.tzinfo is None:
        validation_timestamp = validation_timestamp.tz_localize("UTC")
    else:
        validation_timestamp = validation_timestamp.tz_convert("UTC")
    test_end = pd.Timestamp(reference["timestamp"].max())
    first_timestamp = pd.Timestamp(reference["timestamp"].min())
    if not first_timestamp <= train_timestamp < validation_timestamp < test_end:
        raise ValueError(
            "Explicit split must satisfy data_start <= train_end < "
            "validation_end < data_end"
        )
    boundary = SplitBoundary(
        train_end=train_timestamp,
        validation_end=validation_timestamp,
        test_end=test_end,
    )
    labels = partition(reference, boundary)
    if any(not (labels == label).any() for label in ("train", "validation", "test")):
        raise ValueError("Explicit split creates an empty partition")
    return boundary


def partition(frame: pd.DataFrame, boundary: SplitBoundary) -> pd.Series:
    labels = pd.Series("test", index=frame.index, dtype="string")
    labels.loc[frame["timestamp"] <= boundary.validation_end] = "validation"
    labels.loc[frame["timestamp"] <= boundary.train_end] = "train"
    return labels
