"""CSV ingestion and BDG2-compatible schema normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .contracts import IssueRecord, condition_frame


TIMESTAMP_CANDIDATES = ("timestamp", "datetime", "date_time", "time")
BUILDING_CANDIDATES = ("building_id", "building", "meter_id")
SITE_CANDIDATES = ("site_id", "site", "siteid")
PRIMARY_USE_CANDIDATES = (
    "primary_use",
    "primaryspaceusage",
    "primary_space_usage",
    "primary_space_use",
)
TIMEZONE_CANDIDATES = ("timezone", "time_zone", "tz")


@dataclass(frozen=True)
class SelectionConfig:
    """Pre-model, metadata-only subset selection."""

    site_ids: tuple[str, ...] = ()
    building_ids: tuple[str, ...] = ()
    maximum_building_count: int | None = 12
    maximum_source_rows: int | None = None
    full_run: bool = False
    strategy: str = "metadata_round_robin_by_site_then_primary_use"
    read_chunk_rows: int = 4096


@dataclass
class IngestionResult:
    frame: pd.DataFrame
    issues: list[IssueRecord]
    selected_buildings: list[str]
    selected_sites: list[str]
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    source_rows_read: int = 0


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def read_meter_csv(
    path: str | Path,
    *,
    timestamp_column: str | None = None,
) -> pd.DataFrame:
    """Read a BDG2-style wide CSV or an already-long meter CSV."""

    source = pd.read_csv(path)
    resolved_timestamp = timestamp_column or _find_column(
        source.columns, TIMESTAMP_CANDIDATES
    )
    if resolved_timestamp is None or resolved_timestamp not in source.columns:
        raise ValueError(
            "Meter CSV needs a timestamp column; use --timestamp-column when its "
            "name is non-standard."
        )

    building_column = _find_column(source.columns, BUILDING_CANDIDATES)
    load_column = _find_column(source.columns, ("load", "value", "meter_reading"))
    if building_column and load_column:
        long = source.loc[:, [resolved_timestamp, building_column, load_column]].rename(
            columns={
                resolved_timestamp: "timestamp",
                building_column: "building_id",
                load_column: "load",
            }
        )
    else:
        value_columns = [
            column for column in source.columns if column != resolved_timestamp
        ]
        if not value_columns:
            raise ValueError("Wide meter CSV contains no building value columns.")
        long = source.melt(
            id_vars=[resolved_timestamp],
            value_vars=value_columns,
            var_name="building_id",
            value_name="load",
        ).rename(columns={resolved_timestamp: "timestamp"})

    long["timestamp"] = pd.to_datetime(long["timestamp"], errors="coerce", utc=True)
    long["building_id"] = long["building_id"].astype("string").str.strip()
    long["load"] = pd.to_numeric(long["load"], errors="coerce")
    return long


def read_metadata_csv(path: str | Path) -> pd.DataFrame:
    """Read and normalize the minimum metadata needed downstream."""

    source = pd.read_csv(path)
    building = _find_column(source.columns, BUILDING_CANDIDATES)
    site = _find_column(source.columns, SITE_CANDIDATES)
    primary_use = _find_column(source.columns, PRIMARY_USE_CANDIDATES)
    timezone = _find_column(source.columns, TIMEZONE_CANDIDATES)
    missing = [
        name
        for name, resolved in (
            ("building_id", building),
            ("site_id", site),
            ("primary_use", primary_use),
        )
        if resolved is None
    ]
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")

    selected = [building, site, primary_use]
    if timezone is not None:
        selected.append(timezone)
    normalized = source.loc[:, selected].rename(
        columns={
            building: "building_id",
            site: "site_id",
            primary_use: "primary_use",
            **({timezone: "timezone"} if timezone is not None else {}),
        }
    )
    if "timezone" not in normalized:
        normalized["timezone"] = pd.NA
    for column in ("building_id", "site_id", "primary_use", "timezone"):
        normalized[column] = normalized[column].astype("string").str.strip()
        normalized[column] = normalized[column].mask(normalized[column].eq(""), pd.NA)
    return normalized


def metadata_quality_issues(metadata: pd.DataFrame) -> list[IssueRecord]:
    """Evaluate metadata key/cardinality before canonicalizing the join."""

    denominator = max(len(metadata), 1)
    duplicate_mask = metadata["building_id"].notna() & metadata["building_id"].duplicated(
        keep=False
    )
    missing_key = metadata["building_id"].isna()
    missing_values = metadata[["site_id", "primary_use"]].isna().any(axis=1)

    outcomes: list[IssueRecord] = []
    for code, name, mask, failure, repairable, columns in (
        (
            "METADATA_DUPLICATE_KEY",
            "metadata key cardinality",
            duplicate_mask,
            "fail",
            False,
            ("building_id",),
        ),
        (
            "METADATA_MISSING_KEY",
            "metadata building identifier",
            missing_key,
            "fail",
            False,
            ("building_id",),
        ),
        (
            "METADATA_REQUIRED_VALUE_MISSING",
            "metadata required values",
            missing_values,
            "warn",
            False,
            ("site_id", "primary_use"),
        ),
    ):
        count = int(mask.sum())
        status = failure if count else "pass"
        sample = tuple(
            metadata.loc[mask, [column for column in columns if column in metadata]]
            .head(5)
            .to_dict(orient="records")
        )
        outcomes.append(
            IssueRecord(
                code=code,
                check=name,
                status=status,
                count=count,
                denominator=denominator,
                rate=count / denominator,
                message=(
                    f"{count} metadata rows violate {name}."
                    if count
                    else f"No rows violate {name}."
                ),
                repairable=repairable,
                columns=columns,
                sample=sample,
            )
        )
    return outcomes


def join_metadata(
    meters: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, list[IssueRecord]]:
    """Perform a deterministic many-to-one join and retain cardinality issues."""

    issues = metadata_quality_issues(metadata)
    canonical = metadata.dropna(subset=["building_id"]).drop_duplicates(
        subset=["building_id"], keep="first"
    )
    joined = meters.merge(canonical, on="building_id", how="left", validate="m:1")
    return condition_frame(joined, "reference"), issues


def ingest_dataset(
    meter_path: str | Path,
    metadata_path: str | Path,
    *,
    timestamp_column: str | None = None,
) -> tuple[pd.DataFrame, list[IssueRecord]]:
    """Load meter and metadata files into the reference-condition contract."""

    meters = read_meter_csv(meter_path, timestamp_column=timestamp_column)
    metadata = read_metadata_csv(metadata_path)
    return join_metadata(meters, metadata)


def select_buildings(
    metadata: pd.DataFrame,
    available_buildings: Iterable[str],
    config: SelectionConfig,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Select columns deterministically without inspecting load or forecasts."""

    if config.strategy != "metadata_round_robin_by_site_then_primary_use":
        raise ValueError(f"Unsupported selection strategy: {config.strategy}")
    available = {str(value) for value in available_buildings}
    candidates = metadata.loc[metadata["building_id"].isin(available)].copy()
    exclusions: list[dict[str, Any]] = []
    for building_id in sorted(set(metadata["building_id"].dropna().astype(str)) - available):
        exclusions.append(
            {"building_id": building_id, "reason": "not_present_in_meter_header"}
        )

    if config.site_ids:
        allowed_sites = set(map(str, config.site_ids))
        excluded = candidates.loc[~candidates["site_id"].astype(str).isin(allowed_sites)]
        exclusions.extend(
            {"building_id": str(value), "reason": "site_filter"}
            for value in sorted(excluded["building_id"].dropna().astype(str).unique())
        )
        candidates = candidates.loc[
            candidates["site_id"].astype(str).isin(allowed_sites)
        ]

    if config.building_ids:
        requested = [str(value) for value in config.building_ids]
        candidate_ids = set(candidates["building_id"].dropna().astype(str))
        missing = [value for value in requested if value not in candidate_ids]
        if missing:
            raise ValueError(
                f"Configured building IDs are unavailable after filters: {missing}"
            )
        chosen = list(dict.fromkeys(requested))
    else:
        candidates = candidates.drop_duplicates("building_id", keep="first")
        candidates = candidates.sort_values(
            ["site_id", "primary_use", "building_id"], kind="stable", na_position="last"
        )
        groups = [
            group["building_id"].astype(str).tolist()
            for _, group in candidates.groupby(
                ["site_id", "primary_use"], sort=True, dropna=False
            )
        ]
        chosen = []
        offset = 0
        while any(offset < len(group) for group in groups):
            for group in groups:
                if offset < len(group):
                    chosen.append(group[offset])
            offset += 1

    limit = None if config.full_run else config.maximum_building_count
    if limit is not None:
        if limit <= 0:
            raise ValueError("maximum_building_count must be positive or null.")
        omitted = chosen[limit:]
        exclusions.extend(
            {"building_id": value, "reason": "maximum_building_count"}
            for value in omitted
        )
        chosen = chosen[:limit]
    if not chosen:
        raise ValueError("Building selection produced an empty set.")
    return chosen, exclusions


def _read_selected_wide(
    path: str | Path,
    *,
    timestamp_column: str,
    building_ids: list[str],
    chunk_rows: int,
    maximum_source_rows: int | None,
) -> tuple[pd.DataFrame, int]:
    if chunk_rows <= 0:
        raise ValueError("read_chunk_rows must be positive.")
    remaining = maximum_source_rows
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    with pd.read_csv(
        path,
        usecols=[timestamp_column, *building_ids],
        chunksize=chunk_rows,
    ) as reader:
        for source in reader:
            if remaining is not None:
                if remaining <= 0:
                    break
                source = source.iloc[:remaining]
            rows_read += len(source)
            if remaining is not None:
                remaining -= len(source)
            long = source.melt(
                id_vars=[timestamp_column],
                value_vars=building_ids,
                var_name="building_id",
                value_name="load",
            ).rename(columns={timestamp_column: "timestamp"})
            chunks.append(long)
    if not chunks:
        return pd.DataFrame(columns=["timestamp", "building_id", "load"]), 0
    result = pd.concat(chunks, ignore_index=True)
    result["_raw_timestamp"] = result["timestamp"].astype("string")
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
    result["building_id"] = result["building_id"].astype("string").str.strip()
    result["load"] = pd.to_numeric(result["load"], errors="coerce")
    return result, rows_read


def ingest_selected_dataset(
    meter_path: str | Path,
    metadata_path: str | Path,
    *,
    selection: SelectionConfig | None = None,
    timestamp_column: str | None = None,
    source_version: str = "v1.0",
) -> IngestionResult:
    """Column-prune and normalize a deterministic BDG2 wide-data subset."""

    config = selection or SelectionConfig()
    metadata = read_metadata_csv(metadata_path)
    header = pd.read_csv(meter_path, nrows=0)
    resolved_timestamp = timestamp_column or _find_column(
        header.columns, TIMESTAMP_CANDIDATES
    )
    if resolved_timestamp is None:
        raise ValueError("Meter CSV does not contain a recognized timestamp column.")
    available_buildings = [
        str(column) for column in header.columns if column != resolved_timestamp
    ]
    selected, exclusions = select_buildings(metadata, available_buildings, config)
    meters, rows_read = _read_selected_wide(
        meter_path,
        timestamp_column=resolved_timestamp,
        building_ids=selected,
        chunk_rows=config.read_chunk_rows,
        maximum_source_rows=config.maximum_source_rows,
    )
    # The publication gate applies only to metadata rows that can affect the
    # selected output. Problems in unselected buildings are outside this run's
    # evidentiary scope and must not block a clean selected subset.
    selected_metadata = metadata.loc[metadata["building_id"].isin(selected)].copy()
    issues = metadata_quality_issues(selected_metadata)
    canonical = metadata.dropna(subset=["building_id"]).drop_duplicates(
        "building_id", keep="first"
    )
    joined = meters.merge(canonical, on="building_id", how="left", validate="m:1")
    joined["source_version"] = source_version
    joined = joined.sort_values(
        ["timestamp", "building_id"], kind="stable", na_position="first"
    ).reset_index(drop=True)
    selected_sites = sorted(joined["site_id"].dropna().astype(str).unique())
    return IngestionResult(
        frame=joined,
        issues=issues,
        selected_buildings=selected,
        selected_sites=selected_sites,
        exclusions=exclusions,
        source_rows_read=rows_read,
    )
