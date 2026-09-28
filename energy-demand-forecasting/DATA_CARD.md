# Forecast Input Data Card

## Input contract

The repository consumes three producer-published condition files:

- `reference.csv.gz`
- `corrupted.csv.gz`
- `remediated.csv.gz`

Every file must contain `timestamp`, `building_id`, `load`, `site_id`,
`primary_use`, and `condition`. The uniqueness key is timestamp plus building.
All three conditions must contain exactly the same canonical timestamp/building
keys. The Phase 1 Panther/Eagle/Rat selection is intentionally limited to
US/Eastern source sites. BDG2 raw meter timestamps are source-local wall-clock
values; calendar features preserve those wall-clock values. A producer-added
UTC marker is used for canonical comparison and must not be described as proof
that the source observations were offset-converted to true UTC.

The bundle must also contain `dataset_manifest.json` and
`fault_manifest.json`. The dataset manifest binds every condition file to its
SHA-256 digest and may bind row counts and columns. The fault manifest binds the
reference hash, the `downstream_forecasting` suite, an exclusive fault cutoff,
and exact changed keys and values.
Any changed reference/corrupted value not documented in the fault manifest is a
hard failure.

## Intended source

The Phase 1 intended source is Building Data Genome 2 v1.0 public electricity
meter data plus public building metadata, published by its original maintainers.
Exact source DOI, license, source-file hashes, selection rule, buildings, sites,
date range, and producer transformations belong in the companion producer
manifest and data card. This consumer does not silently substitute a dataset.

## Role of each condition

- `reference` supplies the evaluation targets and reference-training scale.
- `corrupted` contains only producer-documented value-level faults, restricted
  to the training period.
- `remediated` contains deterministic producer remediation while preserving the
  same timestamp/building keys.

Null loads may be preserved so that key sets remain comparable. Learned-model
training excludes null targets; lag and rolling features can remain null and are
imputed by a preprocessing stage fitted on that condition's training partition.

## Leakage and integrity controls

- Lag and rolling features are grouped by building and shifted by the full
  24-hour horizon. For a target at `t`, no load after `t-24` is used.
- Splits are chronological and selected from reference timestamps.
- Producer hashes, row counts, columns, equal keys, exact fault values, and the
  exclusive fault cutoff are checked before training.
- Faults are rejected if any affected timestamp falls outside the experiment's
  training partition.
- Test targets are joined only from `reference`.
- All reported condition/model pairs use one common finite evaluation key set.

## Sensitive-data boundary

The schema contains public building identifiers and aggregate building meter
observations only. It has no fields for a person, private communication,
credentials, conversational state, or user profile. Personally identifying
material must not be added.

## Known data risks

Public meter data can contain missing values, time-zone ambiguity, sensor
failures, unit errors, coverage imbalance, and metadata mistakes. Producer
checks reduce but do not eliminate those risks. A passing manifest verifies the
declared bundle; it does not prove that the public source is complete or free of
all measurement error.
