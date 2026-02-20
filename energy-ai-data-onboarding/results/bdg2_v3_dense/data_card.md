# Data card: Building Data Genome Project 2 onboarding output

This is a reproducible public-data research pipeline for building-energy data onboarding and forecasting preparation.

## Identity and scope

- Source version: `v1.0`
- DOI: `10.5281/zenodo.3887306`
- Execution scope: `verified_bdg2_subset`
- Selected buildings: 12
- Selected rows: 210528
- Time span: `2016-01-01T00:00:00+00:00` through `2017-12-31T23:00:00+00:00`
- Sites: Eagle, Panther, Rat
- Primary-use categories: Education, Office

## Timestamp semantics

BDG2 raw meter timestamps are source-local wall-clock readings. For this prototype they are represented with a UTC marker for stable parsing without performing an offset conversion. The committed subset is restricted to sites with the same source timezone (US/Eastern). This must not be described as cross-timezone UTC harmonization; DST interpretation remains a limitation.

## Conditions

`reference`, `corrupted`, and `remediated` have identical
`(timestamp, building_id)` key sets. Corruption is seeded, value-only, and
strictly earlier than `2017-01-01T00:00:00+00:00`. Remediation is deterministic
and past-only; unresolved values remain present as nulls and are quarantined.

## Quality

Gate outcomes: corrupted=fail, detector_validation=fail, reference=warn, remediated=warn. See `quality_summary.json`,
`quality_issues.csv.gz`, and `detector_metrics.csv` for generated measurements.
These are coverage and sensor-integrity observations, not a demographic fairness
assessment.

## Intended use

Reproducible research on data readiness and downstream forecasting sensitivity.
This is not an operational utility dataset or a production service.

## Privacy

The pipeline is stateless. It contains public energy data, public site
weather and reproducibility metadata only; it stores no personal data.
