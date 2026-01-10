# Energy-data onboarding audit report

This is a reproducible public-data research pipeline for building-energy data onboarding and forecasting preparation.

Created: `2026-09-28T19:05:53Z`

## Run scope

The pipeline processed `210528` long-form observations
from `12` selected buildings. The source is
`Building Data Genome Project 2` `v1.0`. This run is
classified as `verified_bdg2_subset`.

## Source and time handling

BDG2 raw meter timestamps are source-local wall-clock readings. For this prototype they are represented with a UTC marker for stable parsing without performing an offset conversion. The committed subset is restricted to sites with the same source timezone (US/Eastern). This must not be described as cross-timezone UTC harmonization; DST interpretation remains a limitation.

The source bytes, configuration, condition datasets, logs, and reports are
identified by SHA-256 in `dataset_manifest.json`.

## Quality gates

Generated outcomes: corrupted=fail, detector_validation=fail, reference=warn, remediated=fail. A fail or warning remains in the
machine-readable issue log; it is not suppressed to create a preferred result.

## Seeded detector validation

The isolated detector-validation suite injected `31`
key-level fault records. Precision, recall, false-positive rate, and counts by
fault family are in `detector_metrics.csv`. Natural anomalies were not treated
as seeded ground truth.

## Downstream condition preparation

The downstream suite injected `27`
value-level fault records before `2017-01-01T00:00:00+00:00`. It did not add,
remove, or move keys. Past-only remediation logged `22619`
actions, including `586` repaired rows and
`22033` unresolved actions. The quarantine artifact
contains `22033` rows. A suspected unit scale is never silently
corrected by dividing by an inferred factor.

## Limitations

- This is a local batch research prototype, not real-time or production-grade.
- The code does not evaluate operational utility deployment, grid control,
  production reliability, or economic impact.
- Faults are synthetic stress tests and are not estimates of real-world fault
  prevalence.
- Detector thresholds and simple causal remediation are transparent baselines,
  not an assertion of domain-optimal cleaning.
- Raw BDG2 timestamps require the source-timezone caveat stated above.

## Privacy and data boundary

Only public dataset content and reproducibility metadata are processed. No
personal information, conversation, prompt, model memory, embedding, vector
database, user profile, or telemetry is retained.
