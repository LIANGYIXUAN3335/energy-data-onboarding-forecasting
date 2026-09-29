# Energy-data onboarding audit report

This is a reproducible public-data research pipeline for building-energy data onboarding and forecasting preparation.

Created: `2026-09-29T02:19:11Z`

## Findings in brief

The reference data (210,528 hourly rows, 12 buildings) passed publication with gate status `warn`. 855 seeded events, 1660 value-level fault rows (1.57% of the 105,408 training rows) were written before the cutoff; seeded faults were fully detected except long_zero_block (recall 0.98), positive_spike (recall 0.93), stuck_segment (recall 0.68). Remediation repaired 3,015 rows and quarantined 4,418; the gate on the remediated condition is `warn`. Every count here is recomputed from the committed artifacts, and adverse outcomes are kept.

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

Generated outcomes: corrupted=fail, detector_validation=fail, reference=warn, remediated=warn. A fail or warning remains in the
machine-readable issue log; it is not suppressed to create a preferred result.

## Seeded detector validation

The isolated detector-validation suite injected `1664`
key-level fault records. Precision, recall, false-positive rate, and counts by
fault family are in `detector_metrics.csv`. Natural anomalies were not treated
as seeded ground truth.

## Downstream condition preparation

The downstream suite injected `1660`
value-level fault records before `2017-01-01T00:00:00+00:00`. It did not add,
remove, or move keys. Past-only remediation logged `7433`
actions, including `3015` repaired rows and
`4418` unresolved actions. The quarantine artifact
contains `4418` rows. A suspected unit scale is never silently
corrected by dividing by an inferred factor.

## Detector and remediation policy

Run detectors flag every reading of a run once it reaches its threshold. The stuck-sensor threshold (8 h) is raised per building to just above the 0.999 quantile of that building's natural constant-run lengths (`detector_calibration.csv`), so coarsely quantized meters are not mistaken for stuck sensors. A zero run of at least 6 h counts only during hours when the building is normally active (causal hour-of-week median over the previous 8 weeks above 0.25 times the trailing 168-hour median). Calibration uses the reference condition strictly before the fault cutoff, so seeded faults never shape their own thresholds.

Remediation first forward-fills gaps of at most 3 h from the last valid reading, then fills gaps of at most 48 h with the median of the same weekday and hour over the previous 8 weeks, using only earlier, originally valid readings. Longer gaps are quarantined as null. Suspected unit or level shifts are replaced by that profile value rather than divided by an inferred factor.

Site weather (`weather.csv.gz`, 52632 rows for Eagle, Panther, Rat) is published on the same hourly grid as the meters. 6 duplicate source keys were averaged and 19 missing grid hours were materialized as null; the highest per-variable missing rate is 43.0% (cloud cover). Weather is never modified by the fault suites and is imputed downstream only inside training-fitted preprocessing.

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

Only public dataset content, public site weather and reproducibility
metadata are processed. No personal information is retained and nothing is
transmitted.
