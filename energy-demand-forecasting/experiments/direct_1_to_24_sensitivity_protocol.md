# Direct 1–24 Hour Sensitivity Protocol

Status: pre-specified local sensitivity design, separate from the canonical
fixed 24-hour experiment.

## Purpose

Measure whether the observed comparison across reference, corrupted, and
remediated training conditions is stable across target horizons from 1 through
24 hours. The canonical fixed-horizon-24 result remains primary.

## Locked construction

- One forecast origin per source-local calendar day, at hour 23 by default.
- The stride is exactly 24 hours and every retained origin predicts the complete
  next 24-hour vector.
- `forecast_origin` equals `cutoff_timestamp`. All load-derived learned-model
  features are observed at or before that cutoff.
- Seasonal-naive inputs use the target's 168-hour lag, then 24-hour lag. For
  horizons no greater than 24, both are available no later than the cutoff.
- Target hour/day fields are known calendar attributes. They use the preserved
  source-local wall-clock calendar; the stored `Z` is a canonical comparison
  marker and does not establish a true UTC instant.
- Full origin blocks are assigned using target timestamps. The final target of
  a training origin must be at or before `train_end`; validation targets must
  all be after `train_end` and at or before `validation_end`; test targets must
  all be after `validation_end`. A block that crosses either boundary is
  embargoed and excluded.
- All documented injected faults must be in training. At and after
  `fault_cutoff`, corrupted and remediated load values must exactly match the
  reference condition; otherwise the run fails before fitting.
- `fault_cutoff` is exclusive, whereas `train_end` is inclusive. On this hourly
  dataset the cutoff may therefore equal the first hourly timestamp after
  `train_end` (for example, 2017-01-01 00:00 after a 2016-12-31 23:00 training
  endpoint). A later gap is rejected.
- Test targets always come from the reference condition, and every
  condition/model pair is aligned on the exact same forecast key:
  `(forecast_origin, horizon_hours, target_timestamp, building_id)`.

## Models and metrics

The model family and fixed hyperparameters match the compact canonical
benchmark. No sensitivity result is used to retune that benchmark. Pooled
MASE-168 is reported using one reference-training denominator. Supplementary
per-building MASE and its finite-building macro mean use building-specific
reference-training denominators. Per-horizon metrics include every horizon.

MAE confidence intervals resample whole building-week blocks. The
implementation aggregates each block and streams one repetition at a time; it
does not allocate a rows-by-repetitions matrix.

Aggregate and per-horizon paired differences use remediated absolute error
minus corrupted absolute error on identical forecast keys. Their confidence
intervals preserve row pairing and resample the same whole building-week
blocks. Negative values favor remediation; all horizons are retained.

## Interpretation boundary

This is a retrospective public-data sensitivity analysis. It does not measure
real-time operational performance.
