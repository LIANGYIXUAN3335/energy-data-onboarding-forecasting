# Direct 1–24 Hour Forecast Sensitivity

This is a separate sensitivity analysis, not a replacement for the canonical
fixed-horizon 24-hour run and not a deployment claim.

## Locked design

- One forecast origin per source-local calendar day at hour `23`.
- Each origin predicts the complete 1–24 hour vector; stride is 24 hours.
- Full origin blocks are partitioned by target time. Blocks crossing a split
  boundary are embargoed and excluded.
- Train end marker: `2016-12-31T23:00:00+00:00`
- Validation end marker: `2017-03-31T23:00:00+00:00`
- Test end marker: `2017-12-31T23:00:00+00:00`
- Test targets: reference condition only.
- Condition parity at and after the fault cutoff: verified before fitting.
- Calendar features: preserved source-local wall-clock hour/day. The trailing
  `Z` comparison marker does not imply true UTC offset conversion.
- MAE intervals: whole building-week block bootstrap, streamed one repetition
  at a time; no rows-by-repetitions matrix.
- Pooled MASE-168 remains the primary sensitivity summary. Per-building and
  macro-building MASE are supplementary and use reference-training scales only.

## Complete results

| condition | model | n | mae | rmse | wmape | mase | r2 | mae_ci_low | mae_ci_high | mase_macro_building | mase_buildings_evaluated | mase_buildings_undefined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| corrupted | hist_gradient_boosting | 78960 | 70.6036 | 211.1370 | 0.1659 | 1.1398 | 0.8813 | 59.9952 | 83.4324 | 1.9016 | 12 | 0 |
| reference | hist_gradient_boosting | 78960 | 49.1025 | 139.6547 | 0.1154 | 0.7927 | 0.9481 | 41.8894 | 57.8591 | 1.3914 | 12 | 0 |
| remediated | hist_gradient_boosting | 78960 | 47.1280 | 127.6432 | 0.1107 | 0.7608 | 0.9566 | 41.1362 | 53.9493 | 1.2781 | 12 | 0 |
| corrupted | ridge | 78960 | 75.7209 | 150.6360 | 0.1779 | 1.2224 | 0.9396 | 67.9768 | 85.5047 | 5.3407 | 12 | 0 |
| reference | ridge | 78960 | 68.5404 | 139.1322 | 0.1610 | 1.1065 | 0.9485 | 62.3173 | 76.2720 | 5.2945 | 12 | 0 |
| remediated | ridge | 78960 | 67.8160 | 138.1491 | 0.1593 | 1.0948 | 0.9492 | 61.6951 | 75.6416 | 5.0761 | 12 | 0 |
| corrupted | seasonal_naive | 78960 | 64.4769 | 210.7912 | 0.1515 | 1.0409 | 0.8817 | 52.2107 | 78.5953 | 1.1400 | 12 | 0 |
| reference | seasonal_naive | 78960 | 64.4769 | 210.7912 | 0.1515 | 1.0409 | 0.8817 | 52.2107 | 78.5953 | 1.1400 | 12 | 0 |
| remediated | seasonal_naive | 78960 | 64.4769 | 210.7912 | 0.1515 | 1.0409 | 0.8817 | 52.2107 | 78.5953 | 1.1400 | 12 | 0 |

## Per-horizon results

See `horizon_metrics.csv` and the deterministic chart below. Every condition,
model, and horizon from 1 through 24 is retained.

![MAE by horizon](figures/mae_by_horizon.svg)

## Paired corrupted-versus-remediated differences

`mean_difference` is remediated absolute error minus corrupted absolute error
on the identical forecast keys. Negative values favor remediation. Intervals
resample the same whole building-week blocks used for sensitivity MAE.

| model | n | corrupted_mae | remediated_mae | mean_difference | difference_ci_low | difference_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting | 78960 | 70.6036 | 47.1280 | -23.4757 | -30.6189 | -17.8481 |
| ridge | 78960 | 75.7209 | 67.8160 | -7.9049 | -11.0173 | -5.0218 |
| seasonal_naive | 78960 | 64.4769 | 64.4769 | 0.0000 | 0.0000 | 0.0000 |

Per-horizon paired results are retained in
`horizon_paired_differences.csv`; no horizon is selected or omitted.

## Interpretation boundary

The sensitivity isolates horizon behavior under the recorded public-data
experiment. It does not alter the canonical fixed-h=24 result, establish
operational performance, or evaluate real-time deployment.
