# Protocol amendment v3 (forecasting), dated 2026-09-28

Separately labeled experiment, designed after the v2 result was examined.
The pre-specified v2 protocol and its committed bundles (`results/canonical_24h_v2_final`,
`results/direct_1_to_24_sensitivity_v1_final`) remain the baseline and are
not modified. Configuration: `configs/experiment_v3.json` and
`configs/direct_1_to_24_sensitivity_v3.json`.

## Changes

1. **Inputs.** The v3 onboarding bundle (`bdg2_v3`, see the companion
   amendment) with recalibrated detectors and profile repair.
2. **Feature set `v3`** (27 numeric + 3 categorical = 30 features):
   load lags 24/48/168/336 h; trailing 24 h mean, std, min, max and 168 h
   mean, std, all ending at t-24; hour, weekday, day-of-year and month as
   sine/cosine pairs; weekend and U.S. federal holiday flags; and site
   weather observed at t-24 or earlier (air temperature lags 24/48/168 h,
   dew temperature and wind speed at t-24, trailing 24 h and 168 h mean air
   temperature ending at t-24). No observed weather closer to the target than
   24 h is used, so the model needs no weather forecast.
3. **Models.** Seasonal naive, ridge regression (alpha 1.0 on standardized
   inputs), random forest (200 trees, min 5 samples per leaf, 50% features
   per split) and histogram gradient boosting (unchanged hyperparameters).
   Hyperparameters were fixed before the run; the validation partition
   remains unused for selection.
4. **Sensitivity.** The 1-24 h direct sensitivity adds ridge and the same
   origin-time weather (air temperature at the origin, 24 h and 168 h
   earlier, dew temperature and wind speed at the origin, trailing 24 h mean).
   Random forest is omitted there for runtime.
5. **Reports** now contain an auto-generated "Findings in words" section
   computed from the same tables as the figures.

## Unchanged

Split (train through 2016, validation 2017 Q1 reserved, test 2017 Q2-Q4),
reference-only test targets, common-key alignment across all condition/model
pairs, pooled MASE-168 as primary metric, hash-bound producer verification,
and the rule that every load- or weather-derived feature is observed at
t-24 or earlier.
