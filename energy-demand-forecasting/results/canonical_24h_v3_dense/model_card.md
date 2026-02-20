# Run-Specific Model Card

## Intended use

This run is a retrospective public-data benchmark for measuring how documented
training-data conditions affect hourly building-load forecasts. It is not an
operational utility forecast, dispatch system, safety system, or deployment claim.

## Models

- Seasonal naive: lag 168, then lag 24 fallback. No one-hour fallback is used.
- Ridge regression on standardized features: fixed alpha recorded in `run_manifest.json`.
- Random forest: fixed tree count, leaf size and feature fraction recorded in `run_manifest.json`.
- Histogram gradient boosting: fixed configuration recorded in `run_manifest.json`.

All models use the same chronological split and reference-only test targets
across all data conditions. Preprocessing for every learned model is fitted on
its training partition only. For a target at `t`, all load-derived and
weather-derived features use `t-24` or earlier. Feature set:
`v3` (30 features:
lag_24, lag_48, lag_168, lag_336, rolling_mean_24, rolling_std_24, rolling_min_24, rolling_max_24, rolling_mean_168, rolling_std_168, hour_sin, hour_cos, dow_sin, dow_cos, doy_sin, doy_cos, month_sin, month_cos, is_weekend, is_us_federal_holiday, air_temperature_lag_24, air_temperature_lag_48, air_temperature_lag_168, dew_temperature_lag_24, wind_speed_lag_24, air_temperature_mean_24_lag_24, air_temperature_mean_168_lag_24, building_id, site_id, primary_use).

## Evaluation

- Train end: `2016-12-31T23:00:00+00:00`
- Validation end: `2017-03-31T23:00:00+00:00`
- Test end: `2017-12-31T23:00:00+00:00`
- Validation use: reserved; canonical models and hyperparameters fixed a priori.
- Primary metric (unchanged): pooled MASE with the recorded 168-hour
  reference-training scale.
- Supplementary metric: per-building MASE using building-specific
  reference-training scales and its finite-building macro average. Undefined
  building scales remain visible and are not silently treated as zero.
- Supporting metrics: MAE, RMSE, aggregate wMAPE, R-squared, bootstrap MAE
  interval, paired absolute-error difference, and subgroup robustness slices.
- Common evaluation keys: `79178`

## Limitations

The learned model is a compact benchmark, not a tuned production model. The
bootstrap interval samples evaluation rows and does not establish deployment
performance. Findings about seeded faults apply to the recorded fault suite;
natural-data comparisons are observational. Building and quarter slices can be
small and must not be described as a fairness audit. Observed future weather is
excluded, and no grid-control, real-time, security, or economic impact is tested.
The intended Panther/Eagle/Rat Phase 1 subset shares the US/Eastern source
timezone. BDG2 timestamps are local wall-clock values; producer canonical
markers must not be interpreted as proof of true UTC offset conversion.

## Provenance and state

Inputs and producer manifests are hash-verified before fitting. Exact runtime,
features, hyperparameters, split, counts, hashes, and seed are recorded in
`run_manifest.json`. The pipeline is a stateless batch program and retains no
personal data.
