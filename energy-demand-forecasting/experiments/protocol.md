# Pre-Specified Forecasting Protocol

## Question

Measure the effect of documented training-data defects and deterministic remediation on out-of-time hourly electricity-demand forecasting while holding the remaining experimental choices constant.

## Hypotheses

- H1: seeded defects reduce data readiness and may degrade forecast performance relative to the reference condition.
- H2: deterministic, past-only remediation recovers some data readiness without using future information.
- H3: remediated forecasts are more stable or accurate than corrupted forecasts under the same model, split, features, seed, and tuning budget.
- H4: any observed effect is examined across sites, primary-use categories, and buildings rather than being inferred from one aggregate alone.

These are hypotheses, not promised outcomes. All configured condition/model results are reported.

## Locked decisions

- Forecast horizon: 24 hours.
- Seasonal period: 168 hours.
- Test targets: reference condition only.
- Split: pre-specified calendar boundaries: train through 2016-12-31 23:00, validation through 2017-03-31 23:00, and test from 2017-04-01 through the end of available 2017 data. Fixture-only tests may use a documented fractional fallback when they do not span these dates; full runs may not silently use that fallback.
- Validation use: the Q1 validation partition is reserved but not used to choose the canonical models or hyperparameters, which are fixed before the run. Any later tuning or model selection must be declared and reported as a separately labeled experiment.
- Models: seasonal naive and fixed histogram gradient boosting.
- Forecast availability boundary: for target hour `t`, load-derived features use `t-24` or earlier. Model inputs are lags 24, 48, and 168; trailing 24- and 168-hour means and standard deviations shifted by 24 hours; target calendar fields known at issuance; and public building metadata.
- Primary metric: MASE. Supporting metrics: MAE, RMSE, aggregate wMAPE, supplementary R-squared, MAE confidence interval, paired corrupted-versus-remediated absolute-error differences, and subgroup results.
- Alignment: final metrics use the intersection of valid test keys across all conditions and models.
- Subgroups: site, primary-use category, building, and calendar quarter when observations are available. These are coverage and robustness slices, not a fairness assessment.

## Leakage controls

- Fault injection stops at the onboarding fault cutoff.
- Remediation is past-only and never interpolates from a future value.
- Rolling features are shifted by the full 24-hour forecast horizon before aggregation.
- Preprocessing is fit on the training partition only.
- Observed test-period weather is excluded from the primary task.
- Producer hashes, exact fault values, equal condition keys, and the exclusive fault cutoff must pass before feature construction.
- Actual fault timestamps must fall inside the chronological training partition.

## Reporting rule

Do not suppress a condition, subgroup, failed hypothesis, or adverse result after seeing the metrics. Any deviation from this protocol must be dated, justified, and run as a separately labeled sensitivity analysis.
