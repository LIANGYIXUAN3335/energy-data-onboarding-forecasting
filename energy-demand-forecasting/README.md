# Energy Demand Forecasting

This repository evaluates how documented data-quality conditions affect a reproducible 24-hour-ahead electricity-demand forecasting task. It consumes versioned outputs from the companion `energy-ai-data-onboarding` repository and reports every pre-specified condition, including null or negative results.

The repository is a stateless public-data implementation: it stores no
personal data and transmits nothing.

## Research question

Holding the forecasting model, features, split, seed, and tuning budget constant, how do controlled training-data defects and deterministic onboarding remediation affect out-of-time electricity-demand forecasts?

The primary task predicts each building's hourly electricity load at a fixed 24-hour horizon. For target hour `t`, every load-derived feature is computed from `t-24` or earlier: lags 24/48/168 and trailing 24/168-hour aggregates shifted by 24 hours. Target-hour calendar fields and public building metadata are known at issuance. Observed future weather is intentionally excluded from the primary experiment.

## Conditions

- `reference`: source/reference data published by the onboarding pipeline.
- `corrupted`: the same data with a seeded, machine-readable fault suite applied only before the fault cutoff.
- `remediated`: the corrupted condition after deterministic, past-only remediation and quarantine rules.

The test targets always come from `reference`. Predictions are evaluated on one common set of timestamp/building keys across every condition and model.

## Models and metrics

- Seasonal naive using the previous week and then previous day as fallback; it never uses a one-hour lag for this day-ahead task.
- A fixed `HistGradientBoostingRegressor` pipeline with training-only preprocessing.
- v3 adds ridge regression (standardized inputs, alpha 1.0) and a random
  forest (200 trees, minimum 5 samples per leaf, 50% of features per split),
  all with the same training-only preprocessing and fixed a priori.
- Primary metric (unchanged): pooled MASE with a 168-hour scale computed once
  from reference training data.
- Supplementary metric: per-building MASE using each building's own
  reference-training scale, plus an unweighted macro average over buildings
  with finite denominators. Undefined scales remain visible and are counted.
- Supporting metrics: MAE, RMSE, aggregate wMAPE, R-squared, bootstrap 95% confidence interval for MAE, and paired corrupted-versus-remediated absolute-error differences.
- Coverage and subgroup robustness metrics by site, primary-use category, building, and calendar quarter. These are not presented as a fairness assessment.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/energy-forecast verify-inputs \
  --input-dir ../energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened
.venv/bin/energy-forecast run \
  --reference ../energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened/reference.csv.gz \
  --corrupted ../energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened/corrupted.csv.gz \
  --remediated ../energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened/remediated.csv.gz \
  --producer-manifest ../energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened/dataset_manifest.json \
  --fault-manifest ../energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened/fault_manifest.json \
  --config configs/experiment.json \
  --output-dir results/canonical_24h_v2_repeat
.venv/bin/energy-forecast verify-results \
  --result-dir results/canonical_24h_v2_repeat \
  --input-dir ../energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened
```

Input verification fails closed on schema or timestamp errors, duplicate or unequal keys, producer-hash or row-count mismatches, undocumented reference-to-corrupted changes, incorrect fault values, and faults at or after the declared cutoff. Remediated changes outside the injected-fault set are counted separately because they may represent documented natural-data remediation; this consumer does not mislabel them as injected faults. The experiment additionally rejects injected faults outside its chronological training partition.

For a short smoke test, set `"max_model_iterations": 20` in a copied config.
The committed protocol, not a post-hoc edited config, controls full runs.

## Outputs

- `metrics.csv`: one row for every condition/model pair.
- `building_mase.csv`: auditable building-specific reference-training scales,
  MAE, and MASE for every condition/model/building row.
- `subgroup_metrics.csv`: errors by site, primary-use category, building, and quarter.
- `paired_differences.csv`: paired corrupted-versus-remediated error differences by model; negative values favor remediation.
- `predictions.csv.gz`: aligned out-of-time predictions and targets.
- `run_manifest.json`: producer and input hashes, config hash, runtime versions, seed, split, feature/model configuration, partition counts, and result hashes.
- `model_card.md`: a run-specific intended-use, evaluation, provenance, and limitation record.
- `report.md` and `report.html`: generated summaries sourced directly from committed result files.
- `figures/*.svg`: dependency-free, byte-deterministic figures regenerated and
  checked against `metrics.csv`; their hashes are bound in the run manifest.

## Separate 1–24 hour sensitivity

The optional direct multi-horizon path is deliberately separate from the
canonical fixed-h=24 run. It uses one daily source-local origin at hour 23,
predicts the complete next 24-hour vector, partitions by target time, and
embargoes any origin that crosses a split boundary. It fails closed unless all
conditions equal reference at and after the fault cutoff. Test targets remain
reference-only, and output rows retain forecast origin, cutoff, horizon, and
target timestamp.

```bash
.venv/bin/energy-forecast run-sensitivity \
  --reference ../energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened/reference.csv.gz \
  --corrupted ../energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened/corrupted.csv.gz \
  --remediated ../energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened/remediated.csv.gz \
  --producer-manifest ../energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened/dataset_manifest.json \
  --fault-manifest ../energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened/fault_manifest.json \
  --config configs/direct_1_to_24_sensitivity.json \
  --output-dir results/direct_1_to_24_sensitivity_repeat
.venv/bin/energy-forecast verify-sensitivity-results \
  --result-dir results/direct_1_to_24_sensitivity_repeat \
  --input-dir ../energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened
```

The sensitivity output directory must be new or empty, preventing stale files
from appearing to belong to a successful run. Its MAE intervals use whole
building-week block bootstrap samples and stream one repetition at a time.
`paired_differences.csv` and `horizon_paired_differences.csv` report
remediated-minus-corrupted absolute-error differences on identical forecast
keys, using the same dependence-aware block resampling; negative values favor
remediation.
See `experiments/direct_1_to_24_sensitivity_protocol.md` for the locked design.

Completed bundles are the pre-registered v2 baseline
(`results/canonical_24h_v2_final`, `results/direct_1_to_24_sensitivity_v1_final`)
and the v3 amendment dated 2026-09-28 (`results/canonical_24h_v3`,
`results/direct_1_to_24_sensitivity_v3`; see
`experiments/protocol_v3_amendment.md`).

## v3: weather, a full calendar, and four models

`configs/experiment_v3.json` consumes the v3 onboarding bundle and sets
`"feature_set": "v3"`, `"weather_features": true` and
`"models": ["seasonal_naive", "ridge", "random_forest", "hist_gradient_boosting"]`.
The v3 feature set has 30 features: load lags 24/48/168/336 h; trailing 24 h
mean, std, min, max and 168 h mean, std ending at t-24; hour, weekday,
day-of-year and month as sine/cosine pairs; weekend and U.S. federal holiday
flags; air temperature at t-24, t-48 and t-168, dew temperature and wind speed
at t-24, and trailing 24 h and 168 h mean air temperature ending at t-24; plus
building, site and primary-use categories. No weather observed later than
t-24 is used, so the forecast needs no weather forecast as an input. The
weather file is read from the producer bundle and must match the SHA-256
registered in `dataset_manifest.json`.

```bash
.venv/bin/energy-forecast run \
  --reference ../energy-ai-data-onboarding/results/bdg2_v3/reference.csv.gz \
  --corrupted ../energy-ai-data-onboarding/results/bdg2_v3/corrupted.csv.gz \
  --remediated ../energy-ai-data-onboarding/results/bdg2_v3/remediated.csv.gz \
  --producer-manifest ../energy-ai-data-onboarding/results/bdg2_v3/dataset_manifest.json \
  --fault-manifest ../energy-ai-data-onboarding/results/bdg2_v3/fault_manifest.json \
  --config configs/experiment_v3.json \
  --output-dir results/canonical_24h_v3_repeat
```

## Reproducibility and interpretation

The canonical experiment is run by the CLI, not by hidden notebook state. Input
files and both producer manifests are verified before training. The committed
experiment configuration fixes train through 2016, reserves 2017 Q1 as a
validation partition, and tests on 2017 Q2–Q4; fractional boundaries exist only
for short deterministic fixtures. The reported models and hyperparameters were
fixed before the canonical run and were not selected on that reserved validation
partition. Any future tuning must be declared and reported as a separately
labeled experiment. Test targets are joined only from the reference condition.
Final metrics use the same finite timestamp/building keys for every
condition/model pair. `verify-results` checks result completeness, hashes, pair
coverage, target consistency, subgroup dimensions, and common-key alignment.

`verify-results` also recomputes aggregate metrics, per-building MASE, paired
differences, and deterministic SVG content from the prediction table before
accepting the recorded results.

See [LIMITATIONS.md](LIMITATIONS.md) and [DATA_CARD.md](DATA_CARD.md) for scope,
assumptions, and interpretation constraints.

Dataset licensing remains separate from this MIT-licensed code. See the companion repository's source manifest and the [Building Data Genome 2 project](https://github.com/buds-lab/building-data-genome-project-2).
