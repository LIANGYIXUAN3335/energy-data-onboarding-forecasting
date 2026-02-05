# Energy Data Onboarding and Demand Forecasting

[![verify](https://github.com/LIANGYIXUAN3335/energy-data-onboarding-forecasting/actions/workflows/verify.yml/badge.svg)](https://github.com/LIANGYIXUAN3335/energy-data-onboarding-forecasting/actions/workflows/verify.yml)
![python](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**A validated data-onboarding pipeline for U.S. building electricity meters, and a
controlled experiment showing that it protects a 24-hour-ahead demand forecast
from real-world data defects.**

Author: Yixuan Liang · Data: Building Data Genome 2 (public, 12 U.S. buildings,
2016–2017) · Everything here is reproducible from pinned public data with
SHA-256 hashes; see [Why you can trust the numbers](#why-you-can-trust-the-numbers).

---

## Why this exists

Electric utilities and grid operators are moving to AI-based forecasting to plan
generation, integrate renewables and keep the grid stable as demand from data
centers and electrification grows. The U.S. Department of Energy's
[Genesis Mission](https://www.energy.gov/genesis-mission) and the
[AI Action Plan](https://www.whitehouse.gov/articles/2025/07/white-house-unveils-americas-ai-action-plan/)
both single out one prerequisite: **AI-ready, validated data**. A forecasting
model is only as good as the meter data it learns from, and real meter data is
messy: sensors stick, readings drop to zero for hours, units change by a factor
of 100, timestamps go missing.

This project asks a concrete, testable question:

> Can an onboarding pipeline detect and repair meter-data defects *before* a
> model is trained, using only information that would have been available at
> the time, and does that measurably protect the forecast?

The answer, measured on real U.S. building data with seeded, hash-logged
defects, is yes: on corrupted training data a gradient-boosting forecaster's
error rises by 86%; after the pipeline's past-only repair it is back to, and
slightly below, the error on clean data, while keeping 96.7% of the training
targets. The first version of the pipeline got this wrong, and the project
kept that result too; see [Results](#results).

## What the system does

![Pipeline](docs/figures/pipeline.png)

1. **Pinned public source data.** Hourly electricity meters and site weather
   from [Building Data Genome 2 v1.0](https://doi.org/10.5281/zenodo.3887306)
   (Miller et al., *Scientific Data* 2020). File sizes and SHA-256 hashes are
   pinned in `energy-ai-data-onboarding/provenance/bdg2_v1.0.json`; a download
   that does not match is refused.
2. **Ingest and normalize.** Fixed output schema, chunked column-selective
   reads, metadata join, explicit timestamp contract.
3. **Quality gates.** Twenty named checks (gaps, duplicates, missing and
   negative loads, robust outliers, long zero runs, stuck sensors, unit and
   level shifts, metadata integrity, split coverage) with explicit thresholds.
   Run detectors are calibrated per building on clean data before the
   experiment cutoff, so a coarsely quantized meter is not mistaken for a stuck
   one; the applied thresholds are published in `detector_calibration.csv`.
4. **Seeded faults.** Known defects (missing value, negative value, spike,
   8-hour zero block, 8-hour stuck segment, 8-hour ×100 unit error; four events
   of each) are written into the training period only, and every affected key
   is logged with a hash. Nothing at or after the cutoff is ever touched, so
   the evaluation targets cannot be contaminated.
5. **Past-only remediation.** Short gaps are forward-filled from the last valid
   reading; longer gaps (up to 48 h) take the median of the same weekday and
   hour over the previous eight weeks; anything else is quarantined as null
   with a reason code. No repair ever reads a later value.
6. **Forecasting benchmark.** The same four models (seasonal naive, ridge,
   random forest, gradient boosting), same features, same chronological split
   and same seed are trained on each of three versions of the data,
   *reference*, *corrupted* and *remediated*, and evaluated on identical test
   hours whose targets always come from the reference data.
7. **Metrics and verification.** Pooled and per-building MASE, MAE with
   bootstrap intervals, paired remediated-minus-corrupted differences, and
   per-horizon results; every artifact is hash-bound and re-checked by
   `verify` commands.

## See it on one building

Real 2016 data from one education building in Florida (`Panther_education_Annetta`).
The animation steps through the seeded ×100 unit error (July) and the seeded
8-hour zero block (September), the quality gates flagging them, the past-only
repair, and the effect on the forecast.

![Fault, detection and remediation on one building](docs/figures/fault_to_remediation.gif)

## Results

Test period: April to December 2017, 79,178 hourly targets across 12 buildings,
targets always from the reference data. MASE is the mean absolute error scaled
by a seasonal-naive forecaster, so 1.0 equals "as good as repeating last week".

| Model | Reference | Corrupted | Remediated | Paired MAE difference, remediated − corrupted (95% CI) |
|---|---|---|---|---|
| Gradient boosting | 0.760 | 1.415 (+86%) | **0.723** | −42.9 kWh (−44.8 to −41.0) |
| Random forest | 0.741 | 1.040 (+40%) | **0.726** | −19.4 kWh (−20.5 to −18.3) |
| Ridge regression | 0.812 | 1.192 (+47%) | **0.804** | −24.0 kWh (−24.5 to −23.6) |
| Seasonal naive (baseline) | 1.040 | 1.040 | 1.040 | 0 (uses no training data) |

![Results](docs/figures/results.png)

Three things the numbers say:

- **Undetected defects hurt.** Twenty-four seeded events touching only 108 of
  105,408 training rows raised gradient-boosting error by 86%, because a single
  ×100 segment or zero block distorts what the model learns for that building
  (see the per-building panel: `Eagle_office_Amanda` goes from 36 to 251 kWh).
- **The pipeline recovers it.** After detection and past-only repair every
  learned model is back at, or slightly below, its clean-data error, and
  96.7% of training targets are retained.
- **The effect is stable across horizons.** In a separate 1-to-24-hour
  direct-forecast sensitivity run the same pattern holds at every horizon.

![Error by horizon](docs/figures/horizon.png)

### The negative result that came first

The pre-registered v2 run (`canonical_24h_v2_final`) reported the opposite:
remediation made the forecast 26% *worse*. The cause was traced in the
published artifacts: with fixed thresholds, the stuck-sensor and long-zero
rules quarantined 22,014 rows of *natural* data, including 97.9% of one
building's training year, and only 19 rows of seeded faults. The v3 amendment
(dated 2026-09-28, `experiments/protocol_v3_amendment.md` in each package)
calibrates the run detectors per building, flags whole runs instead of tails,
and adds the hour-of-week profile repair. Both bundles are kept and verified.

![v2 versus v3](docs/figures/v2_vs_v3.png)

Two honest footnotes. Adding lagged weather and a fuller calendar (30 features)
did not improve the clean-data gradient-boosting error (0.730 in v2 versus
0.760 in v3 with the same hyperparameters); the value of weather for day-ahead
load comes mostly from *forecast* weather, which this protocol deliberately
excludes. And one seeded stuck segment on a meter that repeats its reading
75% of the time is not detectable by construction and is reported as a miss
(`detector_metrics.csv`, recall 0.75).

## Why you can trust the numbers

- **Pinned inputs.** Source URLs, byte sizes and SHA-256 hashes for the three
  BDG2 files are committed; the downloader fails closed on mismatch.
- **Pre-registered protocol.** Split dates, features, models and
  hyperparameters were fixed before the runs (`experiments/protocol.md`); the
  2017 Q1 validation partition is reserved and unused. Changes are dated
  amendments, not silent edits.
- **No leakage by construction.** Every load- and weather-derived feature for a
  target at *t* is observed at *t − 24 h* or earlier; test targets are joined
  only from the reference data; preprocessing is fitted on training rows only;
  the fault injector and the consumer both enforce the cutoff.
- **Hash-bound artifacts.** Each run writes manifests with the hashes of its
  inputs, configuration, source tree and outputs. `energy-forecast
  verify-results` recomputes the metrics, intervals and figures from the stored
  predictions and binds the targets back to the producer bundle.
- **Tests and CI.** 81 tests (network-free, deterministic fixtures) and the
  verification of all five committed result bundles run on every push
  (`scripts/verify_all.sh`, GitHub Actions).
- **Determinism.** Re-running the pipeline reproduces the committed bundles
  byte for byte with the pinned interpreter and packages (`constraints.txt`).

## Timeline and relationship to earlier work

| When | What |
|---|---|
| Jan 2026 | Capstone prototype: a single-notebook onboarding-and-forecasting study on a small daily series (IIT; poster accepted at IMA 2026). Methods only; not reused here. |
| Sep 27–28, 2026 | This repository: two installable packages on real hourly BDG2 data, pre-registered v2 protocol, committed v2 baseline bundles (the negative result). |
| Sep 28, 2026 | v3 amendment: calibrated detectors, profile repair, weather artifact, four models; committed v3 bundles; public GitHub repository and CI. |

Planned next (in order): an adapter for U.S. grid-level data
([EIA-930](https://www.eia.gov/electricity/gridmonitor/) hourly demand by
balancing authority) so the same gates run on transmission-scale data; more
BDG2 sites and a second timezone; forecast-weather inputs as a separately
labeled experiment; a streaming ingestion interface with the same contracts;
a Zenodo DOI for each released bundle.

## Where this could be used

The pipeline is a research prototype, not an operational system. The pieces
that transfer directly are the data contract, the calibrated quality gates,
the hash-bound provenance chain and the three-condition evaluation design.
Any organization that trains models on meter or sensor streams, such as a
utility forecasting group, a building-energy analytics team or a grid
operator's data platform, can reuse that design to measure whether its own
cleaning rules help or hurt before those rules touch a production model.

## Quick start

```bash
scripts/bootstrap.sh          # creates .venv (prefers Python 3.12, applies constraints.txt)
scripts/verify_all.sh         # 81 tests + hash verification of all committed bundles
```

To reproduce a full run (about 5 minutes; downloads ~200 MB once):

```bash
.venv/bin/energy-onboard download \
  --manifest energy-ai-data-onboarding/provenance/bdg2_v1.0.json \
  --data-dir energy-ai-data-onboarding/data/raw
scripts/run_canonical.sh
```

Each package documents its own commands, contracts and limitations:
[`energy-ai-data-onboarding`](energy-ai-data-onboarding/README.md) and
[`energy-demand-forecasting`](energy-demand-forecasting/README.md).

## Repository layout

```text
energy-ai-data-onboarding/     source verification, quality gates, faults, remediation, weather
  results/bdg2_mvp_v2_hardened/   v2 baseline bundle
  results/bdg2_v3/                v3 bundle (calibrated detectors, weather)
energy-demand-forecasting/     features, four models, metrics, verification, sensitivity
  results/canonical_24h_v2_final/          v2 baseline (negative result)
  results/canonical_24h_v3/                v3 canonical run
  results/direct_1_to_24_sensitivity_v*/   1-24 h sensitivity runs
docs/figures/                  README figures and the script that regenerates them
scripts/                       bootstrap, canonical run, verify_all
constraints.txt                exact package versions behind the committed bundles
```

## Data policy

Only public building-energy measurements, public site weather and
reproducibility metadata are processed. The pipelines are stateless batch
programs; they store no personal data and send nothing anywhere.

## Citation and license

Code is MIT-licensed (see `LICENSE` in each package). Cite the software with
`CITATION.cff`. The data belong to the Building Data Genome 2 project
(CC BY 4.0): Miller, C. et al. *The Building Data Genome Project 2*,
Scientific Data 7, 368 (2020), https://doi.org/10.1038/s41597-020-00712-x.
