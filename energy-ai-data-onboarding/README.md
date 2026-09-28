# Energy AI Data Onboarding

This Python project turns a pinned Building Data Genome 2 (BDG2)
release into three auditable, forecasting-compatible conditions. It verifies
source identity, reads only preselected wide CSV columns in chunks, normalizes
metadata, runs quality gates, validates detectors with a separate structural
fault suite, adds value-only downstream faults before a fixed cutoff, performs
past-only remediation, quarantines unresolved values, and writes hashed
machine-readable artifacts and generated reports.

It is a local batch research prototype, not an operational grid service or
production platform.

## Data policy

The pipeline processes public building-energy measurements, public site
weather and reproducibility metadata only. It is a stateless batch program: it
stores no personal data and transmits nothing. See [PRIVACY.md](PRIVACY.md).

## Source

- Dataset: Building Data Genome 2, version 1.0
- DOI: <https://doi.org/10.5281/zenodo.3887306>
- Paper: <https://doi.org/10.1038/s41597-020-00712-x>
- Pinned repository: <https://github.com/buds-lab/building-data-genome-project-2/tree/v1.0>

`provenance/bdg2_v1.0.json` pins the raw electricity and metadata URLs, byte
sizes, and SHA-256 hashes. Full source files and generated result directories
are ignored by Git.

BDG2 raw meter timestamps are local wall-clock values. The committed subset is
restricted to three `US/Eastern` sites; the current CSV representation attaches
a UTC marker for stable parsing without doing an offset conversion. This is not
cross-timezone UTC harmonization. The limitation is recorded in every manifest
and report.

## Install and test

From the workspace root:

```bash
.venv/bin/pip install -e './energy-ai-data-onboarding[dev]'
.venv/bin/pytest energy-ai-data-onboarding
```

Tests create deterministic local fixtures and never access the network.

## Download the authoritative files

```bash
.venv/bin/energy-onboard download \
  --manifest energy-ai-data-onboarding/provenance/bdg2_v1.0.json \
  --data-dir energy-ai-data-onboarding/data/raw
```

Downloads stream to temporary files, fail closed on size or SHA-256 mismatch,
and do not replace a verified file unless `--force` is explicitly supplied.
The URLs target Git LFS media objects, not pointer text.

## Run the committed BDG2 subset

```bash
.venv/bin/energy-onboard run \
  --config energy-ai-data-onboarding/configs/bdg2_mvp_v2.json \
  --electricity-csv energy-ai-data-onboarding/data/raw/electricity.csv \
  --metadata-csv energy-ai-data-onboarding/data/raw/metadata.csv \
  --output-dir energy-ai-data-onboarding/results/bdg2_mvp_v2_repeat

.venv/bin/energy-onboard verify \
  --result-dir energy-ai-data-onboarding/results/bdg2_mvp_v2_repeat
```

`configs/bdg2_mvp.json` is the superseded first-iteration configuration and is
kept only for comparison. All committed results and `scripts/run_canonical.sh`
use `bdg2_mvp_v2.json`, which adds explicit per-check count/rate thresholds and
a fail-closed reference gate. A run must use a new or empty
output directory; this prevents a blocked run from being mixed with previously
published artifacts.

Two committed bundles exist:

- `results/bdg2_mvp_v2_hardened`: the pre-registered v2 baseline (fixed
  run-detector thresholds, forward-fill-only remediation, one seeded event per
  fault family, no weather).
- `results/bdg2_v3`: the v3 amendment dated 2026-09-28
  (`configs/bdg2_v3.json`, `experiments/protocol_v3_amendment.md`): whole-run
  flagging, per-building stuck-sensor calibration on the reference data,
  profile-aware zero-run detection, past-only hour-of-week profile repair,
  four seeded events per fault family, and the published site weather
  artifact. Run it with the additional `--weather-csv` argument:

```bash
.venv/bin/energy-onboard run \
  --config energy-ai-data-onboarding/configs/bdg2_v3.json \
  --electricity-csv energy-ai-data-onboarding/data/raw/electricity.csv \
  --metadata-csv energy-ai-data-onboarding/data/raw/metadata.csv \
  --weather-csv energy-ai-data-onboarding/data/raw/weather.csv \
  --output-dir energy-ai-data-onboarding/results/bdg2_v3_repeat
```

The configuration explicitly names twelve buildings—two Education and two
Office buildings at each of Eagle, Panther, and Rat. Selection is declared
before modeling and does not inspect forecast accuracy. A repeatable CLI
override is available as `--building-id ID`.

`--fixture-mode` exists only for software integration tests. It bypasses BDG2
release identity checks and labels every generated artifact `fixture_test_only`;
fixture outputs are not project findings. It still requires an existing source
declaration so the run cannot silently lose provenance context.

## Output contract

All condition files contain exactly:

```text
timestamp,building_id,load,site_id,primary_use,condition
```

- `reference.csv.gz`: normalized source values.
- `corrupted.csv.gz`: reproducible value-only defects before the cutoff.
- `remediated.csv.gz`: causal remediation; unresolved values remain as nulls.

The three files have exactly the same `(timestamp, building_id)` keys. Values at
and after the fault cutoff are identical to `reference`; downstream evaluation
targets therefore cannot be overwritten by this pipeline.

Additional artifacts are:

- `quality_issues.csv.gz`, `quality_summary.json`;
- `remediation_log.csv.gz`, `quarantine.csv.gz`;
- `fault_manifest.json`, `detector_metrics.csv`;
- `detector_calibration.csv`: the per-building run-detector thresholds that
  were actually applied and the statistics they were derived from;
- `weather.csv.gz` (v3): site weather on the same hourly grid, columns
  `timestamp,site_id,air_temperature,dew_temperature,wind_speed,cloud_coverage,precip_depth_1hr,sea_lvl_pressure`,
  nulls preserved, never touched by the fault suites;
- `dataset_manifest.json`, `run_manifest.json`;
- generated `data_card.md`, `report.md`, and `report.html`.

`dataset_manifest.json` has a canonical top-level `files` map. Every registered
artifact has a relative path, SHA-256, byte size, and—when tabular—row count and
columns. It also records source/config hashes, subset, exclusions, counts,
runtime versions, seed, cutoff, time semantics, and stateless data policy.

## Fault and remediation discipline

The detector-validation suite contains structural and value defects, including
missing intervals, duplicates, timestamp shifts, and metadata mismatch. It is
used only to calculate precision, recall, and false-positive rate against known
seeded ground truth. Source alerts outside seeded locations are conservatively
counted outside that ground truth; because BDG2 has no complete natural-anomaly
labels, they must not automatically be interpreted as true false positives.

The separate downstream suite preserves every key and includes missing values,
negative values, spikes, long zero blocks, stuck segments, and unit-scale
segments. All changes are strictly before the committed cutoff and each key,
original value, corrupted value, type, and severity is recorded. Each seeded
event also has a deterministic `fault_id`, start/end range, affected-key count,
and SHA-256 digest; verification recomputes these fields from the per-key rows.

Remediation does not consult fault ground truth and never backfills or
interpolates from a future value. The v2 rule forward-fills at most three
hours from the last valid reading and quarantines everything else. The v3 rule
adds a second, still past-only step: gaps of up to 48 hours take the median of
the same weekday and hour over the previous eight weeks, computed only from
earlier, originally valid readings; a suspected unit shift receives that
profile value rather than being divided by an inferred factor. Longer gaps
stay null and quarantined. Negative or adverse experimental outcomes remain
reportable.

Run detectors are calibrated per building in v3: the stuck-sensor threshold is
raised to just above the 0.999 quantile of the building's natural constant-run
lengths (computed on the reference condition strictly before the fault cutoff),
and a zero run is only flagged during hours when the building is normally
active. The applied thresholds are published in `detector_calibration.csv`.

See [the data contract](docs/data_contract.md),
[architecture](docs/architecture.md), and [limitations](LIMITATIONS.md).
