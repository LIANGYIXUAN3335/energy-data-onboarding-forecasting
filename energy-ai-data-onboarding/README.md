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

## Stateless privacy boundary

The pipeline has no LLM or chat integration. It does not retain conversation
history, prompts, user profiles, embeddings, vector indexes, browsing history,
analytics, personal information, or telemetry. It does not transmit local
results. See [PRIVACY.md](PRIVACY.md).

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

The completed final optimized bundle is
`results/bdg2_mvp_v2_hardened`.

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

Remediation does not consult fault ground truth. It uses at most the configured
number of earlier forward-fill hours; it never backfills or interpolates from a
future value. Ambiguous scale shifts and unresolved problems are left null and
quarantined. Negative or adverse experimental outcomes remain reportable.

See [the data contract](docs/data_contract.md),
[architecture](docs/architecture.md), and [limitations](LIMITATIONS.md).
