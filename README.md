# Energy Data Onboarding and Demand Forecasting

This workspace contains two independently runnable Python projects built around
the Building Data Genome 2 (BDG2) public dataset:

- [`energy-ai-data-onboarding`](energy-ai-data-onboarding/README.md) verifies
  source files, normalizes building-load data, runs quality checks, injects
  deterministic test faults, applies past-only remediation, and publishes
  versioned datasets with manifests.
- [`energy-demand-forecasting`](energy-demand-forecasting/README.md) consumes
  those datasets and compares fixed 24-hour-ahead forecasting methods across
  reference, corrupted, and remediated training conditions.

The two projects form one reproducible workflow:

```text
BDG2 source files
    -> verified and normalized datasets
    -> quality checks, fault injection, and remediation
    -> chronological demand-forecasting experiments
    -> metrics, predictions, figures, and run manifests
```

## Setup

```bash
scripts/bootstrap.sh
```

The setup script creates `.venv` and installs both packages in editable mode.
It prefers a Python 3.12 interpreter (`PYTHON_BIN=... scripts/bootstrap.sh`
to choose one) and, on 3.12, applies `constraints.txt` so the environment
matches the one that produced the committed result bundles (CPython 3.12.14,
numpy 2.3.5, pandas 2.2.3, scikit-learn 1.9.1). Other interpreters from 3.11
up work with the latest compatible versions.

The BDG2 source files are not versioned. Download and hash-verify them with:

```bash
.venv/bin/energy-onboard download \
  --manifest energy-ai-data-onboarding/provenance/bdg2_v1.0.json \
  --data-dir energy-ai-data-onboarding/data/raw
```

## Run and verify

- `scripts/run_canonical.sh` runs the onboarding pipeline followed by the
  canonical forecasting experiment.
- `scripts/verify_all.sh` runs both test suites and verifies the committed final
  result bundles (`energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened`,
  `energy-demand-forecasting/results/canonical_24h_v2_final`, and
  `energy-demand-forecasting/results/direct_1_to_24_sensitivity_v1_final`),
  which are versioned so that anyone can re-check their hashes and metrics.

See each project's README for individual commands, data contracts, experiment
details, outputs, and limitations.

## State and privacy

Both projects are local, stateless batch pipelines. They do not implement or
retain ChatGPT or LLM memory, conversation history, prompts, embeddings, vector
databases, user profiles, personal data, or telemetry.
