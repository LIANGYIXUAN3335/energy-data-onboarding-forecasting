#!/usr/bin/env bash
set -euo pipefail

# Runs the onboarding pipeline on the committed BDG2 subset and then the
# canonical fixed-24h forecasting experiment into fresh output directories.
# The committed final bundles (results/bdg2_mvp_v2_hardened and
# results/canonical_24h_v2_final) are never overwritten by this script.

WORKSPACE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
VENV_DIR="$WORKSPACE_DIR/.venv"
ONBOARD_REPO="$WORKSPACE_DIR/energy-ai-data-onboarding"
FORECAST_REPO="$WORKSPACE_DIR/energy-demand-forecasting"
ONBOARD_RESULTS="${ONBOARD_RESULTS:-$ONBOARD_REPO/results/bdg2_mvp_v2_run}"
FORECAST_RESULTS="${FORECAST_RESULTS:-$FORECAST_REPO/results/canonical_24h_v2_run}"

test -x "$VENV_DIR/bin/energy-onboard" || { echo "missing venv; run scripts/bootstrap.sh first" >&2; exit 1; }
test -x "$VENV_DIR/bin/energy-forecast"
test -f "$ONBOARD_REPO/data/raw/electricity.csv" || {
  echo "missing $ONBOARD_REPO/data/raw/electricity.csv; run:" >&2
  echo "  $VENV_DIR/bin/energy-onboard download --manifest $ONBOARD_REPO/provenance/bdg2_v1.0.json --data-dir $ONBOARD_REPO/data/raw" >&2
  exit 1
}
test -f "$ONBOARD_REPO/data/raw/metadata.csv"

"$VENV_DIR/bin/energy-onboard" run \
  --config "$ONBOARD_REPO/configs/bdg2_mvp_v2.json" \
  --electricity-csv "$ONBOARD_REPO/data/raw/electricity.csv" \
  --metadata-csv "$ONBOARD_REPO/data/raw/metadata.csv" \
  --output-dir "$ONBOARD_RESULTS"

"$VENV_DIR/bin/energy-onboard" verify --result-dir "$ONBOARD_RESULTS"
"$VENV_DIR/bin/energy-forecast" verify-inputs --input-dir "$ONBOARD_RESULTS"

"$VENV_DIR/bin/energy-forecast" run \
  --reference "$ONBOARD_RESULTS/reference.csv.gz" \
  --corrupted "$ONBOARD_RESULTS/corrupted.csv.gz" \
  --remediated "$ONBOARD_RESULTS/remediated.csv.gz" \
  --producer-manifest "$ONBOARD_RESULTS/dataset_manifest.json" \
  --fault-manifest "$ONBOARD_RESULTS/fault_manifest.json" \
  --config "$FORECAST_REPO/configs/experiment.json" \
  --output-dir "$FORECAST_RESULTS"

"$VENV_DIR/bin/energy-forecast" verify-results \
  --result-dir "$FORECAST_RESULTS" \
  --input-dir "$ONBOARD_RESULTS"

echo "Canonical run complete: $ONBOARD_RESULTS and $FORECAST_RESULTS"
