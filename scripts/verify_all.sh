#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
VENV_DIR="$WORKSPACE_DIR/.venv"

"$VENV_DIR/bin/pytest" -q "$WORKSPACE_DIR/energy-ai-data-onboarding"
"$VENV_DIR/bin/pytest" -q "$WORKSPACE_DIR/energy-demand-forecasting"
"$VENV_DIR/bin/energy-onboard" verify \
  --result-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened"
"$VENV_DIR/bin/energy-forecast" verify-inputs \
  --input-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened"
"$VENV_DIR/bin/energy-forecast" verify-results \
  --result-dir "$WORKSPACE_DIR/energy-demand-forecasting/results/canonical_24h_v2_final" \
  --input-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened"
"$VENV_DIR/bin/energy-forecast" verify-sensitivity-results \
  --result-dir "$WORKSPACE_DIR/energy-demand-forecasting/results/direct_1_to_24_sensitivity_v1_final" \
  --input-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened"

# v3 bundles (protocol amendment of 2026-09-28)
"$VENV_DIR/bin/energy-onboard" verify \
  --result-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_v3"
"$VENV_DIR/bin/energy-forecast" verify-inputs \
  --input-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_v3"
"$VENV_DIR/bin/energy-forecast" verify-results \
  --result-dir "$WORKSPACE_DIR/energy-demand-forecasting/results/canonical_24h_v3" \
  --input-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_v3"
"$VENV_DIR/bin/energy-forecast" verify-sensitivity-results \
  --result-dir "$WORKSPACE_DIR/energy-demand-forecasting/results/direct_1_to_24_sensitivity_v3" \
  --input-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_v3"

echo "All tests and artifact verifications passed."
