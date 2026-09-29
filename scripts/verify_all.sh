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
  --input-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened" \
  --expected-models seasonal_naive,hist_gradient_boosting
"$VENV_DIR/bin/energy-forecast" verify-sensitivity-results \
  --result-dir "$WORKSPACE_DIR/energy-demand-forecasting/results/direct_1_to_24_sensitivity_v1_final" \
  --input-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_mvp_v2_hardened" \
  --expected-models seasonal_naive,hist_gradient_boosting

# v3 bundles (protocol amendment of 2026-09-28)
"$VENV_DIR/bin/energy-onboard" verify \
  --result-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_v3"
"$VENV_DIR/bin/energy-forecast" verify-inputs \
  --input-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_v3"
"$VENV_DIR/bin/energy-forecast" verify-results \
  --result-dir "$WORKSPACE_DIR/energy-demand-forecasting/results/canonical_24h_v3" \
  --input-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_v3" \
  --expected-models seasonal_naive,ridge,random_forest,hist_gradient_boosting
"$VENV_DIR/bin/energy-forecast" verify-sensitivity-results \
  --result-dir "$WORKSPACE_DIR/energy-demand-forecasting/results/direct_1_to_24_sensitivity_v3" \
  --input-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_v3" \
  --expected-models seasonal_naive,ridge,hist_gradient_boosting

# v3 at literature-calibrated fault prevalence (docs/fault_prevalence.md)
"$VENV_DIR/bin/energy-onboard" verify \
  --result-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_v3_dense"
"$VENV_DIR/bin/energy-forecast" verify-inputs \
  --input-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_v3_dense"
"$VENV_DIR/bin/energy-forecast" verify-results \
  --result-dir "$WORKSPACE_DIR/energy-demand-forecasting/results/canonical_24h_v3_dense" \
  --input-dir "$WORKSPACE_DIR/energy-ai-data-onboarding/results/bdg2_v3_dense" \
  --expected-models seasonal_naive,ridge,random_forest,hist_gradient_boosting

echo "All tests and artifact verifications passed."
