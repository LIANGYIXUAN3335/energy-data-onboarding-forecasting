from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd


def _format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    formatted = metrics.copy()
    for column in [
        "mae",
        "rmse",
        "wmape",
        "mase",
        "mase_macro_building",
        "r2",
        "mae_ci_low",
        "mae_ci_high",
        "data_retained",
    ]:
        formatted[column] = formatted[column].map(
            lambda value: "NA" if pd.isna(value) else f"{value:.4f}"
        )
    return formatted


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_reports(
    metrics: pd.DataFrame,
    subgroup: pd.DataFrame,
    paired: pd.DataFrame,
    manifest: dict[str, Any],
    output_dir: Path,
) -> None:
    shown = _format_metrics(metrics)
    paired_shown = paired.copy()
    for column in [
        "corrupted_mae",
        "remediated_mae",
        "mean_difference",
        "difference_ci_low",
        "difference_ci_high",
    ]:
        paired_shown[column] = paired_shown[column].map(
            lambda value: "NA" if pd.isna(value) else f"{value:.4f}"
        )
    split = manifest["split"]
    markdown = f"""# Energy Demand Forecasting Experiment Report

Generated from machine-readable results for a retrospective public-data experiment.

## Protocol summary

- Train end: `{split['train_end']}`
- Validation end: `{split['validation_end']}`
- Test end: `{split['test_end']}`
- Validation use: reserved; canonical models and hyperparameters fixed a priori
- Common evaluation keys: `{manifest['common_prediction_count']}`
- Prediction rows across all condition/model pairs: `{manifest['prediction_rows']}`
- Seed: `{manifest['seed']}`
- Test-target source: `reference`
- State policy: stateless; no conversational state or personal data

## Complete primary results

{_markdown_table(shown)}

`mase` remains the canonical pooled MASE-168 primary metric. The supplementary
`mase_macro_building` is the unweighted mean of finite building-specific MASE
values; every denominator comes from that building's reference training history.
Coverage and undefined-denominator counts are reported alongside it and in
`building_mase.csv`.

![MAE by condition and model](figures/mae_by_condition.svg)

![Supplementary macro building MASE](figures/macro_building_mase.svg)

## Paired corrupted-versus-remediated differences

`mean_difference` is remediated absolute error minus corrupted absolute error
on identical test keys. Negative values favor remediation; positive values do not.

{_markdown_table(paired_shown)}

## Coverage and subgroup robustness

`subgroup_metrics.csv` reports site, primary-use, building, and calendar-quarter
subgroups. These are coverage and robustness slices, not a fairness assessment.

## Interpretation boundary

The table includes every configured condition and model. A lower error is better, but causal interpretation is limited to the documented seeded fault suite and deterministic remediation. Natural source-data comparisons remain observational. Consult `subgroup_metrics.csv`, `predictions.csv.gz`, and `run_manifest.json` for the complete record.
"""
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")

    table_html = shown.to_html(index=False, border=0, escape=True)
    paired_html = paired_shown.to_html(index=False, border=0, escape=True)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Energy Demand Forecasting Experiment Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #161616; }}
    h1, h2 {{ color: #000; }}
    table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
    th, td {{ border: 1px solid #d9d9d9; padding: .55rem; text-align: right; }}
    th:first-child, th:nth-child(2), td:first-child, td:nth-child(2) {{ text-align: left; }}
    th {{ background: #17365d; color: #fff; }}
    tbody tr:nth-child(even) {{ background: #f4f7fb; }}
    .meta {{ color: #444; }}
  </style>
</head>
<body>
  <h1>Energy Demand Forecasting Experiment Report</h1>
  <p class="meta">Generated from machine-readable results for a retrospective public-data experiment.</p>
  <h2>Protocol summary</h2>
  <ul>
    <li>Train end: {html.escape(split['train_end'])}</li>
    <li>Validation end: {html.escape(split['validation_end'])}</li>
    <li>Test end: {html.escape(split['test_end'])}</li>
    <li>Validation use: reserved; canonical models and hyperparameters fixed a priori</li>
    <li>Common evaluation keys: {manifest['common_prediction_count']}</li>
    <li>Prediction rows: {manifest['prediction_rows']}</li>
    <li>Seed: {manifest['seed']}</li>
    <li>Test-target source: reference</li>
    <li>State policy: stateless; no conversational state or personal data</li>
  </ul>
  <h2>Complete primary results</h2>
  {table_html}
  <p><code>mase</code> remains the canonical pooled MASE-168 primary metric.
  <code>mase_macro_building</code> is supplementary and averages finite
  building-specific MASE values whose denominators come only from reference
  training history. Coverage is retained explicitly in the table and
  <code>building_mase.csv</code>.</p>
  <img src="figures/mae_by_condition.svg" alt="MAE by condition and model" style="max-width:100%;height:auto">
  <img src="figures/macro_building_mase.svg" alt="Supplementary macro building MASE" style="max-width:100%;height:auto">
  <h2>Paired corrupted-versus-remediated differences</h2>
  <p>The signed difference is remediated absolute error minus corrupted absolute error on identical test keys. Negative values favor remediation.</p>
  {paired_html}
  <h2>Coverage and subgroup robustness</h2>
  <p>Subgroup results cover site, primary use, building, and calendar quarter. They are robustness slices, not a fairness assessment.</p>
  <h2>Interpretation boundary</h2>
  <p>The table includes every configured condition and model. Lower error is better. Causal interpretation is limited to the documented seeded fault suite and deterministic remediation. Natural source-data comparisons remain observational.</p>
</body>
</html>
"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")

    model_card = f"""# Run-Specific Model Card

## Intended use

This run is a retrospective public-data benchmark for measuring how documented
training-data conditions affect hourly building-load forecasts. It is not an
operational utility forecast, dispatch system, safety system, or deployment claim.

## Models

- Seasonal naive: lag 168, then lag 24 fallback. No one-hour fallback is used.
- Histogram gradient boosting: fixed configuration recorded in `run_manifest.json`.

Both models use the same chronological split and reference-only test targets
across all data conditions. Preprocessing for the learned model is fitted on its
training partition only. For a target at `t`, all load-derived features use
`t-24` or earlier.

## Evaluation

- Train end: `{split['train_end']}`
- Validation end: `{split['validation_end']}`
- Test end: `{split['test_end']}`
- Validation use: reserved; canonical models and hyperparameters fixed a priori.
- Primary metric (unchanged): pooled MASE with the recorded 168-hour
  reference-training scale.
- Supplementary metric: per-building MASE using building-specific
  reference-training scales and its finite-building macro average. Undefined
  building scales remain visible and are not silently treated as zero.
- Supporting metrics: MAE, RMSE, aggregate wMAPE, R-squared, bootstrap MAE
  interval, paired absolute-error difference, and subgroup robustness slices.
- Common evaluation keys: `{manifest['common_prediction_count']}`

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
`run_manifest.json`. The pipeline is stateless and retains no conversational
state or personal data.
"""
    (output_dir / "model_card.md").write_text(model_card, encoding="utf-8")
