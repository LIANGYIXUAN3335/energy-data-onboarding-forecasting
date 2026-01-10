"""Generate narrative artifacts from machine-readable pipeline summaries."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


PROJECT_STATEMENT = (
    "This is a reproducible public-data research pipeline for building-energy "
    "data onboarding and forecasting preparation."
)


def _counts(summary: dict[str, Any]) -> str:
    gates = summary["quality_gates"]
    return ", ".join(
        f"{condition}={details['status']}"
        for condition, details in sorted(gates.items())
    )


def render_data_card(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    return f"""# Data card: {summary['dataset']['title']} onboarding output

{PROJECT_STATEMENT}

## Identity and scope

- Source version: `{summary['dataset']['version']}`
- DOI: `{summary['dataset']['doi']}`
- Execution scope: `{summary['execution_scope']}`
- Selected buildings: {coverage['buildings']}
- Selected rows: {coverage['rows']}
- Time span: `{coverage['timestamp_min']}` through `{coverage['timestamp_max']}`
- Sites: {', '.join(sorted(coverage['by_site']))}
- Primary-use categories: {', '.join(sorted(coverage['by_primary_use']))}

## Timestamp semantics

{summary['timestamp_semantics']}

## Conditions

`reference`, `corrupted`, and `remediated` have identical
`(timestamp, building_id)` key sets. Corruption is seeded, value-only, and
strictly earlier than `{summary['fault_cutoff']}`. Remediation is deterministic
and past-only; unresolved values remain present as nulls and are quarantined.

## Quality

Gate outcomes: {_counts(summary)}. See `quality_summary.json`,
`quality_issues.csv.gz`, and `detector_metrics.csv` for generated measurements.
These are coverage and sensor-integrity observations, not a demographic fairness
assessment.

## Intended use

Reproducible research on data readiness and downstream forecasting sensitivity.
This is not an operational utility dataset or a production service.

## Privacy

The pipeline is stateless. It contains public energy data and reproducibility
metadata only; it has no conversational memory, prompt store, embeddings,
personal profiles, analytics, or personal data.
"""


def render_report(summary: dict[str, Any]) -> str:
    detector = summary["detector_validation"]
    remediation = summary["remediation"]
    quarantine = summary["quarantine"]
    return f"""# Energy-data onboarding audit report

{PROJECT_STATEMENT}

Created: `{summary['created_at_utc']}`

## Run scope

The pipeline processed `{summary['coverage']['rows']}` long-form observations
from `{summary['coverage']['buildings']}` selected buildings. The source is
`{summary['dataset']['title']}` `{summary['dataset']['version']}`. This run is
classified as `{summary['execution_scope']}`.

## Source and time handling

{summary['timestamp_semantics']}

The source bytes, configuration, condition datasets, logs, and reports are
identified by SHA-256 in `dataset_manifest.json`.

## Quality gates

Generated outcomes: {_counts(summary)}. A fail or warning remains in the
machine-readable issue log; it is not suppressed to create a preferred result.

## Seeded detector validation

The isolated detector-validation suite injected `{detector['fault_records']}`
key-level fault records. Precision, recall, false-positive rate, and counts by
fault family are in `detector_metrics.csv`. Natural anomalies were not treated
as seeded ground truth.

## Downstream condition preparation

The downstream suite injected `{summary['downstream_faults']['fault_records']}`
value-level fault records before `{summary['fault_cutoff']}`. It did not add,
remove, or move keys. Past-only remediation logged `{remediation['log_rows']}`
actions, including `{remediation['repaired_rows']}` repaired rows and
`{remediation['quarantined_rows']}` unresolved actions. The quarantine artifact
contains `{quarantine['rows']}` rows. A suspected unit scale is never silently
corrected by dividing by an inferred factor.

## Limitations

- This is a local batch research prototype, not real-time or production-grade.
- The code does not evaluate operational utility deployment, grid control,
  production reliability, or economic impact.
- Faults are synthetic stress tests and are not estimates of real-world fault
  prevalence.
- Detector thresholds and simple causal remediation are transparent baselines,
  not an assertion of domain-optimal cleaning.
- Raw BDG2 timestamps require the source-timezone caveat stated above.

## Privacy and data boundary

Only public dataset content and reproducibility metadata are processed. No
personal information, conversation, prompt, model memory, embedding, vector
database, user profile, or telemetry is retained.
"""


def write_reports(output_dir: str | Path, summary: dict[str, Any]) -> None:
    directory = Path(output_dir)
    report = render_report(summary)
    data_card = render_data_card(summary)
    (directory / "report.md").write_text(report, encoding="utf-8")
    (directory / "data_card.md").write_text(data_card, encoding="utf-8")
    payload = html.escape(json.dumps(summary, indent=2, sort_keys=True))
    report_html = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Energy-data onboarding audit</title>"
        "<style>body{max-width:960px;margin:2rem auto;font:16px/1.5 system-ui;}"
        "pre{white-space:pre-wrap;background:#f5f5f5;padding:1rem}</style></head>"
        f"<body><h1>Energy-data onboarding audit</h1><pre>{html.escape(report)}</pre>"
        f"<details><summary>Machine-readable summary</summary><pre>{payload}</pre>"
        "</details></body></html>"
    )
    (directory / "report.html").write_text(report_html, encoding="utf-8")
