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

The pipeline is stateless. It contains public energy data, public site
weather and reproducibility metadata only; it stores no personal data.
"""


def _findings_paragraph(summary: dict[str, Any]) -> str:
    """State what this run found, in sentences generated from the summary."""

    gates = summary["quality_gates"]
    remediation = summary["remediation"]
    downstream = summary["downstream_faults"]
    detector = summary["detector_validation"]
    coverage = summary["coverage"]
    training_rows = downstream.get("training_rows_before_cutoff")
    seeded = downstream["fault_records"]
    share = f" ({seeded / training_rows:.2%} of the {training_rows:,} training rows)" if training_rows else ""
    recalls = detector.get("recall_by_type") or {}
    missed = sorted(name for name, value in recalls.items() if value is not None and value < 1)
    recall_text = (
        "every seeded fault family was fully detected"
        if recalls and not missed
        else "seeded faults were fully detected except "
        + ", ".join(f"{name} (recall {recalls[name]:.2f})" for name in missed)
        if missed
        else "detector recall is reported in `detector_metrics.csv`"
    )
    events = downstream.get("fault_events")
    event_text = f"{events} seeded events, " if events else ""
    return (
        f"The reference data ({coverage['rows']:,} hourly rows, {coverage['buildings']} buildings) "
        f"passed publication with gate status `{gates['reference']['status']}`. "
        f"{event_text}{seeded} value-level fault rows{share} were written before the cutoff; "
        f"{recall_text}. Remediation repaired {remediation['repaired_rows']:,} rows and "
        f"quarantined {remediation['quarantined_rows']:,}; the gate on the remediated condition "
        f"is `{gates['remediated']['status']}`. Every count here is recomputed from the committed "
        "artifacts, and adverse outcomes are kept."
    )


def _policy_paragraph(summary: dict[str, Any]) -> str:
    policy = summary.get("detector_policy")
    repair = summary.get("remediation_policy")
    if not policy or not repair:
        return (
            "Run detectors use fixed thresholds (v2 policy) and remediation is a "
            "past-only forward fill with quarantine of everything it cannot cover."
        )
    if policy.get("run_flagging") == "whole_run":
        detector_text = (
            "Run detectors flag every reading of a run once it reaches its threshold. "
            f"The stuck-sensor threshold ({policy['stuck_threshold_configured']} h) is raised per "
            "building to just above the "
            f"{policy['stuck_calibration_quantile']:g} quantile of that building's natural "
            "constant-run lengths (`detector_calibration.csv`), so coarsely quantized meters are "
            "not mistaken for stuck sensors. A zero run of at least "
            f"{policy['long_zero_threshold']} h counts only during hours when the building is "
            "normally active (causal hour-of-week median over the previous "
            f"{policy['profile_weeks']} weeks above {policy['zero_profile_min_ratio']:g} times the "
            "trailing 168-hour median). Calibration uses the reference condition strictly before "
            "the fault cutoff, so seeded faults never shape their own thresholds."
        )
    else:
        detector_text = (
            "Run detectors flag only run positions at or beyond a fixed threshold (v2 policy)."
        )
    if repair.get("repair_strategy") == "forward_then_profile":
        repair_text = (
            f"Remediation first forward-fills gaps of at most {repair['maximum_forward_fill_hours']} h "
            "from the last valid reading, then fills gaps of at most "
            f"{repair['maximum_profile_fill_hours']} h with the median of the same weekday and hour "
            f"over the previous {repair['profile_weeks']} weeks, using only earlier, originally "
            "valid readings. Longer gaps are quarantined as null."
            + (
                " Suspected unit or level shifts are replaced by that profile value rather than "
                "divided by an inferred factor."
                if repair.get("profile_fill_ambiguous_scale")
                else " Suspected unit or level shifts are always quarantined."
            )
        )
    else:
        repair_text = (
            f"Remediation forward-fills gaps of at most {repair['maximum_forward_fill_hours']} h "
            "from the last valid reading and quarantines everything else as null."
        )
    return detector_text + "\n\n" + repair_text


def _weather_paragraph(summary: dict[str, Any]) -> str:
    weather = summary.get("weather")
    if not weather:
        return "No weather artifact was published in this run."
    rates = weather.get("missing_rate_by_site", {})
    worst = max(
        (rate for site in rates.values() for rate in site.values()), default=0.0
    )
    return (
        f"Site weather (`weather.csv.gz`, {weather['rows']} rows for "
        f"{', '.join(weather['sites'])}) is published on the same hourly grid as the meters. "
        f"{weather['duplicate_keys_averaged']} duplicate source keys were averaged and "
        f"{weather['grid_hours_materialized_as_null']} missing grid hours were materialized as null; "
        f"the highest per-variable missing rate is {worst:.1%} (cloud cover). Weather is never "
        "modified by the fault suites and is imputed downstream only inside training-fitted "
        "preprocessing."
    )


def render_report(summary: dict[str, Any]) -> str:
    detector = summary["detector_validation"]
    remediation = summary["remediation"]
    quarantine = summary["quarantine"]
    return f"""# Energy-data onboarding audit report

{PROJECT_STATEMENT}

Created: `{summary['created_at_utc']}`

## Findings in brief

{_findings_paragraph(summary)}

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

## Detector and remediation policy

{_policy_paragraph(summary)}

{_weather_paragraph(summary)}

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

Only public dataset content, public site weather and reproducibility
metadata are processed. No personal information is retained and nothing is
transmitted.
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
