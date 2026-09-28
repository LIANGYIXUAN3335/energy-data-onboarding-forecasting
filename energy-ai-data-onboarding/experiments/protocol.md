# Pre-specified onboarding experiment protocol

## Objective

Measure whether known training-period data defects are detected and whether a
deterministic, causal remediation recovers a forecast-ready dataset without
touching evaluation targets.

## Fixed conditions

- C0 / reference: normalized source observations.
- C1 / corrupted: C0 plus seeded faults strictly before the cutoff.
- C3 / remediated: C1 after the published repair and quarantine rules.

The detection-only C2 condition is represented by C1 plus its issue records;
the data values are not silently modified.

## Fixed fault families

Missing intervals, duplicate timestamps, negative values, spikes, long-zero
runs, unit scaling, metadata mismatch, and timezone shifts. The run manifest
records the seed, cutoff, selected observations, and before/after values.

## Invariants

- Do not choose the seed, cutoff, thresholds, or models after looking at
  downstream test performance.
- Do not inject any fault at or after the cutoff.
- Do not use future observations during remediation.
- Retain null or adverse results.
- Report detector counts and remediation/quarantine counts for every condition.

