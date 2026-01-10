# Limitations

- This is a retrospective public-data benchmark, not an operational utility,
  grid-dispatch, billing, or safety system.
- The primary experiment uses historical hourly meter load and public metadata.
  It excludes observed future weather and does not demonstrate a deployable
  weather-forecast integration.
- The planned Phase 1 Panther/Eagle/Rat subset uses one source timezone
  (US/Eastern). BDG2 raw timestamps are local wall-clock values. Calendar
  features preserve that local clock; a canonical UTC marker in producer files
  does not imply a true offset conversion. This design avoids mixing source
  timezones but does not validate daylight-saving transitions or generalize to
  a multi-timezone deployment.
- The chronological split is one fixed holdout. It is not a multi-region or
  prospective field validation.
- The 2017 Q1 validation partition is reserved but unused in the canonical run
  because model choices and hyperparameters are fixed a priori.
  This run therefore does not demonstrate a tuning or model-selection study.
- The seasonal-naive and histogram-gradient-boosting models are deliberately
  compact controls. Their presence does not establish that the selected model
  class is optimal.
- The canonical primary MASE uses one documented pooled 168-hour scale derived
  from reference training data. Supplementary per-building and macro-building
  MASE improve visibility but may exclude buildings whose reference-training
  seasonal denominator is undefined; coverage counts are therefore required.
- Row-level bootstrap intervals summarize sampling variation in the evaluated
  rows. Serial dependence can make them narrower than intervals from a blocked
  time-series bootstrap; they are therefore supporting, not dispositive,
  uncertainty estimates.
- The separate 1–24 hour sensitivity uses a whole building-week block
  bootstrap, but it remains one retrospective split and does not replace a
  prospective or multi-region validation. Its results must not replace the
  canonical fixed-h=24 primary analysis.
- R-squared is supplementary and may be negative. It is undefined for a
  constant target and must not replace the pre-specified primary metric.
- Site, primary-use, building, and quarter results are coverage and subgroup
  robustness slices. They are not a fairness analysis, and small groups may be
  unstable.
- Causal interpretation is limited to the precisely recorded seeded faults and
  deterministic remediation. Natural source-data comparisons are observational.
- A negative or null remediation effect is a valid result and must not be
  removed. A positive effect does not prove real-world reliability, economic
  benefit, security impact, or grid resilience.
- File hashes and consistency checks establish reproducibility of the recorded
  run; they do not validate model behavior beyond the recorded experiment.
- The prototype does not implement streaming ingestion, online learning,
  production monitoring, cybersecurity controls, a service-level objective, or
  human-in-the-loop operational review.
