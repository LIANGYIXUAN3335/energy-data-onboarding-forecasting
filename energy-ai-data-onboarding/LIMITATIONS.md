# Limitations

- This is a local batch research prototype. It is not a real-time, distributed,
  production, security-certified, or utility-operated system.
- The committed configuration selects twelve buildings across three BDG2 sites
  and two primary-use categories. It is a reproducible subset, not a claim of
  population representativeness.
- BDG2 raw meter timestamps are local wall-clock timestamps. The selected sites
  share the `US/Eastern` source timezone. The current CSV contract represents
  those labels with a UTC marker for stable parsing without an offset
  conversion; this is not cross-timezone UTC harmonization. DST interpretation
  remains a limitation.
- Meter values remain in the physical unit supplied by BDG2. No silent unit
  conversion is performed.
- Detector thresholds and the two past-only repair rules are transparent
  baselines. They can produce false positives, false negatives, or adverse
  downstream results; the v2 run is a documented example, and the v3
  calibration is a documented response to it. All configured outcomes must be
  reported.
- Per-building calibration cannot make an 8-hour constant reading detectable
  on a meter that repeats its value for most of the day; one seeded stuck
  event on such a meter is missed in v3 and reported as such.
- Seeded faults validate software behavior; they do not estimate fault rates in
  operational energy systems.
- Short forward fill uses only earlier values. Ambiguous scale shifts and
  unresolved defects are kept as null-valued keys and recorded in quarantine.
- Fixture runs validate code only and must not be cited as BDG2 findings.
