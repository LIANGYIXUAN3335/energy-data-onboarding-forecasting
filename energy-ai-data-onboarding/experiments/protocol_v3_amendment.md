# Protocol amendment v3 (onboarding), dated 2026-09-28

This amendment is a separately labeled experiment. The pre-registered v2
protocol and its committed bundle (`results/bdg2_mvp_v2_hardened`) are kept
unchanged as the baseline; nothing below rewrites them.

## What the v2 run showed

The v2 remediated condition forecast worse than the corrupted condition
(pooled MASE 0.922 versus 0.740 for gradient boosting). The cause was traced
to the run detectors, not to the seeded faults: 22,014 of the 22,033
quarantined rows were natural data. One coarsely quantized meter
(Eagle_education_April, 72 distinct readings in 17,544 hours) lost 97.9% of
its training year to the stuck-sensor rule, and buildings that are idle at
night or at weekends lost their zero hours to the long-zero rule. The
forward fill also copied a run's own value back into that run because only
the tail of a run was flagged.

## Changes (all recorded in `dataset_manifest.json` and `detector_calibration.csv`)

1. **Whole-run flagging.** A run that reaches its threshold is flagged in
   full, including its first reading, so a repair can only draw on values
   before the run.
2. **Per-building stuck threshold.** The configured threshold (8 h) is raised
   to just above the 0.999 quantile of the building's natural constant-run
   lengths, computed on the reference condition strictly before the fault
   cutoff. The same frozen thresholds are then applied to every condition, so
   seeded faults never influence their own detection.
3. **Profile-aware zero runs.** A zero run of at least 6 h is flagged only
   during hours when the building is normally active: the causal hour-of-week
   median over the previous 8 weeks must exceed 25% of the trailing 168-hour
   median. Where no profile exists yet the plain rule applies.
4. **Past-only profile repair.** After the 3-hour forward fill, gaps of up to
   48 h are filled with the median of the same weekday and hour over the
   previous 8 weeks, using only earlier, originally valid readings. Longer
   gaps stay quarantined. Suspected unit or level shifts receive the profile
   value instead of being divided by an inferred factor.
5. **Four seeded events per fault family** instead of one, and block faults
   (zero block, stuck segment, unit scale) are only written over strictly
   positive readings so that every seeded fault is observable.
6. **Site weather** (`weather.csv.gz`, pinned in `provenance/bdg2_v1.0.json`)
   is published as a condition-independent artifact for the consumer.

## What did not change

Source pinning and hash verification, the fault cutoff (2017-01-01), the
twelve-building selection, the output contract of the three condition files,
the fail-closed gates, and the rule that no remediation reads a later value.

## Known trade-off

A meter whose readings repeat for most of the day (repeat rate 0.75 for
Eagle_education_April) cannot distinguish an 8-hour stuck segment from
normal behaviour; one seeded stuck event on that meter is therefore missed
(stuck-segment recall 0.75 in `detector_metrics.csv`). This is reported, not
tuned away.
