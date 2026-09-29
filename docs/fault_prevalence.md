# How common are meter-data defects? Literature behind the fault-injection density

Compiled 2026-09-28 for the `bdg2_v3_dense` configuration. The v3 canonical
suite seeds four events per fault family (108 of 105,408 training rows,
0.10%). Published field studies report defect rates one to two orders of
magnitude higher, concentrated in long runs and on a minority of meters. The
dense configuration is calibrated to the lower end of those reports so that the
experiment is representative without being an outage simulation.

Rates below are quoted as printed in the source (share of readings, of meters,
or of days as stated). "Read" means the number was confirmed on the fetched
document; "snippet" means it comes from a search-engine excerpt only.

## Missing readings and gaps

| Source | Population | Rate | Status |
|---|---|---|---|
| Weber et al. 2021, *Data-driven copy-paste imputation*, arXiv 2101.01423 | "implemented smart meter systems" (industry figure) | 3–4% of recorded values missing | read |
| Park et al. 2023, arXiv 2305.09474 | Irish CER trial; Korean residential smart meters | 0.03% (Irish), 3.5% (Korean) missing | read |
| Low Carbon London user guide (UK Data Service SN 7857), 2015 | 5,198 London households, half-hourly, 2013 | 0.17% missing on average, never above 3.4% in any period | read |
| SERL smart meter data quality report, ed. 07 (UCL, 2024) | 13,209 GB households, ~1.0 billion half-hourly electricity reads | 17.2% of half-hourly reads missing (includes communication failures) | read |
| Fowler et al. 2015, PNNL-24331 | >1,000 U.S. Army building meters | Missing timestamps are the most frequent suspect-data type; example building 6.4% missing; per-building 2.6–30.5%, most of it in gaps longer than one week | read |
| Garrison and New 2021, *Smart Cities* 4(1) (ORNL/UTK) | 178,333 premises, 15-min, one year | >93% of meters miss <2% of intervals; almost all miss at least one; 0.35% miss ≥90% | read |
| Stojcheska et al. 2026, arXiv 2608.21638 | 17,428 commercial meters, hourly, 25 months | Every meter has gaps; 58.7% of meters' longest gap ≤6 h, 34.0% up to 24 h, 7.4% up to one week | read |
| Quesada et al. 2024, *Scientific Data* (GoiEner, Spain) | 25,559 hourly supply-point series | 95.3% of series <1% missing; 0.43% of series >50% missing | read |
| Li et al. 2024, *Scientific Data* (HKUST campus) | 1,432 campus sub-meters | 14–30% missing at hourly resolution (sub-metering, not utility AMI) | read |

## Zero and flat readings

| Source | Population | Rate | Status |
|---|---|---|---|
| Fowler et al. 2015, PNNL-24331 | example Army building, 133,587 timestamps | 6.2% zero-consumption readings (flagged as suspect, not necessarily wrong) | read |
| Shishido 2012, ACEEE Summer Study (EnerNOC) | U.S. residential 15-min AMI, customer-day records | 1.04% of customer-days contain at least one zero interval; 40.65% on one storm day | read |
| SERL report ed. 07 (2024) | daily electricity readings | ~2% of daily reads are suspicious zeros, mostly meters recording zero during British Summer Time | read |
| Garrison and New 2021 | 178,333 premises | 2.1% of meters read zero for the entire year | read |
| Li et al. 2024 (HKUST) | 1,394 meters | 3.3% of meters returned only zeros | read |
| Public EDA of the ASHRAE GEPIII training set (BDG2 subset) | 2,380 meters, hourly 2016 | ~9.3% of readings zero; one site 33% (a contiguous multi-month block) | snippet |

No source gives a per-reading prevalence for stuck (frozen, non-zero) meters;
the evidence is meter-level (all-zero years above) and qualitative (GEPIII
teams "dropped long streaks of constant values").

## Spikes and outliers

| Source | Population | Rate | Status |
|---|---|---|---|
| Garrison and New 2021 | 178,333 premises | 0.08% of meters contain an extreme spike (>10,000 kWh in 15 min) | read |
| Gulati and Arjunan 2022, LEAD1.0, arXiv 2203.17256 | 1,413 GEPIII/BDG2 meters, hourly 2016 | 1.66% of non-missing readings manually annotated as anomalous (all anomaly types) | read |
| SERL report ed. 07 (2024) | half-hourly gas and electricity | 0.4% of gas reads are register-overflow "max reads"; electricity ~0.0002% | read |
| Shishido 2012 | residential AMI | 5 extreme spike records among 27 flagged (mean 4,774 kWh vs 6 kWh typical) | read |

## Negative values, unit and scale errors, timestamps

| Source | Population | Rate | Status |
|---|---|---|---|
| Fowler et al. 2015, PNNL-24331 | example Army building | 0.1% negative consumption values | read |
| SERL report ed. 07 (2024) | half-hourly and daily reads | 0 negative readings; ~9% of daily electricity reads suspected recorded in kWh instead of Wh (7.2% flagged) | read |
| Shishido 2012 | residential AMI | 1.05% of records are non-identical duplicates, often differing by an integer meter multiplier | read |
| Quesada et al. 2024 | 25,559 series | ~5% of duplicate-timestamp cases carry conflicting values | read |
| Schaffer et al. 2022, *Scientific Data* (Danish district heating) | 3,021 heat meters, hourly, 3 years | 0.02% of rows removed for a daylight-saving storage defect; 0.3% missing after screening | snippet |
| Miller et al. 2020, BDG2 (*Scientific Data*) | BDG2 itself | "several" meters were in the wrong unit; zero runs >24 h and meters with >100 consecutive missing days were dropped in cleaning | read |

## What the dense configuration seeds

`configs/bdg2_v3_dense.json` keeps every v3 policy and changes only the event
counts per fault family, all still confined to the 2016 training period and
written only over strictly positive readings (block faults) or eligible rows:

| Family | Events | Rows | Share of 105,408 training rows | Literature anchor |
|---|---|---|---|---|
| missing_value (single hours) | 600 | 600 | 0.57% | 0.17–3.5% missing in utility AMI studies; long gaps are left to the natural data |
| long_zero_block (8 h) | 60 | 480 | 0.46% | 1–6% zeros; the natural BDG2 data already carry long zero stretches |
| stuck_segment (8 h) | 25 | 200 | 0.19% | no per-reading figure published; kept small |
| positive_spike (single hours) | 40 | 40 | 0.04% | 0.08% of meters with spikes; 0.0002–0.4% overflow reads |
| negative_value (single hours) | 100 | 100 | 0.09% | 0.1% negative readings |
| unit_scale_segment (8 h, ×100) | 30 | 240 | 0.23% | 1–9% of reads affected by unit or multiplier errors on a subset of meters |
| **Total** | **855** | **1,660** | **1.6%** | within the 1–5% band that the studies above support |

The detector-validation suite receives the same counts, so precision and
recall are measured at the same density. The dense run is a separately labeled
experiment; the v3 canonical suite (four events per family) remains the
reference for the headline result.
