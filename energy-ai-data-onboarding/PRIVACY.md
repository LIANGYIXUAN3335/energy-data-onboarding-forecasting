# Privacy and data policy

This package is a stateless batch data pipeline. It processes public
building-energy measurements, public site weather and the minimum technical
metadata needed for provenance, quality auditing, reproducibility and
downstream forecasting.

It stores no personal data and does not transmit local outputs beyond the
selected output directory. A run writes only the documented dataset, issue,
remediation, quarantine, calibration, provenance, configuration, report and
hash artifacts into that directory. There is no database or persistent
background process; deleting a run directory removes that run's artifacts, and
the source data remain under the user's control.
