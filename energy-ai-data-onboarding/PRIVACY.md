# Privacy and statelessness

This package is a stateless batch data pipeline. It processes public
building-energy measurements and the minimum technical metadata needed for
provenance, quality auditing, reproducibility, and downstream forecasting.

It has no chat or LLM integration and does not retain conversation history,
prompts, assistant messages, user profiles, embeddings, vector indexes,
browsing history, analytics, telemetry, personal identifiers, or personal
information. It does not transmit local outputs beyond the selected output
directory. A run writes only the explicitly documented dataset, issue,
remediation, quarantine,
provenance, configuration, report, and hash artifacts into the selected local
output directory.

The software has no database or persistent background process. Deleting a run
directory removes that run's generated artifacts; the source data remain under
the user's control.
