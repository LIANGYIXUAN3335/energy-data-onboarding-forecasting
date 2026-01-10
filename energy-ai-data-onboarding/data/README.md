# Data directory

No large dataset is committed here.

- Put immutable downloads under `data/raw/` and verify them against
  `provenance/bdg2_v1.0.json`.
- Local pipeline runs go under `results/<run-name>/`; the committed, hash-verified
  final bundle is `results/bdg2_mvp_v2_hardened/`.
- Automated tests generate deterministic fixtures in temporary directories
  from `tests/conftest.py`; no fixture dataset is committed.

The `.gitignore` excludes raw data and generated result directories.
