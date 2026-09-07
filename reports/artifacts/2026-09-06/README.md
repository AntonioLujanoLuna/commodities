# Auditable results bundle: 2026-09-06

This directory is generated from the locally verified stage receipts for snapshot `2026-09-06`.
It intentionally excludes large bootstrap replicates and raw/licensed data. `manifest.json` binds
each published file to its SHA-256 digest and records the full analysis source-tree fingerprint.

- `scorecard.csv` and `scorecard.md` combine existing gates; they define no new test.
- `figures/` visualizes primary estimates, gate passage, and the panel specification family.
- Stage summary JSON and selected result CSV files are byte-for-byte copies of the run artifacts.

Run `enso-publication --check --bundle reports/artifacts/2026-09-06` to validate this bundle.
