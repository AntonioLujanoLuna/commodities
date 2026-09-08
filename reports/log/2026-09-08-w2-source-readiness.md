# W2 physical-chain source audit: blocked without estimating a partial chain

- Type: `finding`, `data-audit`
- Date: 2026-09-08
- Workstream: W2 cold-phase disruption chain
- Status: `blocked_source_contract`

The frozen rainfall-throughput-price estimator remains unrun. The public-source audit found no
combination that currently satisfies all three registered links, and the implementation now makes
that state executable with `make disruption-readiness`.

## What was verified

- Queensland Government SILO provides reproducible monthly gridded rainfall files from 1889, but
  the outcome-independent Hunter coal-chain extraction region has not been frozen.
- The Transport for NSW Port of Newcastle workbook is machine-readable and currently contains 103
  monthly rows from January 2018 through July 2026. Its downloaded SHA-256 is
  `1c7178923a83a37a465d6c82820795642dce118f04af864fae3c979544994158`. That is nine
  calendar-year clusters, below the registered minimum of ten, and the workbook does not preserve
  an observation-level historical publication date.
- North Queensland Bulk Ports publishes monthly Hay Point terminal throughput from financial year
  2020-21 onward: seven calendar-year clusters in the live table. The audit runtime also received a
  Cloudflare challenge rather than a reproducible source response.
- Gladstone Ports exposes month-keyed cargo pages over a long enough apparent span, but the runtime
  endpoint timed out and the pages inspected do not expose observation-level publication vintages.
- The World Bank Australian coal price series is already reproducibly acquired. That does not fix
  the missing dated throughput treatment.

## Why no coefficient was produced

Changing clusters from years to quarters would make the short port panels pass by relabelling the
same serially dependent history. Assigning a conventional month-end release lag would invent the
information date required by the third link. Both would weaken a frozen design after inspecting
the available data. The gate therefore reports all three links as not ready and preserves the
program rule that a partial chain is incomplete, never partial support.

## Reproducible command

```text
uv run python scripts/audit_disruption_sources.py
```

The candidate URLs, observed coverage and readiness flags are frozen in
`config/cold_phase_mechanism_sources.yaml`.
