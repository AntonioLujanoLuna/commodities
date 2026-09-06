# External exposure v2 and the panel-family timing null

- Type: `decision`, `finding`, `negative`
- Date: 2026-09-06
- Base commit: `c251dc1`
- Data snapshot: `2026-09-06`
- Status: implemented, run on real data, and receipt-verified

## Decision

The four-level v1 exposure scale is retired from the active panel contract. It was physically
motivated but authored after this repository's event-study outcomes were known. Version 2 instead
maps supported commodities to SPAM 2020 physical crop-area rasters and weights the positive FAO
ASIS historical El Niño drought-hotspot signal by all mapped crop area. The mapping, transformation,
source URLs and explicit exclusions are frozen in `config/exposure_v2.yaml`.

The computation is outcome-independent, but the mapping was authored after earlier event-study and
v1 panel results were visible. Its scope is therefore `retrospective_external_validation`, not
confirmatory. External data solve the mechanical outcome-contamination problem; they do not create
a preregistration after the fact.

An unsupported candidate is excluded, not assigned zero. Zero has a substantive meaning in this
design: measured absence of hotspot burden. Eleven candidates lack a defensible crop raster and are
therefore outside the v2 panel; twenty-one crop candidates and the three frozen negative controls
remain. The smallest included hotspot crop-area share is 0.2110.

## Finding

The v2 retrospective-validation primary RONI lag-six exposure estimate is 0.042449, with a calendar-year block interval of
[-0.022915, 0.109700] and studentized p=0.20840. Its control interaction has p=0.48776. The observed
mapping is at the 73.7th percentile of shuffled assignments and has two-sided p=0.52674. No exposure
cell survives BH correction across the 20-cell grid, and no control cell rejects at raw 5%.

This is a useful negative result. Replacing the post-outcome ordinal weights with external physical
weights removes the earlier nominal primary-panel rejection. The earlier result should not be read
as evidence that v2 failed to reproduce; v2 asks the more defensible question and returns a null.

## Joint timing null

`config/specification_curve.yaml` freezes 60 circular whole-year shifts. Each alignment refits all
20 panel cells, for 1,220 fits including the observed alignment. The primary curve statistic is the
median absolute naive t across the family; maximum absolute t and positive-estimate share are
secondary summaries. Finite-sample p-values include the observed alignment.

The observed median absolute naive t is 1.0010 and its joint timing p-value is 0.40984. The observed
maximum absolute naive t is 1.8387 with p=0.73770; the positive-estimate share is 0.75 with p=0.21311.
The actual ENSO alignment is not unusually strong relative to arbitrary whole-year alignments.

This null covers the panel's 20-cell specification family only. It is not a program-wide null over
the event study, placebo, mechanism and other stages. Naive t is used for the cross-alignment curve
so the identical statistic can be computed tractably for every fit; the primary panel receipt still
uses the full block-bootstrap inference.

## Integrity change

The panel narrative in `reports/current_results.md` is now generated from the panel and timing-null
receipts. The run context also fingerprints the analysis source tree, so `report-check` detects code
or configuration changes even when the data receipts themselves have not moved.
