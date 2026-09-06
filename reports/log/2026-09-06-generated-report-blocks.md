# Implemented: the current-results numbers are generated, not typed

- Type: `decision`
- Date: 2026-09-06
- Follows: [2026-09-06 design review](2026-09-06-design-review.md), item F7
- Status: implemented and unit-tested; the generated blocks in `current_results.md` are still
  placeholders, because no snapshot exists in the environment this was written in

## The problem

Roughly sixty lines of prose numbers appeared twice, in `README.md` and in
`reports/current_results.md`, both typed by hand from run artifacts. Two hand-maintained copies of
every number is a drift guarantee. It had not drifted yet only because the project is young, and
the cost of the first drift is high: the study's whole claim is that its artifacts are auditable.

## What was done

`reporting.py`, `make report` and `make report-check`. The generator reads the stage receipts
under `tables/<snapshot>/` and renders three blocks of `current_results.md` between
`<!-- generated:... -->` markers: the run identity, the evidence summary, and a table of receipt
digests. Prose outside the markers is never touched.

Two properties are what make it worth having rather than merely convenient.

*Stale is refused.* Every output hash each receipt recorded is re-verified against the file on
disk before anything is quoted. A receipt describing artifacts that have since changed cannot be
turned into a report at all. The generator also collects every configuration digest the receipts
recorded and raises if two stages disagree about one, because that means they were not run
against the same configuration and their numbers do not belong in one table.

*Drift is detectable.* The render is a pure function of the receipts -- no timestamps, no
environment capture -- so `--check` re-renders and compares. A stale report is a build failure
rather than something a reader might notice.

A receipt missing a declared metric raises rather than rendering a gap. That is deliberate: if a
stage stops emitting a diagnostic the report quotes, the report should break, not quietly shrink.
The cost is that changing a stage's summary keys now also means updating `STAGES` in
`reporting.py`, which is the coupling that keeps the two honest.

The README's numeric prose was replaced with a `What each stage found` section that describes what
each stage does and what it concluded, without counts, and points at `current_results.md` for
every number. Every methodological sentence was kept; only the figures went. The panel section
keeps its full design argument and loses its three quoted estimates.

## What this does not do

- The generated blocks currently read `Not yet generated`. Filling them requires a completed
  snapshot, which this environment does not have. The hand-recorded run identity -- git commit,
  runtime versions, regeneration date -- stays outside the markers, labelled as hand-recorded,
  because no receipt carries it yet. Putting the interpreter and library versions into the stage
  receipts would let that move inside too, and is worth doing.
- `report-check` is not in CI, because CI has no artifacts to check against. It belongs in
  whatever runs `make real-data`.
- The interpretation prose in `current_results.md` is still hand-written and still has to be
  updated by a person after each run. That is correct -- it is judgement, not arithmetic -- but it
  means the note can still be misleading while every number in it is right.

## Next

F6, outcome-blind exposure weights from production geography and a published teleconnection
table, cheap route first. Then F5, the whole-year circular-shift null and the specification curve
built on it. F4 last, since the flavour split is only affordable once the minimum-detectable-effect
stage says what the design can see.
