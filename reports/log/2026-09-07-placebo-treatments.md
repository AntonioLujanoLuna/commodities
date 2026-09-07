# 2026-09-07 — Placebo treatments: the negative-control gate is miscalibrated, not informative

- Type: `finding`, `extension`, `negative`
- Data snapshot: `2026-09-06`
- Climate-index snapshot: `2026-09-07`
- Stage receipt: `tables/2026-09-06/surrogate_treatment_run_summary.json`
- Contract: `config/surrogate_treatment.yaml`
- Status: implemented and run on the real snapshot; exploratory, changes no frozen gate

## Why this was run

Every control block added since the primary event study narrowed the precious-metal problem
without removing it, and the design review recorded that pattern as looking more like a
specification error than a missing regressor. The project had no way to distinguish the two,
because its only specificity test moves the dates and keeps the treatment: the neutral-date
placebo compares ENSO onsets with quiet stretches of history that sit in a different
macro-financial era, which the era-balance diagnostic already flagged as a candidate explanation
for the precious-metal failure that no additional regressor can remove.

The complementary experiment keeps the dates and replaces the treatment. If Gold rejects under
RONI and rejects just as often under a series with RONI's spectrum and no relationship to
history, the negative-control gate is measuring the machinery, not ENSO specificity.

## What was built

`config/surrogate_treatment.yaml` freezes an exploratory contract that refuses to load if it
claims to change a gate. The stage re-runs the frozen pipeline end to end — episode construction,
seasonal and common-market adjustment, the external-macro model, the +12 endpoint, the
studentized whole-episode bootstrap, candidate-family Benjamini-Hochberg and the
calendar-matched neutral-date placebo — with the ENSO index swapped for a substitute series.

Two substitute families: four NOAA PSL climate indices (DMI, PDO, NAO, AMO), and 400
phase-randomized surrogates of RONI itself in two variants, one preserving only the power
spectrum and one preserving the calendar-month climatology as well. Every substitute is
moment-matched to RONI over the shared analysis window before the frozen 0.5 threshold is
applied, so an episode is the same number of standard deviations for every treatment.

Two things make the run auditable rather than merely fast. The stage runs a reference cell with
RONI itself through the substitute-treatment code path and aborts unless it reproduces the frozen
macro-adjusted estimates; it reproduced them to 1.1e-16, and its gate counts match the macro
stage exactly (9 candidates passing, 3 controls rejecting bootstrap, 3 rejecting placebo). And
because the full -12/+24 event path costs about twenty seconds per treatment, the grid uses a
vectorized endpoint builder, pinned by test against the frozen construction to machine precision.

## Findings

**The candidate result is treatment-specific.** RONI produces 9 candidates passing every gate.
Across 400 surrogates the mean is 0.32 and the maximum is 8; none reached 9, giving a
finite-sample p of 0.0025. Per commodity, each of the nine passes under between 0.25% and 2% of
surrogate treatments. Whatever the nine associations are, they are not what this machinery
produces from an arbitrary persistent series.

**No alternative climate index reproduces them.** Not one of DMI, PDO, NAO or AMO produces a
single candidate passing every gate, and at most one control rejection.

**The negative-control placebo failure is exactly what the machinery produces on its own.** Gold,
Silver and Platinum reject the neutral-date placebo at raw 5% under the real index. They also
reject it under 42.5%, 40.8% and 35.0% of surrogate treatments respectively. The observed count
of three control placebo rejections has a finite-sample p of 0.115 against the surrogate null: it
is unremarkable. Their bootstrap rejections are milder but the same story — about 12% under
surrogates, p=0.127 per commodity, p=0.017 for the joint count of three.

That is a calibration statement about a test, not about gold. A placebo whose nominal size is 5%
and which fires on roughly 40% of null treatments is not evidence of anything when it fires on
the real one. The prespecified reading of the failing negative-control gate — that the study's
associations cannot be ENSO-specific because the controls fail too — does not survive this
measurement.

## What this does and does not change

It changes no gate, and it is explicitly not a promotion. The stage was authored after the
discovery results were visible, so it is a diagnostic on an existing contract rather than a
preregistered test, and it is recorded as exploratory in its own receipt.

What it does establish is negative and specific: the negative-control specificity failure, which
has been the stated reason no survivor could be called an ENSO mechanism, is not informative as
currently measured. Two consequences follow for how the results are described. The failing
control gate should no longer be cited as evidence against ENSO specificity without this
calibration alongside it. And the neutral-date placebo needs either a size correction or an
era-matched anchor pool before it can carry the weight the evidence ladder puts on it.

The remaining barriers are untouched: the physical chain still fails at the supply-to-price link,
the exposure panel still finds nothing surviving FDR, the phase contrast is still largely
non-specific, and the forecast benchmark is still pseudo-out-of-sample. This entry removes one
argument against the survivors; it supplies no argument for them.

## Open items this creates

- The neutral-date placebo's finite-sample size should be reported next to every placebo p-value
  it produces, in the same way the studentized bootstrap's size was measured before it replaced
  the centred test.
- An era-matched or era-reweighted anchor pool is the obvious repair, and would be a change to a
  frozen specification, so it needs its own decision entry and a stated reason before any number
  from it is read.
- Surrogate treatments give a cheap calibration harness for any future gate. The panel stage and
  the forecast benchmark could both be run under the same substitution.
