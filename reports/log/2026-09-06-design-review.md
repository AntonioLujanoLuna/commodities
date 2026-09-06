# Design review: what to fix and what to build next

- Type: `review`
- Date: 2026-09-06
- Git commit reviewed: `913bbcf`
- Data snapshot referenced: `2026-09-05` (artifacts not present in this working tree)
- Scope: full read of `src/enso_commodities/`, `config/`, `tests/`, `README.md`,
  `reports/current_results.md`. No pipeline stage was rerun; every numeric claim
  below is quoted from `reports/current_results.md` or derived from the source.

## Assessment

The methodology is stronger than the result. The evidence ladder, the frozen
registry, the hash-verified immutable snapshots, the shared-draw episode
bootstrap, the recalibration from the centred to the studentized test, the
`test_gates_hold_their_nominal_size_under_the_complete_null` test that measures
the false-positive rate of the study's own gates — this is a study that is
harder on itself than the literature it is answering. The decision to report
Benjamini-Hochberg, Benjamini-Yekutieli and Westfall-Young side by side, and to
keep the resampling-based one because the bootstrap already preserves the
dependence, is the correct call and is correctly implemented.

The honest summary of where it stands: **the study currently has no result, and
knows it.** Three commodities survive the timing/index grid, none survives
phase-specificity, the precious-metal negative controls reject under every
control block tried, and the panel's primary cell does not survive its own grid
FDR. The project has been responding to that by adding controls — macro, then
financial, then a fixed-effects panel. Four blocks in, the negative controls
still reject. That is the signal to stop adding regressors and question the
null distribution instead, which is what most of the findings below are about.

One structural observation. There are sixteen pipeline stages and each defines
its own FDR family. The candidate-family correction is rigorous *within* a
stage and absent *across* them. A reader cannot presently tell how many tests
the program has run in total, which is the exact criticism the README levels at
the literature in its second paragraph.

## Findings, ranked

### F1 — The neutral-date placebo pool is era-selected, and that is the most likely explanation of the negative-control failure

`placebo.py` draws anchors that satisfy `|RONI| < 0.5` *and* sit outside every
real onset's frozen -12/+24 window, and matches the draw on **calendar month
only** (`draw_calendar_matched_anchors`, grouping on `calendar_month`). With 17
episodes and 37-month exclusion windows, those windows cover most of a ~780-month
sample — the README says as much about the market-model baseline. So the
placebo anchors are not scattered across history; they are concentrated in
whatever quiet stretches remain, and those stretches are not a random sample of
macro-financial eras.

Gold, platinum and silver have most of their 65-year variance in two episodes:
the 1970s inflation and the 2001-2011 bull market. If ENSO onsets happen to fall
disproportionately inside those eras and the eligible neutral anchors
disproportionately outside them, the placebo compares event windows drawn from
one financial regime with placebo windows drawn from another — and the negative
controls reject for reasons that have nothing to do with the pipeline being
broken or with ENSO. Every control block added so far (dollar, CPI, activity,
real rate, credit spread, NFCI) attacks this as a *missing regressor* problem.
It is a *null distribution* problem, and no regressor fixes it.

Two cheap tests decide this, in order:

1. Tabulate the era distribution directly: onset months and eligible neutral
   anchor months by decade. If the two distributions differ materially, the
   placebo is not exchangeable with the events and the specificity gate is
   measuring era, not phase. This is a dozen lines against artifacts that
   already exist and should be done before anything else in this list.
2. Add **era-stratified matching** to `draw_calendar_matched_anchors`: match on
   calendar month *and* decade (or a coarse pre/post-1990 split, given how few
   anchors there are), and rerun the specificity gate. If the precious-metal
   rejections weaken under stratified matching, F1 is confirmed and the study's
   central obstacle is a fixable design detail rather than a real failure.

If instead the rejections survive era stratification, that is itself a major
finding and worth its own entry — it would mean the pipeline genuinely
manufactures significance, and the right response is to shrink the claim, not to
keep controlling.

### F2 — The panel's year-block bootstrap uses blocks shorter than the dependence it needs to preserve

`year_block_bootstrap` resamples whole **calendar** years. The object it needs to
keep intact is an ENSO episode and its price response: episodes are constrained
to at least five months (`minimum_duration_months: 5`), typically run 12-18
months, and the panel then reads them at lags of 0 to 12. A run of elevated
regressor is therefore comfortably longer than twelve months, so it necessarily
straddles a January boundary and is split across two independently-drawn blocks.
A block bootstrap whose block is shorter than the dependence length understates
the standard error, and the bias runs toward over-rejection — the direction that
flatters the headline `p=0.013`.

The calendar alignment compounds it: ENSO is phase-locked to the annual cycle
and peaks in NDJ, so a January block boundary cuts through the middle of the
peak of essentially every event, rather than at a random point.

Fix, in increasing order of cost:

- Make the block's start month configurable and report a May-start "ENSO year"
  alongside the calendar year. May-April keeps each NDJ peak whole. This is a
  small change to one derived column plus a config field and its validator.
- Report a 2- and 3-year block sensitivity. If the interval widens materially at
  longer blocks, the one-year figure was never the right one.
- Consider resampling *episodes and the neutral stretches between them* rather
  than fixed-length years, which is the same unit the primary event study
  already resamples and would make the two designs' uncertainty comparable.

Keep `block: calendar_year` as the frozen primary so the published number does
not move, and report the alternatives as a sensitivity. If the interval crosses
zero under an ENSO-aligned block, `current_results.md` needs to say so in the
panel row.

### F3 — There is no minimum detectable effect, so the "NO MATERIAL ENSO EFFECT" bucket is not supported

The evidence ladder's fourth bucket asserts a negative. Nothing in the repository
estimates what effect the design *could* have detected. With 17 episodes and
commodity return volatility, the minimum detectable +12 cumulative abnormal
return at 80% power is plausibly very large — quite possibly larger than any
effect the physical literature would predict. If so, most of the 30 candidates
are not "no material effect", they are "underpowered", and the study is currently
overclaiming in the one direction it believes it is being conservative.

`synthetic.py` and `test_the_studentized_gate_still_detects_a_real_common_effect`
already contain the machinery: the test plants `effect=0.08` and measures the
detection rate. Generalising that into a `power.py` stage that inverts it — per
commodity, using that commodity's own observed episode-level dispersion and
episode count, sweeping effect size until the studentized-plus-BH gate fires 80%
of the time — is a contained module with no new data dependency. It produces one
extra column in the scorecard, and it changes how every null result in the study
should be read.

This is the highest value-per-unit-effort item in this document.

### F4 — Binary warm/not-warm treatment discards the two things most likely to carry the signal

Every stage treats an episode as a binary object derived from a threshold and a
persistence rule. Two well-documented sources of heterogeneity are thrown away:

- **Amplitude.** 1997-98 and 2015-16 are not the same event as 2004-05, and a
  dose-response specification on peak RONI uses information the binary design
  discards. The panel already interacts a continuous index with exposure, so the
  event study is the stage that is out of step here, not the panel.
- **Flavour.** Eastern Pacific and Central Pacific ("Modoki") events have
  materially different teleconnection patterns over Southeast Asia, Australia and
  the Americas — which is to say, over the production regions of most of the
  candidate registry. Pooling them averages two different treatments and is a
  plausible reason the phase-specificity contrast comes out null.

Splitting 17 episodes further is a power problem, which is why F3 comes first: a
flavour split is only worth running once the design can say what it could detect.
A dose-response on amplitude costs no degrees of freedom at all and should be
added regardless, as a reported secondary endpoint rather than a change to the
frozen contract.

### F5 — Program-level multiplicity is unaccounted for

Sixteen stages, each with an internal family, and the reader has no way to count
the total. The remedy that fits this codebase is a **specification curve**: a
stage that enumerates the cross-product of choices already implemented (index ×
anchor × adjustment level × horizon × weighting), reports each commodity's
estimate across the whole family, and tests the *distribution* — median
estimate, share of specifications rejecting — against a joint null rather than
testing each cell separately.

The joint null it should be tested against is worth building on its own terms:
**circularly shift the ENSO index by whole years** and rerun the entire pipeline
on each shift. Whole-year shifts preserve ENSO's phase-locking to the calendar,
preserve the returns' own serial and cross-sectional structure exactly, and ask
the only question that matters — does the *actual* timing of ENSO explain more
than an arbitrary timing of the same-shaped signal. Sixty-odd distinct shifts is
a small reference distribution, but it applies identically to candidates and to
negative controls, which makes it the cleanest available test of whether the
pipeline manufactures significance. It also subsumes the placebo stage's
intent without inheriting F1's era-selection problem.

### F6 — Outcome-blind exposure weights are achievable now; the current framing conflates two different things

`panel.yaml` disclaims its weights as `post_outcome_expert_judgment` and the
validator refuses to let them be labelled outcome-blind. That discipline is
right. But the honesty note reasons from *when* the weights were authored, and
what actually matters is *what they were derived from*. A weight computed by a
fixed, published rule from data that contains no commodity prices is
outcome-blind whatever date it was written — the residual risk is in the choice
of rule, and that risk is bounded by reporting the whole family of rules rather
than one.

Which makes the README's own stated next step both correct and closer to hand
than it sounds:

    exposure[c] = sum over countries p of
        (production share of country p in commodity c)
      x (teleconnection strength of the index at p)

The production side needs no new infrastructure: `palm_oil_data.py` already
parses FAOSTAT with hash-verified provenance, and generalising it to
country-by-commodity production shares is a parser change, not a new subsystem.
The teleconnection side has two routes, and the cheap one should be tried first:

- **Cheap:** use a published, citable teleconnection classification or
  correlation table as a fixed input, hashed like any other source. No gridded
  data, no `[spatial]` extra, no new dependency. It is coarse, and coarse is
  fine — the weights only need to be ordinally sensible.
- **Expensive:** compute correlations between the index and gridded precipitation
  (CHIRPS/ERA5) over each production region. Higher fidelity, an order of
  magnitude more infrastructure, and a fresh set of researcher degrees of
  freedom in the region definitions.

Either way: freeze as `exposure_provenance.version: 2` with
`method: external_production_and_teleconnection`, `outcome_blind: true`,
`inference_scope: confirmatory`, keep version 1 in the file for comparison, and
rerun the panel. The existing mapping-permutation test then becomes a check on
version 2 rather than an apology for version 1, and the panel graduates from
exploratory to a design whose result can be reported as a result.

### F7 — Results are duplicated between `README.md` and `reports/current_results.md`, and hand-transcribed from receipts

The README carries roughly 60 lines of prose numbers — "10 of 30 candidates
reject", "17 macro-covered episodes", "0.00624 per unit of exposure",
"98.3rd percentile" — that also appear in `reports/current_results.md`. Two
copies of every number, both typed by hand from run artifacts, is a drift
guarantee. It has not drifted yet only because the project is young.

Every stage already writes a hash-linked JSON receipt. The fix is to generate the
numbers rather than type them: a `scripts/build_report.py` that reads the
receipts under `tables/<snapshot>/` and emits the evidence-summary table and the
run-identity block into `reports/current_results.md`, leaving the interpretation
prose hand-written. The README then keeps the pitch, the ladder, the quick start
and the layout, and points at `reports/` for every number. `reports/results.md`
is already gitignored as a generated path, so the convention exists — it just
has nothing writing to it.

The same generator should fail loudly when a receipt's input hashes do not match
the snapshot it is asked to describe. That converts "the report is stale" from
something a reader might notice into something the build refuses to do.

### F8 — Engineering gaps, all minor

- **No determinism test.** The seeds and salts are carefully designed; nothing
  asserts that the same seed and inputs produce byte-identical outputs. One test
  that runs a small stage twice and compares receipt hashes protects the entire
  reproducibility claim.
- **No end-to-end synthetic pipeline test.** `synthetic_market` feeds
  `test_pipeline_recovers_a_planted_effect`, but no test walks a synthetic
  snapshot through download-shaped inputs to a final receipt. Stage-boundary
  regressions are the failure mode this catches.
- **Analysis stages are untested.** `tests/` covers the computational modules;
  the eight `*_analysis.py` orchestration modules have no direct tests, and they
  are where the hash-linking and receipt-writing logic lives.
- **Coverage is not enforced.** `pytest-cov` is a declared dev dependency and CI
  does not invoke it.
- **`disallow_untyped_defs = false`.** The codebase is almost fully annotated
  already; turning this on costs little and locks the state in.
- **CI runs no stage.** A synthetic-fixture smoke run of one analysis stage would
  catch orchestration breakage that unit tests cannot.

## Where findings go

Established in this commit: `reports/log/`, append-only, one dated file per
entry, conventions in `reports/log/README.md`. The split is

- `README.md` — pitch, ladder, how to run, layout. No numbers.
- `reports/current_results.md` — the latest run only. Overwritten. Numbers
  generated from receipts (F7), interpretation hand-written.
- `reports/log/` — dated, immutable. Decisions, findings, negative results, and
  the reasoning behind every freeze.
- `config/` — the frozen contracts themselves, because a machine enforces them.

The reason to spend a commit on this: the study's strongest claim is that it did
not go looking for its answers. `config/` proves what was frozen; only a dated
log proves what was *known* when it was frozen. `panel.yaml`'s
`outcome_blind: false` disclosure is exactly the kind of statement that belongs
in a record which cannot be quietly revised later.

## Suggested order

1. F1 step 1 — tabulate onset vs. neutral-anchor eras. One afternoon, and it may
   reframe everything else in this list.
2. F3 — minimum detectable effect. Contained, no new data, changes how every
   null in the study reads.
3. F2 — ENSO-aligned block sensitivity for the panel. Small, and the current
   interval is optimistic until it is done.
4. F7 — generate the report numbers from receipts before the two copies diverge.
5. F1 step 2 — era-stratified placebo matching, if step 1 justifies it.
6. F6 — outcome-blind exposure weights v2, cheap route first.
7. F5 — whole-year circular-shift null, then the specification curve on top of it.
8. F4 — amplitude dose-response now; flavour split once F3 says it is affordable.
9. F8 — alongside whatever else is being touched.

Items 1, 2 and 3 are all things that could change the interpretation of results
already published in `current_results.md`. Items 6 and 7 are the ones that could
turn the project's central negative into a defensible finding either way.
