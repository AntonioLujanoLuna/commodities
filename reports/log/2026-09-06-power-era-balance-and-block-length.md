# Implemented: minimum detectable effect, placebo era balance, panel block length

- Type: `decision`
- Date: 2026-09-06
- Follows: [2026-09-06 design review](2026-09-06-design-review.md), items F3, F1 and F2
- Status: implemented and unit-tested; **not yet run against a real-data snapshot**

## What was added

Three diagnostics, in the order the design review ranked them. None of them changes a gate,
a frozen contract, or any number in `current_results.md`. Each exists to make an existing
result readable rather than to produce a new one.

### Minimum detectable effect (`make power`, F3)

`power.py` and `power_analysis.py`. For each commodity: the smallest additive shift on the
frozen endpoint that the two-sided studentized bootstrap would reject with probability
`target_power`.

No new resampling happens. The alternative is an additive shift, which moves the sample mean
by exactly its own size and leaves the sample variance untouched, so the observed statistic
under the alternative is the stored null statistic plus `effect / standard_error`. The
inference stage's stored replicates are therefore already the correct reference distribution,
and the stage is a rearrangement of a bootstrap that has already run. `tests/test_power.py`
checks that claim the hard way: it plants the reported effect in fresh samples and counts how
often the real bootstrap rejects, over several reference draws to average out the noise in a
single standard error. It lands at 0.77 against a target of 0.80.

Two thresholds are reported. The marginal one uses `fdr_alpha`. The family bound uses
`fdr_alpha / family_size`, the level Benjamini-Hochberg demands when exactly one member of the
family rejects; what the FDR gate actually applies sits between them. Negative controls sit
outside the family and carry no penalty.

The column that matters for the evidence ladder is `observed_effect_below_marginal_mde`. A
commodity it flags is uninformative, not null, and does not belong in the NO MATERIAL ENSO
EFFECT bucket. How many of the thirty candidates it flags is the open question, and the
answer will not be flattering: at seventeen episodes the minimum detectable effect is likely
to be a large cumulative abnormal return.

One caveat, stated in the module and worth repeating. The estimate is a function of one
observed standard error, and a standard error from seventeen episodes carries roughly a fifth
of its own size in sampling noise; the test's own reference draws moved the reported figure by
a factor of two across identical processes. Read it as an order of magnitude. It is precise
enough to separate "no effect" from "no power to see one" and nowhere near precise enough to
become a gate.

### Placebo era balance (`exchangeability.py`, F1 step 1)

The neutral-date placebo matches anchors on calendar month alone. Anchors must also be neutral
and sit outside every event window, and with seventeen episodes those windows cover most of
the sample, so the eligible pool is whatever quiet stretches survive both filters. Nothing
guarantees those stretches are spread across history the way the onsets are.

The diagnostic tests the real onsets' `mean_year`, `median_year` and `year_dispersion` against
the placebo's own calendar-month-matched draw distribution, and breaks the comparison down by
decade in `placebo_era_composition.csv`. It runs inside the placebo stage and writes to the
same receipt.

What it decides: whether the precious-metal failure is a null-distribution problem rather than
a missing-regressor problem. Gold, platinum and silver carry most of their sixty-five-year
variance in the 1970s inflation and the 2001-2011 bull market. If onsets and eligible anchors
sit in systematically different eras, the specificity gate is partly comparing macro-financial
regimes, and four blocks of added controls were attacking the wrong thing. If the eras balance,
that hypothesis is dead and the failure is real -- which is also worth knowing, and points at
shrinking the claim rather than controlling harder.

### Panel resampling-block sensitivity (F2)

`assign_resampling_blocks` cuts blocks on an absolute month index, so a start month and a block
length define the unit. `config/panel.yaml` keeps `calendar_year` frozen as the primary and adds
May-aligned one-, two- and three-year variants; `panel_block_sensitivity.csv` reruns the primary
cell's bootstrap under each, changing nothing but the resampling.

The argument for doing this: episodes run at least five months and the panel reads them at lags
of up to twelve, so a run of elevated regressor exceeds twelve months and necessarily straddles
a January boundary, and ENSO's phase-locking puts its NDJ peak on that boundary rather than at
a random point. A block shorter than the dependence understates the standard error, so the
frozen `p=0.013` is the optimistic end. `tests/test_panel.py` demonstrates the mechanism on a
panel whose residual dependence runs five years: three-year blocks return a standard error more
than a quarter larger than calendar years, and moving the boundary without lengthening the block
does not help.

If the interval crosses zero under a longer block, the panel row in `current_results.md` has to
say so.

## What this does not do

- No gate, threshold or frozen contract moved. `commodities.yaml` is untouched, and
  `panel.yaml` keeps `block: calendar_year` and its `exploratory_only` scope.
- None of the three has been run on real data. The container this was written in has no
  snapshot, and rebuilding one would change the snapshot date and break the hash chain that
  `current_results.md` reports against. Running `make placebo`, `make power` and `make panel`
  against the existing `2026-09-05` snapshot is the next step, and all three read hashes that
  the existing artifacts already satisfy.
- F1 step 2 -- era-stratified anchor matching -- is deliberately not implemented. It changes
  the specificity gate, and that is a decision to make after seeing step 1's answer, not before.

## Next

In order: run the three stages against the `2026-09-05` snapshot and record the numbers here as
a `finding` entry; then F7, generating the report tables from receipts before the hand-typed
copies in `README.md` and `current_results.md` diverge; then F1 step 2 if the era check
justifies it.
