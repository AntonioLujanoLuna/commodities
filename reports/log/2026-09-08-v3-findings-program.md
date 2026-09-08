# v3 findings program: five designs that can produce a result at this sample size

- Type: `decision`, `review`
- Date: 2026-09-08
- Git commit at authoring: `9e1004d`
- Discovery snapshot referenced: `2026-09-06`
- Status: contract frozen here; implementation follows in this entry's commits. **No stage
  below has been run against a real-data snapshot.**

## Why a new program rather than another control block

The design review's structural criticism was that the project answers every failure by adding a
regressor. The placebo-treatment stage settled that argument in the other direction: the gate
that kept failing was miscalibrated, not under-controlled. What remains after that correction is
not a control problem at all. It is a **power problem with a specific shape**, and it is worth
stating precisely because all five workstreams below are responses to it.

The frozen primary design reduces each commodity to *one number per episode* — the cumulative
abnormal return at +12 months — and there are seventeen warm episodes in sixty-five years. The
estimand is a mean over seventeen draws. That is the entire sample, no matter how many decades
of monthly prices sit underneath it, and it is why 22 of 30 candidates fall below their own
minimum detectable effect. No additional price history fixes this: ENSO episodes arrive at a
fixed rate and the sample grows by roughly one episode every four years.

Three moves change the arithmetic. Each is a different way of not throwing away the months.

1. **Change the estimand** so that every month inside the window is an observation rather than
   one seventeenth of one. Dispersion does this (W1).
2. **Change the treatment** to something that arrives more often than an episode does. Forecast
   revisions do this: monthly, signed, and — unlike an onset — genuinely news (W3).
3. **Change the outcome** to a physical intermediate whose sample size is set by geography
   rather than by ENSO (W2's pending chain).

W4 and W5 are not power moves. W4 opens an estimand that spot indexes cannot see at all, and W5
is a split that costs power and is included because its result is interpretable either way.

### A prior commitment about what these can deliver

The honest expectation, recorded before any of these runs, is that **W1 is the most likely to
produce a publishable positive result and W5 the least**. If W1 also comes back null, the
correct conclusion is not to invent a sixth design: it is that a commodity-price ENSO effect
large enough for this data to see does not exist, which is a real finding and should be written
as one. This paragraph exists so that a later entry cannot quietly reinterpret a null as a
motivation for more searching.

## Program-level multiplicity

Design review F5 observed that each stage defines its own FDR family and nothing counts across
stages. Five new stages make that worse, so the v3 contract carries a rule the earlier ones did
not:

- Every workstream declares its family size and its primary statistic **in its config file,
  before the stage runs**, and writes both into its receipt.
- `config/findings_v3.yaml` holds the program-level register: one row per workstream, the
  number of primary tests it contributes, and its designated primary endpoint. The count of
  primary tests across the whole v3 program is fixed by that file.
- Each stage reports its own within-family correction *and* a program-level Bonferroni
  threshold against the registered total. A result is described as a finding only if it clears
  the program-level threshold; anything clearing only the within-stage threshold is reported as
  exploratory, in those words.

This is deliberately conservative. The alternative — a joint resampling correction across
stages with different units of observation — is not well defined here, and a stated Bonferroni
that a reader can recompute is worth more than a sophisticated correction they cannot check.

---

## W1 — Dispersion: variance and tail frequency at the event window

**The power argument.** A mean over 17 episodes has 17 effective observations. A variance
computed over the same windows has one observation per commodity-month: roughly 17 × 37 ≈ 600
inside the windows, against the rest of a ~780-month sample outside them. Variance is not a
harder quantity to estimate than a mean — at this sample size it is a much easier one, because
the design stops discarding the within-window months.

**Estimand.** Two, both per commodity, both ratios:

- `log_variance_ratio`: log of the variance of seasonal-adjusted monthly log returns inside
  event windows divided by the variance outside them.
- `log_exceedance_ratio`: log of the ratio of the frequency of |return| above the commodity's
  own unconditional quantile, inside versus outside.

The second exists because a variance ratio can be driven by one month. If the two disagree, the
result is a single outlier and is reported as one.

**The null is the hard part.** Commodity returns have volatility clustering, so any test that
treats months as exchangeable will find "significant" variance differences everywhere — the
negative controls would light up first. The null must preserve the return series' own
volatility structure exactly and vary only the *timing* of ENSO. So: **circular whole-year
shifts of the episode mask**, the same device the specification-curve stage already uses. A
whole-year shift preserves ENSO's phase-locking to the calendar, preserves each commodity's
seasonality, and preserves its volatility clustering exactly, because the return series is never
touched. The reference distribution is the ~60 available shifts; that is small, so the smallest
attainable p-value is 1/61 ≈ 0.0164 and the contract says so up front rather than reporting
`p < 0.001` from a bootstrap that would not be valid here.

**Falsification.** Negative controls run through the identical path. Precious metals have
famously regime-dependent volatility, so if the shift null is doing its job they should *not*
reject; if they do, the null is not preserving what it claims to and W1 fails on its own terms.

**Cost.** No new data. Reads the existing monthly panel and episode table.

## W2 — Cold phase, signed hypotheses, and a disruption endpoint

**The mis-specification this repairs.** `config/commodities.yaml` records the Australian coal
pathway as *"rainfall affects mining and export logistics"*. That is a La Niña story: eastern
Australian rainfall is one of the strongest and best-documented ENSO teleconnections anywhere,
and the mechanism is flooded pits and washed-out rail, which raises prices. The frozen design
tests it as a **warm**-phase **mean** return, and duly finds −0.228. The registry wrote down a
mechanism and then tested something else.

**Three changes, all prespecified here.**

1. *Cold episodes.* `enso.py` already builds them; no new machinery.
2. *Signed hypotheses.* Every commodity in the cold-phase family declares a direction in the
   config — supply disruption is positive, demand loss is negative — and the test is one-sided
   against that registered direction. A signed prediction made in advance is worth far more
   evidentially than a two-sided test, and it costs nothing to commit to it in a file whose hash
   is recorded. A commodity with no defensible signed prediction is excluded rather than given
   a two-sided pass.
3. *Disruption endpoint.* A flood does not move the annual mean; it produces a spike. The
   endpoint is the frequency of upper-tail monthly returns inside cold windows, with the mean
   reported alongside as a secondary.

**The chain, which is the actual prize.** The reason to prefer coal to palm oil as a mechanism
target is that every intermediate is public and monthly: Queensland rainfall (BoM), coal export
throughput (port authority), and export volumes. That gives hundreds of observations per link
instead of seventeen, which is exactly where the palm-oil chain failed. Those inputs are not in
this repository and acquiring them is a separate piece of work, so W2 ships the chain as a
**strict interface** in the `mechanism_v2.yaml` idiom: required columns, information-date
ordering, cluster requirements, and a hard refusal to report a link that has no data behind it.
An input schema is not evidence, and the receipt says so structurally.

## W3 — Forecast revisions: test the news, not the weather

**The identification argument, which is the most interesting thing in this program.** Every
study in this literature, this one included, event-studies a *realization*: it dates an onset
and measures what happened afterwards. But ENSO is forecast six to nine months ahead and those
forecasts are public. If commodity markets are even weakly efficient, an onset declaration is
not news — it is the confirmation of something already priced — and finding no abnormal return
after it is exactly what an efficient market with a real ENSO effect would produce.

That single observation is a sufficient alternative explanation for most of this project's null
results, and nothing in the repository currently distinguishes it from "there is no effect".

**Design.** The treatment becomes the *revision*: the change between consecutive monthly
issuances in the forecast probability of El Niño conditions at a fixed lead. Revisions are
monthly, signed, and continuous, so the sample is issuances (roughly 280 monthly observations
from the early 2000s) rather than 17 episodes. The outcome is the contemporaneous monthly
return; the coefficient answers "does a surprise upgrade in El Niño odds move this price in the
month the surprise arrives".

**Two placebos, both essential and both cheap.**

- *Lag placebo.* Regress on revisions lagged one to three months. Under efficiency these are
  zero: the news is already in the price. A non-zero lag coefficient means either the market is
  slow or the design has a mechanical artifact, and either way the contemporaneous estimate
  cannot be read as a news response.
- *Lead placebo.* Regress on the *next* month's revision. This must be zero. A non-zero lead
  coefficient is a pre-trend and would indicate the revision series is picking up something
  already in prices — the standard falsification for this design, and the one that makes the
  contemporaneous result interpretable.

**Both outcomes are publishable**, which is the property this program has otherwise lacked.
A response to revisions but not to realizations is a market-efficiency finding and explains the
whole preceding null program. No response to either is much stronger evidence against a material
ENSO price effect than anything currently in the repository, because it removes the "the effect
was anticipated" defence.

**Data.** The CPC/IRI probabilistic ENSO forecast archive. The stage ships with a downloader,
a strict parser, and a refusal to run without a hash-verified real manifest, on the existing
rule that an input schema is not a result. **The archive's exact layout and start date must be
verified against the live source before any number from this stage is quoted.**

## W4 — Term structure: an estimand spot indexes cannot see

A supply shock shows up in the shape of the forward curve before it shows up in the level: the
front contract moves relative to the deferred, and the curve tips toward backwardation. The
World Bank series this project runs on are spot price indexes, which are structurally blind to
that, and the +12-month cumulative-return endpoint is close to the worst available way to look
for a transient supply squeeze.

W4 defines the estimand — the front-minus-deferred log spread, and the ENSO response *of the
spread* rather than of the level — on top of the existing `tradability.py` adapter, which
already refuses to splice returns across a roll without an explicit roll-yield method. It is
gated on licensed futures data that this project does not have, so it produces no empirical
result, exactly as the v2 futures gate does today. It is specified now so that the estimand is
frozen before anyone looks at contract data, which is the only moment at which freezing it is
worth anything.

## W5 — Eastern versus Central Pacific flavour

Design review F4 flagged that pooling EP and CP ("Modoki") events averages two treatments with
materially different teleconnection patterns over precisely the production regions in this
registry, and that this is a plausible reason the phase-specificity contrast comes out null.

The classification is cheap: Niño 3 and Niño 4 come from the NOAA PSL source family the
climate-index downloader already handles, and the relative-magnitude rule at each episode's peak
is a few lines. The split is the problem — it divides 17 episodes into two groups of roughly
eight, and the minimum detectable effect within a subgroup will be enormous.

W5 is therefore specified as a **diagnostic that reports its own MDE first**. The stage
computes the per-subgroup minimum detectable effect before it reports any contrast, and the
contract requires that number to be quoted in the same sentence as any estimate. Its likely
honest output is "the flavour split cannot be resolved at this sample size", which is worth
recording once, with a number attached, so that it stops being raised as an open possibility.

---

## Order of implementation

1. **W1** — no new data, largest power gain, and its shift null is reused by W2.
2. **W5** — cheap, and its MDE result bounds how seriously any later subgroup claim can be taken.
3. **W2** — price side runs on existing data; the chain ships as a gated interface.
4. **W3** — most novel and most valuable, but the only one with an external data dependency on
   the critical path.
5. **W4** — interface only.

## What would make this program wrong

Recorded here so it can be checked later rather than argued about afterwards:

- If W1's negative controls reject under the shift null, the null does not preserve what it
  claims to and every W1 number is void.
- If W3's lead placebo is non-zero, the contemporaneous news coefficient is not a news response
  and must not be reported as one.
- If W2's signed hypotheses are revised after seeing a result, the one-sided tests become
  two-sided tests with a story, and the config hash will show it.
- If the program-level register in `config/findings_v3.yaml` is edited after any stage runs, the
  multiplicity correction is meaningless. It is frozen in the same commit as the first stage.
