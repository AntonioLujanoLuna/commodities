# El Niño Is Back: Which Commodities Actually Care?

**A reproducible 65-year ENSO event-study project on commodity prices.**

> **Implementation status:** real-data ingestion, ENSO episode construction, monthly returns and
> raw, seasonal-adjusted and common-market-adjusted event paths are working and validated.
> The outcome-blind commodity registry and primary endpoint are frozen, and whole-episode
> bootstrap inference, candidate-family FDR and calendar-matched neutral-date placebos are
> implemented. External dollar, CPI and global-activity controls are also implemented. The
> leave-one-episode-out and alternate-index/onset gates are implemented too. The prespecified
> negative-control gate still fails, so the numerical associations below are not ENSO-specific
> findings. An exploratory warm-versus-cold falsification diagnostic is also implemented.
>
> The bootstrap test has been recalibrated: the centred percentile test is anti-conservative at
> this many episodes, and the studentized version is now what the gates read. An exploratory
> exposure-weighted panel with commodity and calendar-month fixed effects is also implemented.
> Its v2 weights come from external crop-area and El Niño drought-hotspot rasters rather than
> observed returns; because the mapping was authored after prior results were visible, it is a
> retrospective external validation rather than a preregistered confirmatory test. A circular
> whole-year timing null evaluates its full 20-cell family.
> Neither design currently establishes an ENSO-specific price effect. This file carries no run
> numbers; they live in
> [`reports/current_results.md`](reports/current_results.md), generated from the run receipts.
> A separately versioned v2 contract now freezes the three discovery survivors before new
> validation inputs are acquired. Its current forecast run is explicitly pseudo-out-of-sample,
> and its regional-mechanism and licensed-futures gates remain closed until their required data
> exist; an input schema is not counted as empirical evidence.

The compact, hash-grounded interpretation of the latest completed run is in
[`reports/current_results.md`](reports/current_results.md), which is overwritten on
every run. The append-only record of decisions, findings and retired
specifications is in [`reports/log/`](reports/log/README.md).

Most El Niño commodity analyses stop at *El Niño dates → average commodity return*. That
calculation is easy and almost always misleading: there have only ever been a couple of dozen
episodes, commodity prices are seasonal, they share a macro cycle, and testing thirty-five
commodities across twenty-four horizons produces "significant" results by construction.

This project asks a harder question:

> When ENSO strengthens, which commodity prices show **statistically robust** abnormal returns,
> **when** do those returns occur, and can each one be traced to an **identifiable physical
> supply mechanism**?

Four levels of evidence, in increasing order of strength: historical association → statistical
robustness → physical transmission mechanism → out-of-sample usefulness. A commodity is only
described as having a strong ENSO relationship if it survives all four.

---

## The evidence ladder

Every commodity passes through eight gates, evaluated at a horizon fixed in advance:

| Gate | Question |
|---:|---|
| 1 | Is the effect economically meaningful at the pre-specified horizon? |
| 2 | Do the median, the mean and the share of positive episodes agree? |
| 3 | Is it unusual relative to random dates drawn from ENSO-neutral periods? |
| 4 | Does it survive Benjamini-Hochberg correction across every test the study ran? |
| 5 | Does it survive deleting any single historical episode? |
| 6 | Does it hold under RONI and ONI, and under alternative onset definitions? |
| 7 | Is there a weather-to-supply pathway with the right sign? |
| 8 | Does it improve an out-of-sample forecast or survive implementation costs? |

Commodities land in one of four buckets: **HIGH CONFIDENCE**, **INTERESTING BUT UNPROVEN**,
**FRAGILE**, **NO MATERIAL ENSO EFFECT**. Negative results are published with the same
prominence as positive ones — a study that finds an effect everywhere has found a bug.

---

## Quick start

```bash
uv sync --extra dev --extra spatial
uv run python scripts/download_data.py
uv run python scripts/build_dataset.py
uv run python scripts/build_raw_events.py
uv run python scripts/build_adjusted_events.py
uv run python scripts/build_universe.py
uv run python scripts/run_inference.py
uv run python scripts/run_placebo.py
uv run python scripts/run_power.py
uv run python scripts/run_dose_response.py
uv run python scripts/download_macro_data.py
uv run python scripts/build_macro_dataset.py
uv run python scripts/run_macro_analysis.py
uv run python scripts/run_fragility.py
uv run python scripts/run_robustness.py
uv run python scripts/run_specificity.py
uv run python scripts/run_endpoint_diagnostics.py
uv run python scripts/download_financial_data.py
uv run python scripts/build_financial_dataset.py
uv run python scripts/run_financial_analysis.py
uv run python scripts/download_palm_oil_data.py
uv run python scripts/build_palm_oil_dataset.py
uv run python scripts/run_palm_oil_mechanism.py
uv run python scripts/download_exposure_data.py
uv run python scripts/build_external_exposure.py
uv run python scripts/run_panel_analysis.py
uv run python scripts/run_specification_curve.py
uv run python scripts/run_forecast_analysis.py
uv run python scripts/run_program_timing_null.py
uv run python scripts/build_report.py
uv run python scripts/build_publication.py
```

The downloader creates an immutable, dated snapshot under `data/raw/YYYY-MM-DD/`. Each input has
a JSON provenance sidecar containing its source URL, retrieval time, response metadata, byte
count and SHA-256 hash. The build refuses non-real manifests and verifies every raw hash before
parsing.

Processed outputs land under `data/processed/YYYY-MM-DD/`:

- `enso_monthly.csv`: RONI and ONI keyed to the centre month of each three-month season.
- `world_bank_indices_monthly.csv`: the World Bank Total Index used as the common-market factor.
- `commodity_prices_monthly.parquet`: all source cells, analytical values and quality flags.
- `monthly_panel.parquet`: the overlapping ENSO and price history in tidy long form.
- `summary.json`: coverage, missingness and source-workbook diagnostics.

Synthetic fixtures may be added later for unit and recovery tests, but the production downloader
and processed research panel accept only `data_provenance: real`.

Raw event outputs land under `tables/YYYY-MM-DD/`. They include derived RONI episodes, gap-safe
monthly returns, event paths from month -12 through +24, the pre-specified horizon subset,
coverage diagnostics and a run summary with input and configuration hashes.

The adjustment stage adds monthly and event-level seasonal-adjusted returns, a World Bank Total
Index factor, per-commodity factor models, adjusted horizon tables and their coverage diagnostics.
The universe stage classifies every source series and writes the frozen primary candidate family
and separate negative-control sample. The inference stage writes candidate and control estimates,
whole-episode bootstrap replicates, confidence intervals, p-values and candidate-family q-values.

Macro inputs use a separate immutable snapshot under `data/macro/raw/YYYY-MM-DD/`, with the same
sidecar and SHA-256 receipt rules. The macro analysis refits each commodity outside event windows,
then reruns the bootstrap and neutral-date placebo on the macro-adjusted endpoint.

---

## What the pipeline does

```
scripts/download_data.py     immutable, dated, sha256-verified raw store       [implemented]
        ↓
scripts/build_dataset.py     tidy RONI, ONI and commodity-price panel          [implemented]
        ↓
episodes, monthly returns and raw event-relative paths                         [implemented]
        ↓
seasonal and common commodity-market adjustment                               [implemented]
        ↓
outcome-blind universe, primary endpoint and control freeze                    [implemented]
        ↓
whole-episode bootstrap, sign agreement and candidate-family FDR               [implemented]
        ↓
calendar-matched neutral-date placebo, with an era-balance diagnostic          [implemented]
        ↓
minimum detectable effect at the frozen endpoint                               [implemented]
        ↓
warm-episode peak-amplitude dose-response diagnostic                          [implemented]
        ↓
external dollar, CPI and global-activity controls                              [implemented]
        ↓
leave-one-episode-out bootstrap/FDR fragility                                  [implemented]
        ↓
alternate-index and onset-definition robustness                               [implemented]
        ↓
warm-versus-cold phase-specificity falsification                              [implemented]
        ↓
endpoint convexity, time-drift and observed-regime diagnostics                [implemented]
        ↓
real-rate, credit-spread and financial-conditions controls                    [implemented]
        ↓
palm-oil physical-mechanism pilot                                             [implemented]
        ↓
external crop-area x El Niño drought-hotspot exposure weights                [implemented]
        ↓
exposure-weighted panel with commodity and month fixed effects, and its
resampling-block sensitivity                                                  [implemented]
        ↓
whole-year circular-shift null over the 20-cell panel family                  [implemented]
        ↓
locked 96-cell selected-family whole-year timing null                         [implemented]
        ↓
expanding-window forecast comparison on revised index and price-index data    [implemented]
        ↓
regional crop-calendar and timestamped supply-revision input contracts        [implemented; inputs pending]
        ↓
licensed futures, roll and transaction-cost input contract                    [implemented; inputs pending]
        ↓
generated scorecard, publication figures and auditable compact bundle         [implemented]
        ↓
new-input mechanism validation, real-time-vintage forecasts and tradability   [data pending]
```

The current narrative report in `reports/current_results.md` is generated from auditable tables
and hash-linked run receipts.

The compact publication layer under `reports/artifacts/<snapshot>/` carries the small receipts,
selected result tables, a one-row-per-candidate scorecard and three generated figures. Its
manifest hashes every published file and fingerprints the complete analysis source tree, allowing
a clone to audit the reported numbers without committing the large bootstrap replicate tables.

### Analysis decisions

- **The macro benchmark is estimated outside event windows**, so it is not fitted on the
  episodes it is later asked to price.
- **The bootstrap resamples whole episodes**, not months, and shares one draw across all cells,
  so a replicate is one coherent alternative history.
- **Placebo dates preserve the onset calendar-month distribution**, have `|RONI| < 0.5`, and sit
  outside every real onset's frozen -12/+24 window. Each replicate samples without replacement.
- **q-values are what get interpreted.** Benjamini-Hochberg is applied only to the 30 eligible
  mechanism candidates; controls are reported separately as diagnostics.
- **The bootstrap test is studentized.** Each replicate mean is divided by that replicate's own
  standard error rather than being compared on the observed scale. At seventeen episodes with
  heavy-tailed returns this matters: under a complete null with a shared factor across
  commodities, the centred percentile test plus Benjamini-Hochberg produces at least one false
  rejection in 13% of synthetic draws against a nominal 5%, while the studentized version
  produces one in 2%. Both p-values are always written out, and `p_value_method` in
  `research.yaml` records which one the gates read.
- **Multiplicity is reported three ways.** Benjamini-Hochberg is the pre-registered gate.
  Benjamini-Yekutieli repeats it without assuming anything about how the commodity tests
  co-move. Westfall-Young step-down max-T reuses the stored shared-draw replicates to build the
  joint null of the largest statistic, controlling the family-wise error rate under the
  dependence the resampling scheme already preserves. The bootstrap deliberately shares one
  episode draw across every commodity; applying only a marginal correction afterwards throws
  that information away. Westfall-Young is a strictly higher bar than either FDR rule and is
  reported as such, not as a replacement.
- **Every null is reported with what could have been detected.** The minimum-detectable-effect
  stage inverts the studentized bootstrap that the gates read: an additive shift on the endpoint
  moves the sample mean by its own size and leaves the standard error alone, so the stored null
  replicates already are the reference distribution and no new resampling is needed. A commodity
  whose estimate falls below its own minimum detectable effect is uninformative rather than null,
  and does not belong in the NO MATERIAL ENSO EFFECT bucket.
- **Episode amplitude is a secondary endpoint, not a moved primary gate.** The dose-response stage
  regresses the frozen month-12 return on standardized peak RONI across the same warm episodes.
  It permutes one whole-episode amplitude mapping across every commodity, preserving cross-series
  dependence, and reports candidate-family BH and max-|t| correction. It neither splits the
  17-episode sample nor changes the binary event-study contract.
- **The placebo is checked for era balance.** Anchors are matched on calendar month, but they must
  also be neutral and outside every event window, so the eligible pool is whatever quiet stretches
  survive both filters. If those stretches sit in a different part of history than the onsets, the
  specificity gate is partly comparing macro-financial regimes rather than ENSO phases -- which is
  a candidate explanation for the precious-metal failure that no additional regressor can remove.
  The diagnostic tests that against the placebo's own draw distribution and changes no gate.
- **The panel's resampling block is reported at several lengths.** A calendar year is shorter than
  an ENSO episode and its price response, and ENSO peaks in November-January, so a January boundary
  splits nearly every event. Blocks shorter than the dependence understate the standard error, so
  the frozen calendar-year interval is the optimistic one; May-aligned one-, two- and three-year
  blocks are reported alongside it.
- **The implemented forecast benchmark is strictly recursive** and labelled
  *pseudo*-out-of-sample, because it uses final revised ENSO indices and non-investable World
  Bank price indexes. Genuine out-of-sample and tradability labels are structurally forbidden
  until timestamped vintages and licensed futures inputs pass their contracts.

Two timing decisions are already implemented. Episodes are derived from the configured threshold
and persistence rule rather than transcribed. Event paths are built around both retrospective
onset and an observable date: the fifth qualifying season plus two months for the centered window
and publication timing. The observable date is a conservative historical timing convention, not
a reconstruction of archived release vintages.

Four return measures now travel together: raw, seasonally adjusted, common-market adjusted and
external-macro adjusted.
Seasonality is estimated by commodity and calendar month outside qualifying warm-episode months.
Each commodity's loading on the World Bank Total Index is estimated only outside the union of all
configured -12/+24 event windows. This removes a transparent common commodity-market component.
The external robustness model adds BIS dollar appreciation, BLS CPI inflation and the first
difference of the Dallas Fed/Kilian global real-activity index; it is fitted outside the same
event-window union.

The primary inferential contract is frozen in `config/commodities.yaml`: retrospective onset,
month +12, market-adjusted cumulative return, a two-sided alternative and at least 10 valid
episodes. The registry contains 32 mechanism candidates and three precious-metal negative
controls. Controls are diagnostics outside the candidate FDR family. Every one of the 71 source
series is classified; an unclassified future source column stops the build.

## What each stage found

Numbers live in [`reports/current_results.md`](reports/current_results.md), where they are
generated from the run receipts rather than typed, and are regenerated whenever the pipeline is
rerun. This section describes what each stage does and what it concluded; it deliberately carries
no counts, so it cannot drift away from the artifacts.

**Primary event study.** A minority of the 30 candidates reject at candidate-family BH FDR;
fewer survive the more conservative BY adjustment, and fewer still survive Westfall-Young FWER.
Most candidates have agreeing mean, median and sign share. A handful pass the bootstrap, sign and
neutral-date placebo gates together. Fish meal lacks the frozen 90% valid-placebo-replicate
coverage and is conservatively assigned placebo p=1.

**External macro controls.** The specification estimates a model for every commodity and leaves
roughly the same set of candidates passing its bootstrap, sign and placebo gates. It does not
repair the specificity failure: Gold, Platinum and Silver still reject under both macro-adjusted
bootstrap and placebo inference, so none of the survivors is promoted to a validated El Niño
mechanism.

**Leave one episode out.** Each macro-covered episode is removed in turn and the bootstrap and
candidate-family FDR are rerun. Most of the passing candidates survive every deletion, and the
precious metals remain positive and significant after every one. The unexplained control pattern
is therefore broad across episodes rather than an artifact of one exceptional event.

**Timing and index grid.** Macro-adjusted paths are independently rebuilt for RONI and ONI at
both retrospective and conservative observable anchors. Three candidates pass bootstrap FDR,
calendar-matched placebo FDR and the direction and sign gates in all four cells and also pass the
leave-one-episode-out gate: coconut oil, palm oil and RSS3 rubber, with positive mean +12 returns
throughout. Gold rejects in every diagnostic bootstrap cell, and Gold and Silver keep failing the
placebo specificity diagnostic across the grid. Those three associations are robust to timing and
index choice and still fail the study's specificity requirement.

**Warm versus cold.** An exploratory falsification stage compares warm episodes directly with
persistently cold ones using shared-label randomization. No precious-metal cell distinguishes warm
from cold at raw 5%; gold and silver are frequently positive after cold episodes too. Among the
three grid survivors only coconut oil rejects the warm-minus-cold contrast consistently, and
palm oil in a single cell. The dominant pattern is phase-nonspecific. This is reported separately
and does not rewrite the frozen primary design.

**Endpoint diagnostics.** Simple-return convexity is ruled out as the main explanation: the three
survivors stay significant in cumulative log-return space in every warm cell, and about half the
precious-metal warm/cold cells reject in log space as well. Jensen gaps are present but too small
to create the result. Two control cells show raw time trends, and a handful of control/factor
correlations reject at unadjusted 5%, led by Silver's relationship with dollar changes; none is
family-adjusted, because controls remain diagnostics. This narrows the problem toward missing
financial-regime structure and the non-specific timing of ENSO extremes rather than arithmetic
compounding.

**Financial controls.** An exploratory extended-control model adds a CPI-deflated three-month
Treasury rate, the Moody's Baa-minus-10-year-Treasury spread and the monthly mean Chicago Fed
NFCI. These remove most of the precious-metal bootstrap rejections -- Platinum and Silver stop
rejecting, Gold does not -- and no control distinguishes warm from cold. Coconut oil, palm oil and
rubber keep positive warm estimates without consistently passing the direct phase contrast.
Financial controls narrow the failure without establishing ENSO specificity.

**Palm-oil mechanism pilot.** Seven transparent NASA POWER weather points across Indonesian and
Malaysian producing regions are combined with FAOSTAT oil-palm fruit and palm-oil series. RONI
significantly predicts contemporaneous drying and warming; rainfall one year earlier predicts
higher fruit yield and contemporaneous heat predicts lower yield, both surviving within-link FDR
with the prespecified signs. Aggregate palm-oil production growth does not predict annual
palm-oil price growth at either the contemporaneous or the one-year lag. The physical chain
therefore fails at the supply-to-price link, and palm oil remains interesting but unproven.

### The exposure-weighted panel

Every control block added so far has narrowed the precious-metal problem without removing it.
That pattern is what a specification error looks like rather than a missing regressor. The event
study defines an abnormal return against a baseline fitted outside the union of all -12/+24 warm
windows, and with seventeen episodes those windows cover most of the sample, so the baseline is a
small residual of history that is itself selected on the ENSO state. Adding a control only helps
if the confound is a variable someone thought of.

The panel stage removes that whole class of problem by construction:

```
y[c,t] = a[c] + d[t] + b1 * (w[c] * ENSO[t-L]) + b2 * (control[c] * ENSO[t-L]) + e[c,t]
```

A fixed effect on every calendar month absorbs whatever global regime moves gold in that month,
named or not. The price is that the ENSO level goes with it: identification comes only from the
cross-section, so `b1` is the differential response per unit of physical exposure relative to the
same-month average, not a level effect, and a shock that moved every commodity identically would
be invisible. The negative controls carry zero exposure, so `b2` is their own ENSO response
measured against the same month effects, and a design that is working reports it as
indistinguishable from zero. That makes the negative-control test structural rather than
empirical.

The v2 exposure weights are outcome-independent. `config/exposure_v2.yaml` maps each supported
crop commodity to a SPAM 2020 physical crop-area raster and combines it with FAO's historical
El Niño drought-hotspot layer. The weight is the crop-area-weighted positive hotspot burden over
all mapped crop area. Candidates without a defensible crop raster are explicitly excluded rather
than assigned zero, and the build refuses incomplete or extra mappings. The recipe does not read
commodity returns, but was frozen after earlier outcomes were visible, so its declared scope is
retrospective external validation. The `uniform` variant
gives every included candidate a weight of one and reduces the design to a candidate-versus-control
contrast. Under uniform weights the two interactions are collinear once the fixed effects are
swept out, so only that term is fitted.

Inference resamples whole calendar years of the cross-section with replacement, since ENSO is a
single time series and months are not independent draws. A year drawn twice receives two separate
sets of month effects, keeping a replicate one coherent alternative history in the same sense as
the episode bootstrap. The grid covers RONI and ONI, lags of 0 to 12 months, and both weighting
schemes; the exposure term across cells is one FDR family and the control term stays outside it.

With the external v2 weights, the primary RONI lag-six interval crosses zero, its mapping is not
unusual under permutation, and no exposure cell survives FDR across the grid. A separate
specification-curve stage shifts the ENSO series circularly by each whole-year offset from 1 to 60,
refits all 20 cells, and compares the observed median and maximum absolute naive t statistics with
those null alignments. This is a joint timing null for the panel family, not for every stage in the
project, and uses naive t statistics so the same tractable statistic is evaluated across all 1,220
fits. Current estimates, intervals and finite-sample p-values are generated in
[`reports/current_results.md`](reports/current_results.md).

Two limitations are worth stating plainly. Two-way fixed effects absorb *additive* common shocks,
not heterogeneous loadings on them, so a control series that loads three times as heavily on a
global factor still contributes extra noise to `b2` -- measurably more variance, but no bias, in
the synthetic tests. And because the month effects are estimated from the same thirty-five
series, a genuinely broad ENSO effect is partly absorbed into them; the design trades level
identification for immunity to unmodelled global regimes.

This stage is exploratory. It does not promote or demote any commodity in the frozen event-study
contract.

### Validation v2

`config/validation_v2.yaml` freezes coconut oil, palm oil and RSS3 rubber after the discovery
snapshot and before any new validation inputs are inspected. It is deliberately described as a
post-discovery validation contract, not a retrospective preregistration.

Two parts can run on the existing snapshot. The recursive forecast stage compares a return-lag
and calendar-month baseline with a nested ENSO model at 3, 6 and 12 months. A training target is
admitted only after its endpoint is observable at the forecast origin, preventing overlapping
horizons from leaking future returns into the fit. Paired loss inference resamples calendar-year
blocks. The strategy calculation is only a price-index diagnostic and reports its excess over a
long-only benchmark; it cannot satisfy the tradability gate.

The selected-family timing null reconstructs a 96-cell family: three commodities, RONI and ONI,
retrospective and observable anchors, four horizons and two adjusted-return endpoints. Sixty
whole-year circular shifts provide one joint reference distribution. Because the three
commodities were selected using the discovery results, this is retrospective calibration of the
locked family, not independent confirmation.

`mechanism_v2.yaml` and the mechanism validator require three timestamp-safe links: ENSO to local
weather during an externally sourced active crop season, production-weighted weather to yield
surprise, and supply-forecast revision to the subsequent price response. The futures adapter
requires contract-level settlements, volume, open interest, expiry and information dates, and
leaves roll-crossing returns undefined until roll yield is supplied explicitly. Neither layer
will manufacture substitutes from the World Bank price indexes.

`config/validation_sources.yaml` records the authoritative acquisition targets without treating
links as data. The current CPC RONI series and outlook do not supply the archived issue-by-issue
history required by the genuine real-time gate. USDA FAS PSD exposes forecast records and release
dates but still requires an authenticated acquisition and commodity-coverage audit. Palm-oil and
rubber contract histories are exchange data and are not present in this repository.

---

## Current repository layout

```
config/            official sources, research settings, frozen commodity registry
                   and the panel exposure weights
data/raw/          immutable date-stamped downloads + .meta.json provenance sidecars
data/processed/    tidy real-data tables and the joined monthly panel
data/macro/        separate immutable raw and processed external-control snapshots
data/financial/    immutable real-rate, credit-spread and NFCI snapshots
data/mechanisms/   immutable weather and production snapshots for physical pilots
src/enso_commodities/
  download.py      streamed downloads, format checks and immutable snapshots
  parsers.py       NOAA ASCII and Pink Sheet workbook parsers
  dataset.py       hash verification and monthly panel construction
  enso.py          threshold-and-persistence episode construction
  returns.py       gap-safe monthly and event-relative raw returns
  raw_events.py    real-data raw-event output stage and run receipt
  adjustments.py   seasonal estimates, market models and strict adjusted paths
  adjusted_events.py adjusted-event output stage and run receipt
  universe.py      registry validation and primary-family construction
  statistics.py    shared episode bootstrap and Benjamini-Hochberg correction
  inference.py     primary/control inference outputs and hash-linked receipt
  placebo.py       neutral anchors, strict endpoints, matched draws and empirical tests
  placebo_analysis.py real-data placebo outputs and hash-linked receipt
  exchangeability.py whether the placebo samples the same era as the events
  power.py         minimum detectable effect from the stored bootstrap replicates
  power_analysis.py minimum-detectable-effect stage and hash-linked receipt
  dose_response.py amplitude regression and shared episode-label permutation null
  dose_response_analysis.py real-data dose-response outputs and hash-linked receipt
  macro_data.py    BIS, BLS/FRED and Dallas Fed parsing and transformations
  macro_adjustments.py out-of-event multivariate commodity regressions
  macro_analysis.py macro-adjusted bootstrap/placebo outputs and receipt
  fragility.py     leave-one-out candidate and control summaries
  fragility_analysis.py deletion-specific bootstrap/FDR stage and receipt
  robustness.py    four-cell gate aggregation and specificity diagnostics
  robustness_analysis.py RONI/ONI and timing-grid reconstruction and inference
  specificity.py warm/cold randomization and episode-influence diagnostics
  specificity_analysis.py exploratory real-data falsification stage and receipt
  endpoint_diagnostics.py endpoint shape, time trend and regime correlations
  endpoint_analysis.py exploratory endpoint diagnostic stage and receipt
  financial_data.py verified financial-control parsing and transformations
  financial_adjustments.py extended out-of-event OLS residuals
  financial_analysis.py exploratory financial-control inference and receipt
  palm_oil_data.py NASA POWER and FAOSTAT parsing and validation
  palm_oil_mechanism.py weather, yield, production and price link tests
  panel.py         exposure weights, two-way within estimator, year-block bootstrap
  panel_analysis.py exposure-weighted panel grid and hash-linked receipt
  validation.py    frozen v2 contract and conservative evidence labels
  forecasting.py   leakage-safe expanding-window forecasts and proxy cost accounting
  forecast_analysis.py real-data pseudo-out-of-sample forecast receipt
  program_timing_null.py selected event-study family timing null
  program_timing_analysis.py real-data timing-null receipt
  mechanism_validation.py crop-calendar mechanism validation and clustered links
  tradability.py   strict licensed-futures adapter and explicit roll handling
  synthetic.py     fixture generators for recovery and calibration tests only
  provenance.py    file hashing and atomic JSON receipts
scripts/           command-line entry scripts for downloading and building
tests/             ingestion, episode, adjustment, integrity, universe, panel,
                   and synthetic recovery/calibration tests
reports/           data dictionary and the latest-run interpretation
reports/log/       append-only research log: decisions, findings, negative results
reports/artifacts/ compact receipts, result tables, scorecard and publication figures
```

## Reproducibility

The implemented downloader records the retrieval timestamp, source URL, HTTP metadata, byte count
and SHA-256 of each raw file. The dataset builder verifies those hashes and records the source
manifest hash. The raw-event stage also records hashes for its processed inputs and research
configuration. The generated report fingerprints the complete analysis source tree, with text
line endings normalized for cross-platform verification, rather than embedding the current Git
commit: a tracked report containing `HEAD` would invalidate itself as soon as that report was
committed. Runtime manifests belong in future stage receipts rather than being sampled from
whichever machine happens to check the tracked report.

The bootstrap and placebo use a recorded seed, stable independent salts for candidates and
controls, and store every replicate and placebo draw. Leave-one-out scenarios use a stable seed
salt for each deleted episode and retain all scenario-level estimates, p-values and q-values.
The timing/index grid stores every reconstructed episode, model, event path, bootstrap replicate,
placebo draw and placebo replicate, with a hash-linked run receipt.

## Data sources

| Layer | Source | Status |
|---|---|---|
| ENSO (primary) | NOAA CPC Relative Oceanic Niño Index (RONI) | implemented |
| ENSO (robustness) | NOAA CPC Oceanic Niño Index (ONI) | implemented |
| Prices and broad market factor | World Bank Pink Sheet, monthly | implemented |
| Dollar | BIS US narrow nominal effective exchange rate, monthly | implemented |
| Inflation | BLS CPI-U, seasonally adjusted, via FRED | implemented |
| Global activity | Dallas Fed/Kilian global real economic activity index | implemented |
| Short rate | Federal Reserve 3-month Treasury bill rate via FRED | implemented |
| Credit stress | Moody's Baa yield relative to 10-year Treasury via FRED | implemented |
| Financial conditions | Chicago Fed NFCI via FRED | implemented |
| Palm weather | NASA POWER MERRA-2 monthly precipitation and temperature | implemented pilot |
| Palm production | FAOSTAT oil-palm fruit and palm-oil annual series | implemented pilot |
| Global crop exposure | FAO ASIS El Niño drought hotspots + IFPRI/FAO SPAM 2020 crop area | implemented |
| Other weather | ERA5, CHIRPS, or a pre-aggregated regional CSV | contract implemented; data pending |
| Other production/forecast revisions | FAOSTAT, USDA PSD | contract implemented; timestamped inputs pending |
| Futures | vendor-licensed contract data (not redistributed) | adapter implemented; data pending |

Optional-source acquisition and licensed futures data remain later-stage inputs.
Licensed data will not be redistributed.

## Development

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run python -m pytest --cov=enso_commodities --cov-fail-under=50
uv run enso-publication --check --bundle reports/artifacts/2026-09-06
uv run enso-report --check --published-bundle reports/artifacts/2026-09-06
```

Validation is explicit and directly tested. Fatal format, history-length, key-integrity and hash
failures stop the build. Recoverable source-data problems remain visible through raw-value and
quality-flag columns rather than being silently repaired.

Two of the tests check the study rather than a function. `test_pipeline_recovers_a_planted_effect`
plants a known cumulative abnormal return in synthetic prices and asserts the analytical path
returns it, with the market model recovering the loadings it was given and the untouched series
staying flat. `test_gates_hold_their_nominal_size_under_the_complete_null` draws repeated panels
with no effect and a shared factor across commodities, and measures how often each gate fires --
which is the direct measurement of the false-positive rate the negative-control gate is failing
on. Synthetic fixtures live in `src/enso_commodities/synthetic.py` and are never labelled
`data_provenance: real`, so no stage will accept them as input.

## Licence

MIT (see `LICENSE`). Source data carry their own terms; see `config/sources.yaml` and
`config/macro_sources.yaml`.
