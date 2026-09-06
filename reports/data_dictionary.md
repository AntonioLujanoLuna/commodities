# Real-data foundation

## Sources

The source URLs and expected formats are declared in `config/sources.yaml`.

- NOAA CPC RONI, ERSSTv6: <https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt>
- NOAA CPC ONI, ERSSTv6: <https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt>
- World Bank historical monthly Pink Sheet: <https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx>

## ENSO table

`enso_monthly.csv` has one row per centre month of a three-month ENSO season.

| Column | Meaning |
|---|---|
| `date` | First day of the season's centre month |
| `roni` | Relative Oceanic Nino Index anomaly, degrees Celsius |
| `roni_season` | NOAA three-month season code |
| `oni` | Oceanic Nino Index anomaly, degrees Celsius |
| `oni_season` | NOAA three-month season code |

## Commodity-price table

`commodity_prices_monthly.parquet` preserves every named commodity cell from the World Bank
`Monthly Prices` worksheet.

| Column | Meaning |
|---|---|
| `date` | First day of the observation month |
| `period` | Original World Bank `YYYYMmm` period label |
| `commodity` | Original World Bank series name, trimmed of outer whitespace |
| `unit` | Original unit label |
| `source_value` | Numeric value exactly as parsed from the source cell |
| `value` | Analysis-ready positive price; invalid non-positive source values are missing |
| `quality_flag` | Explicit reason an otherwise numeric source cell is excluded |
| `source_column` | One-based column position in the source worksheet |

Missing cells remain missing. No interpolation, backfill, inflation adjustment or outlier removal
is performed in the ingestion layer.

## Monthly panel

`monthly_panel.parquet` inner-joins the commodity table to months for which both RONI and ONI are
available. It retains all commodity columns and adds `roni`, `roni_season`, `oni` and
`oni_season`. The ingestion layer does not calculate returns or define ENSO episodes.

## ENSO episodes

`tables/YYYY-MM-DD/enso_episodes.csv` derives warm episodes from the primary index and parameters
in `config/research.yaml`. It never uses a manually transcribed event list.

| Column | Meaning |
|---|---|
| `onset_date` | First centre month at or above the configured threshold |
| `persistence_date` | Centre month when the minimum-duration rule is first satisfied |
| `observable_date` | Persistence date plus the configured centered-window publication delay |
| `end_date` | Last consecutive qualifying centre month |
| `duration_months` | Number of consecutive qualifying seasons |
| `peak_date`, `peak_value` | Strongest index observation within the qualifying run |
| `end_censored` | Whether the run reaches the final available index observation |

The observable date is a conservative timing convention. It does not recreate historical NOAA
release vintages and must not be described as true real-time data.

## Monthly returns

`commodity_returns_monthly.parquet` adds one-month simple and log returns to the preserved price
records. A return is calculated only when current and previous prices exist at consecutive
calendar months. Missing values and calendar gaps are never bridged. `return_quality_flag`
explains why a return is unavailable.

## Event-relative returns

`raw_event_return_paths.parquet` has one row per episode, anchor type, commodity and relative
month. `retrospective` anchors use `onset_date`; `observable` anchors use `observable_date`.
Relative returns use the price at month -1 as their base:

`raw_cumulative_return(h) = price(anchor + h) / price(anchor - 1) - 1`

`raw_event_returns_horizons.parquet` is the pre-specified horizon subset. Missing base or event
prices remain missing and receive an `event_quality_flag`. These are nominal raw returns, not
seasonally adjusted or macro-adjusted effects.

## Seasonal and market adjustments

`commodity_returns_adjusted_monthly.parquet` keeps raw returns and adds both adjustment layers.
Seasonal means are estimated separately for each commodity and calendar month, excluding the
months from retrospective onset through the end of every qualifying RONI episode. A seasonal
estimate must meet the configured minimum observation count.

`market_factor_monthly.csv` derives log returns from the World Bank `Total Index` in the source
workbook and applies the same calendar-month seasonal adjustment. `market_models.csv` estimates
one ordinary least-squares model per commodity:

`seasonal_adjusted_log_return = alpha + beta * market_seasonal_adjusted_log_return + residual`

The regression sample excludes the union of every configured retrospective event window, currently
month -12 through +24. `market_adjusted_log_return` is the residual from that model. The Total
Index is a common commodity-market factor that contains weighted commodity prices; it is not an
independent dollar, inflation or global-demand control.

`adjusted_event_return_paths.parquet` merges adjusted monthly returns onto the original event grid.
Cumulative adjusted log returns equal the strict sum from the month after the -1 base through the
requested relative month. If any required intermediate monthly return is missing, the cumulative
adjusted return is missing; the calculation never jumps across the gap. Simple cumulative returns
are `exp(cumulative_log_return) - 1`.

`adjusted_event_returns_horizons.parquet` contains only the configured reporting horizons.
`adjusted_event_return_coverage.csv`, `seasonal_estimates.csv`, `market_models.csv` and
`adjusted_event_summary.json` expose sample coverage, estimation counts, model status and hashes.

## Frozen inference universe

`config/commodities.yaml` assigns every source series to exactly one role. The build fails when a
series is duplicated, omitted, unknown, or added by a future workbook without an explicit registry
decision.

- `mechanism_candidate`: a named physical pathway was specified without inspecting event returns.
- `negative_control`: no agricultural supply pathway was specified; controls are diagnostic and
  remain outside the candidate FDR family.
- `excluded`: the registry records a benchmark-duplication, composite-index, processed-product or
  pathway reason.

The frozen primary endpoint uses retrospective onset, month +12,
`market_adjusted_cumulative_return`, a two-sided alternative and a minimum of 10 valid episodes.
The 32 registered mechanism candidates remain in `primary_candidate_sample.parquet` even when they
lack sufficient history. `primary_inference_family.parquet` contains only non-missing episode
returns from eligible candidates. `negative_control_sample.parquet` remains separate.

`commodity_registry.csv` exposes roles, groups, selection notes, valid episode counts and current
eligibility. `universe_summary.json` records the contract, counts and input/output hashes. This
stage performs no hypothesis test and does not inspect or rank return magnitudes.

## Primary bootstrap inference

`config/research.yaml` freezes the bootstrap replicate count, confidence level, FDR threshold and
random seed. `primary_inference_results.csv` reports the episode mean, median, positive share,
percentile confidence interval, centered-null two-sided bootstrap p-value, Benjamini-Hochberg
q-value and rejection indicator for every eligible mechanism candidate.

Each bootstrap replicate samples whole episode IDs with replacement from the common episode
universe. The same draw is used for every candidate, preserving cross-commodity dependence.
Commodity observations that are genuinely unavailable in a sampled episode remain unavailable;
`sampled_observations` in `primary_bootstrap_replicates.parquet` exposes the resulting replicate
sample size. The p-value uses the finite-replicate correction
`(extreme null replicates + 1) / (replicates + 1)`.

`sign_agreement` is true only when the mean, median and majority episode sign point in the same
direction. Benjamini-Hochberg correction is confined to the frozen eligible-candidate family.

Two p-values are always written. `centered_p_value` compares each replicate's centred mean
against the observed mean on the observed scale. `studentized_p_value` divides both by their own
standard errors, so the test adapts to the replicate's dispersion instead of assuming the observed
scale is right; `standard_error` and `t_statistic` carry the observed quantities it uses, and
`studentized_ci_lower`/`studentized_ci_upper` the bootstrap-t interval. `studentized_status` is
`estimated`, `degenerate_observed_scale` when a commodity has no observed dispersion, or
`insufficient_valid_replicates` when too few replicates produce a finite statistic.
`bootstrap_p_value` holds whichever `research.yaml` selected through `p_value_method`, and
`bootstrap_p_value_method` records what that resolved to per commodity, including
`studentized_fallback_centered` where the studentized statistic does not exist. Downstream gates
read `bootstrap_p_value`, so a degenerate commodity falls back rather than dropping out of the
family with a missing value.

Three multiplicity verdicts accompany them. `bh_q_value` is the pre-registered Benjamini-Hochberg
gate and drives `reject_fdr`. `by_q_value` is Benjamini-Yekutieli, valid under arbitrary
dependence, and drives `reject_fdr_arbitrary_dependence`. `westfall_young_p_value` is a step-down
max-T adjustment computed from `studentized_statistic` in the stored replicate table: because one
episode draw is shared across every commodity, the per-replicate maximum over the untested tail is
the joint null of the largest statistic, giving family-wise error control under the dependence the
resampling already preserves. It drives `reject_fwer` and is a strictly higher bar than either FDR
rule. `westfall_young_status` is `estimated`, `insufficient_valid_replicates`, or `outside_family`.

`negative_control_inference.csv` and `negative_control_bootstrap_replicates.parquet` use a
separately salted random stream. Their q-values are diagnostic only and never enter the candidate
family, and carry a `diagnostic_` prefix. `inference_summary.json` records the settings, gate
counts, input hashes and output hashes.
These tests estimate historical association after the implemented seasonal and common-market
adjustments; they do not establish causality. The neutral-date comparison and external-control
robustness model are described next.

## Calendar-matched neutral-date placebo

`placebo_eligible_anchors.csv` contains observed RONI months satisfying `|RONI| < 0.5`, bounded so
the +12 endpoint is observable and outside every real onset's frozen -12/+24 exclusion window.
The exact threshold and exclusion window are frozen in `config/research.yaml`.

Each replicate samples without replacement while exactly reproducing the calendar-month counts of
the 19 real onsets. Candidate and control draws use independently salted streams and are retained
in `primary_placebo_draws.parquet` and `negative_control_placebo_draws.parquet`. These are random
dates applied to real adjusted returns; no price or return observation is simulated.

`placebo_endpoints.parquet` reconstructs the +12 return by strictly summing observed monthly
market-adjusted log returns from relative month 0 through +12. Any missing month invalidates the
endpoint. The run also reconstructs the actual event endpoints from monthly data and stops if they
differ from the frozen primary sample by more than `1e-12`.

`primary_placebo_results.csv` compares each observed episode mean with the empirical distribution
of calendar-matched placebo means. The two-sided p-value doubles the smaller finite-corrected tail
probability. Candidates require at least 10 valid dates in a replicate and at least 90% valid
replicates; an insufficient candidate receives p=1 and remains in the 30-candidate FDR family.
`placebo_bh_q_value` applies Benjamini-Hochberg to that complete frozen family.

`negative_control_placebo_results.csv` remains diagnostic and outside candidate FDR.
`placebo_summary.json` records the contract, coverage, gate counts, endpoint reconstruction error,
input hashes and output hashes. Neutral-date significance is evidence of unusual historical timing,
not causality; the external-control model and fragility gates remain necessary.

## External macro-control robustness model

Macro data use a separate immutable snapshot under `data/macro/raw/YYYY-MM-DD/`. Each downloaded
file has a provenance sidecar, and `manifest.json` records the live URL, retrieval time, response
metadata, byte count and SHA-256 digest. Production parsing accepts only a `data_provenance: real`
manifest whose three files match their recorded hashes.

`macro_controls_monthly.csv` contains:

| Column | Source and transformation |
|---|---|
| `us_neer_log_change` | Consecutive-month log change in the BIS US narrow nominal effective exchange rate; positive means dollar appreciation |
| `us_cpi_log_change` | Consecutive-month log change in seasonally adjusted BLS CPI-U obtained through FRED |
| `global_real_activity_change` | First difference in the Dallas Fed/Kilian global real economic activity index, in index points |

No macro value is interpolated. The common usable history begins in February 1968 because the
Dallas Fed activity index is the shortest series.

For each commodity, `macro_models.csv` estimates the following OLS regression with an intercept:

`seasonal_adjusted_log_return = alpha + beta_market * market_return + beta_neer * neer_change + beta_cpi * cpi_change + beta_activity * activity_change + residual`

The estimation sample excludes the union of all real retrospective -12/+24 event windows and
requires at least 60 complete observations. `macro_adjusted_log_return` is the residual. The model
is a robustness specification, not a replacement for the frozen primary common-market endpoint.

`macro_adjusted_event_paths.parquet` strictly accumulates those residuals over the original event
grid. `macro_primary_results.csv` and `macro_control_results.csv` rerun whole-episode bootstrap and
calendar-matched neutral-date inference over the 17 macro-covered episodes. Candidate FDR still
uses all 30 frozen candidates; precious-metal controls remain outside that family.

`macro_analysis_summary.json` records source and configuration hashes, model counts, gate counts,
endpoint reconstruction error and hashes for all 15 output artifacts. The macro controls reduce
specific omitted-variable risk, but they do not establish a physical ENSO transmission mechanism.

## Leave-one-episode-out fragility

`macro_leave_one_out_candidates.csv` removes each of the 17 macro-covered RONI episodes from the
entire candidate family in turn. For each shared deletion it reruns 10,000 whole-episode bootstrap
draws, centered-null two-sided p-values and Benjamini-Hochberg correction across all 30 frozen
candidates. The deleted episode never contributes to any commodity in that scenario.

A candidate passes the fragility gate only when it already passes the full macro bootstrap,
sign-agreement and placebo gates and, after every deletion, retains its direction, retains
mean/median/majority-sign agreement and rejects at candidate-family FDR 5%.
`macro_fragility_summary.csv` records the leave-one-out mean range, largest departure from the full
mean, most influential episode, rejection count and worst q-value.

`macro_leave_one_out_controls.csv` applies independently salted whole-episode bootstrap draws to
Gold, Platinum and Silver without placing them in the candidate FDR family.
`macro_control_fragility_summary.csv` records whether each diagnostic rejects at raw p<0.05 after
every deletion. `macro_fragility_run_summary.json` stores the frozen contract, row counts and the
complete input/output hash chain. Scenario bootstrap replicates are regenerated from the recorded
seed and episode-specific salt rather than stored; all 561 scenario-level results are retained.

## Timing and index-definition robustness

`robustness_episodes.csv` independently derives qualifying episodes from both real NOAA RONI and
ONI observations. Each index is evaluated at both its retrospective onset and the conservative
observable date defined by the fifth qualifying centered season plus the configured two-month
publication delay. These are historical timing conventions, not archived real-time vintages.

For every one of the four index/anchor cells, `robustness_macro_models.csv`,
`robustness_macro_adjusted_monthly.parquet` and `robustness_event_paths.parquet` independently
re-estimate the external macro model and reconstruct strict +12 cumulative residual returns.
`robustness_candidate_results.csv` records cell-level means, sign agreement, whole-episode
bootstrap p/q-values, calendar-matched placebo p/q-values and the combined pass decision.

The robustness placebo preserves the exact calendar-month counts of the active cell and excludes
dates in the frozen -12/+24 windows around actual retrospective episode onsets. Because some
observable-anchor month cells have fewer eligible neutral dates than active episodes, this
robustness-only placebo samples eligible dates with replacement. The original primary and macro
placebos remain without replacement. All draws and real-return replicate statistics are retained
in `robustness_placebo_draws.parquet` and `robustness_placebo_replicates.parquet`; no commodity,
index, macro or return values are simulated.

`robustness_candidate_summary.csv` requires a candidate to pass all four cell-level gates and the
prior leave-one-episode-out gate. `robustness_control_summary.csv` reports raw-test behavior for
Gold, Platinum and Silver separately from candidate FDR. `robustness_summary.json` records the
four-cell contract, diagnostics and complete input/output hash chain.

## Exploratory specificity diagnostics

`config/specificity.yaml` freezes a separate exploratory contract without changing the primary
research configuration. `specificity_cold_episodes.csv` applies the same absolute threshold,
five-month persistence rule and observable-delay convention to negative RONI and ONI values.

`specificity_event_endpoints.parquet` combines warm and cold +12 endpoints using the macro-adjusted
monthly series already estimated by the robustness stage. `specificity_candidate_results.csv`
tests the warm-minus-cold mean difference by randomly reassigning the observed episode-direction
labels 10,000 times; each draw is shared across commodities. Benjamini-Hochberg correction is
confined to the 30 mechanism candidates in each timing/index cell. Controls remain separate in
`specificity_control_results.csv` and use raw diagnostic p-values.

`specificity_control_episode_influence.csv` reports every warm and cold control endpoint and the
change in its cell mean when that episode is removed. `specificity_contrast_replicates.parquet`
retains every randomization statistic. `specificity_summary.json` labels the stage exploratory and
records the real-data input/output hash chain. No ENSO, price, macro or return values are simulated;
only observed episode labels are randomized under the comparison null.

## Exploratory endpoint diagnostics

`config/endpoint_diagnostics.yaml` freezes a separate diagnostic contract.
`endpoint_diagnostic_results.csv` reports the arithmetic mean, mean cumulative log return,
geometric-equivalent return and Jensen gap for every candidate/control, index, anchor and warm/cold
cell. Whole-episode bootstrap inference is rerun on cumulative log returns; candidate q-values are
computed within the 30-candidate family, while controls retain raw diagnostic p-values.

The same table reports Spearman association between endpoint log return and episode date as a
secular-drift diagnostic. `endpoint_factor_windows.csv` strictly sums the 0-through-12 observed
market, dollar, CPI and global-activity changes for each event. `endpoint_regime_correlations.csv`
tests their Spearman association with commodity log endpoints, with candidate-family correction
within each cell and factor. These correlations diagnose residual regime dependence; they do not
identify causal controls or replace the primary model.

`endpoint_log_bootstrap_replicates.parquet` retains every bootstrap statistic.
`endpoint_diagnostics_summary.json` records the exploratory scope and full real-data hash chain.

## Exploratory financial controls

`data/financial/raw/YYYY-MM-DD/` stores immutable FRED downloads and provenance sidecars for the
Federal Reserve three-month Treasury bill rate, Moody's Baa yield relative to the ten-year
Treasury, and the weekly Chicago Fed NFCI. `financial_controls_monthly.csv` converts NFCI to a
calendar-month mean and derives an ex-post real short-rate proxy by subtracting trailing 12-month
CPI log inflation from the annualized Treasury bill rate. No value is interpolated.

`financial_adjusted_monthly.parquet` refits each commodity's seasonal-adjusted log return on the
existing market, dollar, CPI and global-activity controls plus the real short-rate proxy, credit
spread and NFCI. Models exclude the same -12/+24 warm-event windows and require 120 observations.
This exploratory model is separate from the frozen primary and macro-control specifications.

`financial_control_results.csv` reruns 10,000 whole-episode bootstrap draws for all warm and cold
RONI/ONI timing cells. Candidate FDR remains confined to the 30-candidate family; controls retain
raw diagnostic p-values. `financial_direction_contrasts.csv` directly randomizes warm/cold labels.
Both replicate tables, model estimates, strict endpoints and a complete hash-linked receipt are
retained alongside the results.

## Palm-oil mechanism pilot

`config/palm_oil_mechanism.yaml` freezes the exploratory geography, expected signs and lags.
Seven NASA POWER MERRA-2 points proxy Riau, North Sumatra, Central and East Kalimantan, Sabah,
Sarawak and Johor. They are equal-weighted within country and then combined using the prior year's
FAOSTAT oil-palm fruit production shares. These points are transparent regional proxies, not a
crop-area-weighted gridded exposure.

`palm_weather_monthly.csv` converts corrected precipitation from millimetres per day to monthly
totals and retains monthly two-metre temperature. `palm_weather_anomalies_monthly.csv` subtracts
each point's 1991–2020 calendar-month climatology. `faostat_palm_oil_annual.csv` retains reported
and estimated flags for Indonesian and Malaysian oil-palm fruit area, yield and production plus
palm-oil production from 1961 onward.

`palm_mechanism_results.csv` tests five ENSO-weather monthly lags, three weather-yield annual lags
and two production-price annual lags. Monthly regressions use HAC standard errors with 12 lags;
annual regressions use two, and yield models include country and linear-year controls. FDR is
applied separately within each physical link. `palm_mechanism_summary.json` passes the complete
chain only if every prespecified link contains a sign-correct FDR rejection. This is an exploratory
pilot and does not promote or modify the frozen price-study classification.

## Exposure-weighted panel

`make panel` writes `panel_exposure_weights.csv`, `panel_specification_results.csv`,
`panel_bootstrap_replicates.parquet`, `panel_exposure_weight_permutations.parquet` and
`panel_summary.json`. The stage is exploratory and does not feed the frozen event-study contract.

`panel_exposure_weights.csv` is the registry restricted to mechanism candidates and negative
controls, with `exposure_weight` from `config/panel.yaml` (1.00 direct teleconnection, 0.50
weaker or less consistent, 0.25 pathway mediated through other commodities, 0.00 controls),
`uniform_weight` for the judgement-free variant, and `is_negative_control`. The build fails if the
weights do not exactly cover the candidate registry. The same table carries the exposure version,
method, authorship date, outcome-blind flag and allowed inference scope. Configuration validation
prevents the current post-outcome expert-judgment weights from being labelled outcome-blind or
confirmatory.

`panel_specification_results.csv` has one row per term per grid cell, keyed by `cell`,
`index_definition`, `lag_months`, `weighting` and `term`. `exposure_x_enso` is the differential
response per unit of exposure per unit of the ENSO index, relative to the same-month average
across the estimation sample; `control_x_enso` is the negative controls' own response measured
against the same month effects, and is a diagnostic that should not fire. Under `uniform`
weighting only `exposure_x_enso` is fitted, because the control indicator is then one minus the
candidate indicator and the two terms are collinear once the fixed effects are swept out.

`estimate` is the two-way within OLS coefficient, obtained by alternating projections rather than
an explicit dummy design. `naive_standard_error` treats every commodity-month as an independent
draw and is reported only for comparison. `block_standard_error`, `ci_lower`, `ci_upper`,
`block_bootstrap_p_value` and `studentized_p_value` come from resampling whole calendar years of
the cross-section with replacement, relabelling a repeated year so its two copies carry separate
month effects. `valid_replicates` counts the replicates that produced a solvable design.
`observations`, `units`, `periods`, `within_r_squared` and `design_condition_number` describe the
fitted cell; a rank-deficient design raises rather than being silently fitted.

`bh_q_value` and `reject_fdr` are applied across the `exposure_x_enso` cells only. The control
term stays outside the family, in line with every other stage.

`panel_exposure_weight_permutations.parquet` holds the falsification draws for the primary panel
cell. Each draw shuffles the observed exposure weights across mechanism candidates while controls
remain fixed at zero. The two-sided permutation p-value asks whether the named commodity-to-weight
assignment is unusually informative among arbitrary assignments of exactly the same weights. It
does not make the post-outcome weights prospectively specified.

`panel_summary.json` records the design, the primary cell's estimate, interval and p-values, the
control term's estimate and p-value, grid-level counts, and the input and output hashes.
