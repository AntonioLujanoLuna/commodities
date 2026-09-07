# Current results

This note records the latest locally validated analysis state. It is a compact interpretation of
the hash-linked tables, not a substitute for the detailed data dictionary or a claim of causal
identification.

It is **overwritten on every run** and describes only the snapshot named below. The dated,
append-only record of decisions, findings and retired specifications lives in
[`reports/log/`](log/README.md); open [the 2026-09-06 design
review](log/2026-09-06-design-review.md) for known limitations of the numbers on this page.

The blocks between `generated:` markers are produced by `make report` from the stage receipts
under `tables/<snapshot>/`, so their numbers cannot drift from the run they describe; the
generator refuses to quote a receipt whose artifacts have since changed, and `make report-check`
fails if this file no longer matches them. Everything outside those markers is hand-written
interpretation.

## Run identity

<!-- generated:run-identity start -->

- Data snapshot: `2026-09-06`
- Stage receipts: `inference`, `placebo`, `power`, `dose_response`, `macro`, `fragility`, `robustness`, `specificity`, `endpoint`, `financial`, `palm`, `external_exposure`, `panel`, `specification_curve`, `forecast`, `program_timing_null`
- `commodities.yaml` SHA-256: `9a46280084782b9cf1b09b217a964a0774d725e6e4a46669911a60cd89f1a046`
- `dose_response.yaml` SHA-256: `6650ed543f2483b5c1dc35e885df8b82f1a6121f00feaa5d1328ddae39f14b30`
- `endpoint_diagnostics.yaml` SHA-256: `3faf58d68cb94ac3899f3fc0aee577173f8331fe5a8011914314b5fcfe543866`
- `exposure_v2.yaml` SHA-256: `f6e01cc6c56cec042a4ddf61b8e6de639a2244c6efbe4ef19effcfe112543e98`
- `financial_robustness.yaml` SHA-256: `3bc06010b146d7bb1b7a4e3fed811012a284a84a6b7fd662fbee8b33c46275a1`
- `palm_oil_mechanism.yaml` SHA-256: `32c0308feb2e255f76903b39311b5ad46ff8e536e035968158a10354c9a2c867`
- `panel.yaml` SHA-256: `a42f4702d88bb338770fc4e80353b1964a052dfae4643ba7289acbdbb41349bd`
- `research.yaml` SHA-256: `faf1582043e0a3b52f42539cd3850b509b364f4b2767c178dda2b809dbd9c614`
- `specification_curve.yaml` SHA-256: `1aeab89ab10fb3c2235396550cd8f8f8478d007e6a00c5c32d86c11e46f0f72a`
- `specificity.yaml` SHA-256: `aba273464931378a9c1056513c9c536c6438e72f2694bb23f1f05cee738848ed`
- `validation_v2.yaml` SHA-256: `f002ab4d284c7e1526598a575e1b9c65278dfdd6a4e82785be12072bf751e40a`

<!-- generated:run-identity end -->

<!-- generated:run-context start -->

- Generated for snapshot: `2026-09-06`
- Analysis source-tree SHA-256: `f5ef55e97b13958f31aaf2206189afb6980b193c9c65cc925dc106cf6d9c3829`
- Primary bootstrap: 10000 whole-episode draws
- Panel bootstrap: 2000 block draws per specification

<!-- generated:run-context end -->

## Evidence summary

<!-- generated:evidence-summary start -->

| Stage | Result |
|---|---|
| Primary event study | Candidate family size 30; BH rejections 10; BY rejections 6; Westfall-Young rejections 5; Sign agreement 26; Controls with raw p<0.05 3 |
| Neutral-date placebo | Eligible anchors 105; Candidates passing gates 8; Controls rejecting raw 3; Onset mean year 1992.1; Placebo mean year 1985.9; Era-balance p 0.1138 |
| Minimum detectable effect | Median MDE, marginal 0.1917; Median MDE, family bound 0.3377; Candidates below their own MDE 22; Candidates with unreachable MDE 0 |
| Episode-amplitude dose response | Candidate family size 30; Positive slopes 10; Raw p<0.05 2; BH rejections 0; Max-t rejections 0; Controls with raw p<0.05 0 |
| External macro controls | Macro episodes 17; Models estimated 71; Candidates passing macro gates 9; Controls rejecting bootstrap 3; Controls rejecting placebo 3 |
| Leave one episode out | Episode deletions 17; Candidates surviving every deletion 7; Controls rejecting after every deletion 3 |
| Timing and index grid | Candidates passing every cell 3; Controls failing specificity everywhere 1 |
| Warm versus cold | Candidate cells 120; Control cells 12; Control direction-specific cells 0 |
| Endpoint diagnostics | Control log-mean rejections 13; Control time-trend rejections 2; Control regime correlations 9 |
| Financial controls | Models estimated 122; Control raw rejections 3; Control contrast rejections 0 |
| Palm-oil mechanism | Tests 18; Complete physical chain no |
| External physical exposure | Included crop candidates 21; Explicitly excluded candidates 11; Minimum crop area in hotspot support 0.2110 |
| Exposure-weighted panel | Primary exposure estimate 0.04245; Primary studentized p 0.20840; Control interaction p 0.48776; Weight-mapping permutation p 0.52674; Exposure cells passing grid FDR 0 |
| Whole-year timing null | Specification cells 20; Shifted alignments 60; Joint timing p 0.4098; Maximum-t timing p 0.7377 |
| Recursive forecast benchmark | Forecast cells 9; Cells improving RMSE 8; Loss tests rejecting raw 1; Proxy cells beating long-only 5; Best cell commodity Rubber, RSS3; Best cell horizon 12; Best cell p 0.0137; Best cell RMSE improvement 0.0384; Best cell excess over long-only -1.1752; Genuine out of sample no; Tradability claim permitted no |
| Selected-family timing null | Specification cells 96; Shifted alignments 60; Median-|t| timing p 0.0164; Maximum-|t| timing p 0.0164; Outcome-informed selection yes |

<!-- generated:evidence-summary end -->

### What those numbers mean

Generated one line per stage from the same receipts as the evidence table above.

<!-- generated:interpretations start -->

| Stage | Interpretation |
|---|---|
| Primary event study | 10 candidate associations survive BH and 5 survive Westfall-Young; the control failures prevent a causal reading. |
| Neutral-date placebo | 8 candidates pass the current gates; the mean-year imbalance has p=0.1138 and is diagnostic, not a gate. |
| Minimum detectable effect | 22 of 30 candidates are below their own marginal MDE and are underpowered, not established nulls. |
| Episode amplitude | 0 candidate slopes survive BH and 0 survive max-t; 0 controls reject at raw 5%. |
| Timing/index grid | 3 candidates pass every timing/index cell and the leave-one-episode-out gate. |
| Warm versus cold | 0 control cells show a direction-specific contrast; the broader candidate pattern remains mostly phase-nonspecific. |
| Financial controls | 3 controls still reject in levels, while 0 reject the warm-minus-cold contrast. |
| Palm-oil mechanism | The prespecified physical chain is incomplete. |
| External physical exposure | 21 crop candidates have outcome-independent weights; 11 unsupported candidates are excluded rather than coded as zero. |
| Exposure panel | The primary interval is not robust under every block sensitivity, but 0 exposure cells survive grid FDR. |
| Whole-year timing null | The observed specification family has joint timing p=0.4098 against circular whole-year shifts. |
| Recursive forecast benchmark | ENSO improves RMSE in 8 of 9 cells, with paired-loss p<0.05 in 1 cell; final RONI and non-investable indexes make this pseudo-OOS. |
| Best forecast cell | Rubber, RSS3 at 12 months has paired-loss p=0.0137 and RMSE improvement 0.0384, but its price-index strategy excess over long-only is -1.1752. |
| Selected-family timing null | The locked 96-cell selected family has median-|t| timing p=0.0164, but selection used the discovery outcomes and this is retrospective calibration, not independent validation. |

<!-- generated:interpretations end -->

## Stage receipts

<!-- generated:receipts start -->

| Stage | Receipt | SHA-256 |
|---|---|---|
| Primary event study | `inference_summary.json` | `eb87020ddff7b28c22ce29d14ca7cc0873275b7427a25f2a31189e0f53bdcef7` |
| Neutral-date placebo | `placebo_summary.json` | `708288261153d3ccb0f03a23d599a843e84f392ed63046b3d3883353ba6c9885` |
| Minimum detectable effect | `power_summary.json` | `2e82a53771cc1998cac77cf57b2e2ea642bd79f124f928f74530caeaea9d546c` |
| Episode-amplitude dose response | `dose_response_summary.json` | `a4d1b19444c888db9ec64de7ad4875030df74f18b790d63c38fafb3c483158ae` |
| External macro controls | `macro_analysis_summary.json` | `ffc23350b02ee104767c668703728da7438d0d3d7bea3214e49fc25ee2bd9ad7` |
| Leave one episode out | `macro_fragility_run_summary.json` | `e618ade0cc5e79b6a261ad886d2f16fcf2e89cbd6af662d0a9011f21953d114c` |
| Timing and index grid | `robustness_summary.json` | `6f09751e643c96c902839f630b561bcdf942110287a07acc1c609c62e7934bc6` |
| Warm versus cold | `specificity_summary.json` | `ac3ea45455964d669f45a8fbc188f6af0a1b3b1795d6a38e351629d05e50626e` |
| Endpoint diagnostics | `endpoint_diagnostics_summary.json` | `160de331e8e51cfcedb8be26656be3fd7f3aecdb9ee21a772539703672a4127f` |
| Financial controls | `financial_control_summary.json` | `57932049a8c82a5f76b0f868fcfa2d1a05cd8e63d494b7f9ef662ccb42f3cf49` |
| Palm-oil mechanism | `palm_mechanism_summary.json` | `323295e83a6826f62386a9fbcca187a144bc5ba05420ea5e464578b541c73cee` |
| External physical exposure | `external_exposure_summary.json` | `8f5192c97e9a8cb44844bb094dfff273db06404570a53c335defe0feecd69ec5` |
| Exposure-weighted panel | `panel_summary.json` | `d3419f09c0960a793353af35f7426dfd0d9cdafc05031ea753e0cbe3b5fa2d5e` |
| Whole-year timing null | `specification_curve_summary.json` | `f9c09f557ef0ab4923580fd91aa7b34bdbbd7b47953d4dbb573dc62448581126` |
| Recursive forecast benchmark | `forecast_summary.json` | `0db6291a1e48a9b8808997944a3380a8c0611ba8aa786bd72add5acaeddf13f2` |
| Selected-family timing null | `program_timing_null_summary.json` | `aac58a9495181466fbddefdcd4eb0e27e3477b053047da0325003fd2d4b6391a` |

<!-- generated:receipts end -->

## Surviving associations

The frozen primary event study has ten BH discoveries: Australian coal, coconut oil, Arabica and
Robusta coffee, Malaysian logs, European and US natural gas, palm oil, Thai 5% rice and RSS3
rubber. Five survive Westfall-Young FWER: Australian coal, coconut oil, European natural gas, palm
oil and RSS3 rubber.

After the macro, leave-one-episode-out, timing and index gates are combined, only coconut oil, palm
oil and RSS3 rubber remain. All three have positive mean month-12 returns in every timing/index
cell. They are robust historical associations, not validated ENSO mechanisms.

## Panel result

<!-- generated:panel-result start -->

The retrospective external-validation primary panel uses outcome-independent external physical weights, RONI lagged 6 months, commodity and calendar-month fixed effects, and seasonal-adjusted log returns. Its exposure coefficient is `0.042449`, with a year-block 95% interval of `[-0.022915, 0.109700]` and studentized `p=0.20840`. The corresponding negative-control interaction has `p=0.48776`.

The named exposure assignment is at the 73.7th percentile of weight shuffles, with a two-sided mapping-permutation `p=0.52674`. Across the 20-cell panel grid, 0 exposure cells survive BH correction and 0 control cells reject at raw 5%.

The panel-family timing null compares the observed curve with 60 circular whole-year ENSO shifts. Its joint median-|t| p-value is `0.40984` and its maximum-|t| p-value is `0.73770`. The observed alignment is therefore not unusually strong within this panel specification family.

<!-- generated:panel-result end -->

## Bottom line

<!-- generated:bottom-line start -->

The pipeline leaves 3 timing/index-robust historical associations, while 22 candidates remain below their own marginal detection threshold. The phase and negative-control diagnostics still prevent an ENSO-specific causal interpretation, and 0 panel exposure cells survive correction across the specification grid. The retrospective forecast benchmark has paired-loss p<0.05 in 1 cell and does not yet satisfy the genuine out-of-sample gate.

<!-- generated:bottom-line end -->
