# First real-data run of W1 dispersion and W2 cold-phase disruption

- Type: `finding`, `negative`
- Date: 2026-09-08
- Data snapshot: `2026-09-06`
- Follows: [the v3 implementation](2026-09-08-v3-implementation.md)
- Status: W1, the price side of W2, and W5 run; W3, W4, and the W2 physical chain remain unrun.

The first two runnable v3 endpoints were executed against the unchanged `2026-09-06`
real-data snapshot. The run used the contracts frozen before these results were observed:

- `config/dispersion.yaml` SHA-256
  `07ce8cab368950150c7ca2aeb54d4c6d4738b83f34f5a9417a925aa1c02cfc80`
- `config/cold_phase.yaml` SHA-256
  `241db636b49e80e4bd52c2315fb7f7d3aa00c86a5047ddd1115a6f47b2bc72f0`
- `config/findings_v3.yaml` SHA-256
  `43dcf21ef1cae3d57b022c8467cf0470ec50ffdc823e2512f61a34606e2c9419`

## W1 is a clean null

The warm-window dispersion family has whole-year shift `p=0.2154`. No candidate survives
BH and none of the three negative controls rejects at raw 5%, so the falsification gate passes.
Twenty candidate commodities have variance and exceedance statistics pointing in the same
direction, but agreement without corrected rejection is not evidence of an effect.

Bananas reach the shift-null resolution floor on both prespecified statistics
(`p=1/65=0.0154`) with a log variance ratio of `0.5827` and log exceedance ratio of `0.9115`.
Their candidate-family BH q-value is `0.4615`. This is the clearest illustration of why an
individual extreme value must not be promoted when the registered family does not reject.

The run contains 22 usable warm RONI episodes, rather than the rough count of 17 used in the
v3 program's motivating arithmetic. That does not change the result or the resolution floor;
it records the distinction between the old +12-month endpoint sample and this monthly-window
estimand instead of silently carrying the rough count into the receipt.

## W2 is suggestive only at the family level

The nine-commodity signed cold-phase family has shift-null `p=0.0308`. No commodity survives
BH, the three negative controls do not reject under the Bonferroni-corrected worst-case-tail
rule, and the result cannot clear the frozen program threshold of `0.0125`. The receipt therefore
labels it `exploratory_within_stage_only`.

Palm oil and RSS3 rubber each reach the individual shift floor (`p=0.0154`), with BH
`q=0.0692`. Rubber's signed tail and mean statistics agree. Palm oil's registered negative tail
effect is accompanied by a negative *signed* mean difference, meaning that its mean moves
against the registered hypothesis; the strict `statistics_agree` diagnostic is therefore
false. Neither commodity is a corrected finding. Coal, the commodity whose mechanism
motivated W2, has the registered positive signs in both endpoints but
`p=0.1538`.

The W2 result is worth carrying forward because the family timing is unusual at the ordinary
5% level and the controls remain quiet. It is not a program-level finding, and the shift-null
resolution bound means the price-side statistic cannot become one under the frozen design.
The useful next evidence for W2 is the independently measured rainfall-throughput-price chain,
not another price specification.

## W5 resolves the question by showing that the split is underpowered

The peak-month classification yields six Eastern-Pacific and thirteen Central-Pacific episodes
with usable candidate outcomes. The family bootstrap gives `p=0.6050`, no candidate survives
BH, and all 30 candidates have an observed contrast below their own minimum detectable contrast.
The median minimum detectable contrast is `0.0887` in seasonal-adjusted log-return units.

Only `0.6818` of episode labels agree when the classification reference changes from the peak
month to the episode mean. This sensitivity and the power result together retire flavour as an
explanation that this dataset can resolve: the output is not evidence that EP and CP effects are
equal, but a measured statement that the available episodes cannot distinguish them.

The classification-only Niño 3/Niño 4 inputs were acquired from the URLs frozen in
`config/flavour_sources.yaml` on `2026-09-08`. Their processed panel is bound into the W5 receipt
with SHA-256 `0bd60bed212093704427fecd71b6245d1054624ea5555109008bf079d4d317fa`.

## Publication consequence

Both receipts and their compact result tables are now part of the auditable publication bundle,
and `reports/current_results.md` renders their metrics and interpretation directly from the
hash-verified receipts. This is reporting integration only: it adds no gate and changes no
analysis result.
