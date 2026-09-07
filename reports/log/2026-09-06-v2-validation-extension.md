# Validation v2: selected-family timing, forecasting and strict new-input gates

- Type: `decision`, `finding`, `extension`
- Discovery snapshot: `2026-09-06`
- Status: code and synthetic validation complete; supported stages run on the real snapshot

## Decision

The broad retrospective study remains unchanged. `config/validation_v2.yaml` freezes coconut
oil, palm oil and RSS3 rubber as outcome-informed discovery survivors before any new validation
inputs are inspected. This is a post-discovery validation contract, not a claim that the three
commodities were selected prospectively.

The contract adds two stages supported by the existing snapshot and two strict interfaces whose
external inputs are not present. The supported stages are an expanding-window forecast benchmark
and a 96-cell selected-family whole-year timing null. The pending interfaces are a regional
crop-calendar mechanism chain and licensed contract-level futures data. Pending layers create no
empirical result and cannot change a scorecard gate.

## Forecast finding

The benchmark compares a return-lag and calendar-month model with a nested RONI model at 3, 6 and
12 months. A training row is admitted only when its target endpoint is observable by the forecast
origin. Eight of nine cells improve RMSE. One paired calendar-year loss comparison is below 5%:
RSS3 rubber at 12 months. Its price-index strategy does not beat the long-only price-index proxy,
so this is predictive-error evidence rather than economic usefulness.

The stage is not genuine out-of-sample validation. It uses final revised RONI rather than archived
real-time vintages, World Bank indexes rather than investable contracts, and a configured cost
proxy without roll yield. The receipt structurally sets `genuine_out_of_sample: false` and
`tradability_claim_permitted: false`.

## Selected-family timing finding

The timing family crosses three commodities, two ENSO indexes, two anchors, four horizons and two
adjusted log-return outcomes: 96 cells per alignment. Against 60 circular whole-year shifts, both
the median absolute t and maximum absolute t statistics have finite-sample p=1/61, or 0.01639.
Positive-estimate share has p=4/61, or 0.06557.

This is stronger timing coherence than the earlier exposure-panel family, whose joint timing null
is not unusual. It does not independently validate the three commodities because their inclusion
was determined from the same discovery history. Its purpose is to calibrate the complete locked
v2 event-study family and provide a test that can be repeated unchanged on new inputs.

## Mechanism and tradability gates

The mechanism interface requires externally sourced crop calendars, lagged production weights,
regional weather, yield surprises and timestamped supply revisions. It estimates ENSO-to-weather,
weather-to-yield and revision-to-price links with cluster-robust inference. A supply observation
dated after its measured price is rejected.

The futures adapter requires contract settlements, volume, open interest, expiry and information
dates. It does not splice returns across rolls. Until licensed inputs and an explicit roll-yield
method are supplied, neither mechanism completion nor tradability is reported.

## Interpretation change

The scorecard no longer labels a non-rejection as `no_detected_association`. It distinguishes
`inconclusive_underpowered` from `inconclusive`; `evidence_against_material_effect` is available
only when a confidence interval lies inside a prespecified equivalence bound. This prevents low
power from being described as evidence of no material effect.
