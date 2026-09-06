# Warm-episode amplitude dose response

- Type: `decision`, `finding`, `negative`
- Date: 2026-09-06
- Base commit: `c251dc1`
- Data snapshot: `2026-09-06`
- Status: implemented, run on real data, and receipt-verified

## Decision

The binary warm-event endpoint remains frozen. This stage asks a secondary question using the same
month-12 seasonal-adjusted returns: does the response change with the warm episode's peak RONI?
Peak amplitudes are standardized across the 17 warm episodes, so each slope is cumulative log return
per one standard deviation of peak RONI.

The estimator is OLS with an HC3 standard error. Inference does not use its asymptotic p-value.
Instead, each of 10,000 replicates permutes the complete episode-amplitude mapping once and applies
that same mapping to every commodity. This preserves the observed return histories, missingness and
cross-commodity dependence. The 30 candidates form one BH family and one maximum-absolute-t family;
the three negative controls remain diagnostics. The scope is `exploratory_secondary` and no primary
gate changes.

## Finding

Ten of 30 candidate slopes are positive. Two reject at raw 5%: sorghum is negative, with a slope of
-0.02297 per peak-RONI standard deviation and permutation p=0.00710; coconut oil is positive, with a
slope of 0.04154 and p=0.02310. Neither survives BH, no candidate survives max-|t| family correction,
and no negative control rejects at raw 5%.

There is therefore no family-level evidence that warm-event amplitude explains the frozen commodity
endpoint. The strongest raw relationship also runs in the negative direction, which is compatible
with heterogeneous mechanisms but not with a generic stronger-El-Nino-means-higher-prices story.

## Limits

Permutation treats the 17 episode-amplitude labels as exchangeable. It preserves commodity returns
exactly but not a possible relationship between amplitude and historical era. The stage is therefore
a useful dose-response falsification, not a substitute for a time-aware model or an out-of-sample
test. ENSO flavour remains unresolved; splitting 17 episodes into Eastern- and Central-Pacific groups
would be too underpowered to promote beyond descriptive work.
