# Current results

This note records the latest locally validated analysis state. It is a compact interpretation of
the hash-linked tables, not a substitute for the detailed data dictionary or a claim of causal
identification.

It is **overwritten on every run** and describes only the snapshot named below. The dated,
append-only record of decisions, findings and retired specifications lives in
[`reports/log/`](log/README.md); open [the 2026-09-06 design
review](log/2026-09-06-design-review.md) for known limitations of the numbers on this page.

## Run identity

- Data snapshot: `2026-09-05`
- Regenerated: `2026-09-06`
- Git commit: `4b27862`
- Research configuration SHA-256:
  `22c62895e9d30858c6d45ba159626b972e72817fe9d60ea5e7bd922d9bb35c5a`
- Runtime: Python 3.13.1, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.1,
  statsmodels 0.15.0
- Primary bootstrap: 10,000 whole-episode draws, studentized two-sided test
- Panel bootstrap: 2,000 calendar-year block draws

Every stage from raw-event construction through the panel completed against the existing real-data
snapshots, and each stage accepted its upstream hashes. The generated artifacts remain under
`tables/2026-09-05/` and are intentionally excluded from version control.

**Three diagnostics postdate this run and have not been applied to it**: the
minimum-detectable-effect stage (`make power`), the placebo era-balance check, and the panel's
resampling-block sensitivity. All three can change how the numbers below should be read -- the
first by separating null results from underpowered ones, the second by explaining the
negative-control failure, the third by widening the panel interval -- and none of them has been
run. See [the design review](log/2026-09-06-design-review.md) for what each is expected to
show.

## Evidence summary

| Stage | Current result | Interpretation |
|---|---|---|
| Primary event study | 10/30 BH FDR; 6/30 BY FDR; 5/30 Westfall-Young FWER | Several non-null associations remain under stricter inference. |
| Neutral-date placebo | 8 candidates pass bootstrap, sign and placebo gates | Fewer than under the superseded centred test. |
| External macro controls | 9 candidates pass the macro gates | Dollar, CPI and activity controls do not remove the broad pattern. |
| Leave one episode out | 7 candidates survive every deletion | The strongest associations are not driven by one episode. |
| Timing/index grid | Coconut oil, palm oil and RSS3 rubber pass all four cells | These are the remaining robust event-study candidates. |
| Warm versus cold | Only three survivor/cell contrasts pass FDR | Most of the apparent response is not reliably warm-phase-specific. |
| Financial controls | 3/24 precious-metal cells reject, all Gold | Controls narrow but do not fully explain the falsification failure. |
| Palm-oil mechanism | Weather and yield links pass; production to price fails | The physical chain remains incomplete. |
| Exposure panel | Primary p=0.013; mapping permutation p=0.035; 0/20 cells pass grid FDR | The mapping is informative, but there is no family-wide panel result. |

## Surviving associations

The frozen primary event study has ten BH discoveries: Australian coal, coconut oil, Arabica and
Robusta coffee, Malaysian logs, European and US natural gas, palm oil, Thai 5% rice and RSS3
rubber. Five survive Westfall-Young FWER: Australian coal, coconut oil, European natural gas, palm
oil and RSS3 rubber.

After the macro, leave-one-episode-out, timing and index gates are combined, only coconut oil, palm
oil and RSS3 rubber remain. All three have positive mean month-12 returns in every timing/index
cell. They are robust historical associations, not validated ENSO mechanisms.

## Panel result

The primary exposure-weighted panel uses RONI lagged six months, commodity and calendar-month fixed
effects, and seasonal-adjusted log returns. Its exposure coefficient is `0.006238`, with a
year-block 95% interval of `[0.000975, 0.011100]` and studentized `p=0.01299`. The corresponding
negative-control interaction has `p=0.12694`.

The observed coefficient is at the 98.3rd percentile of 2,000 assignments produced by shuffling
the same exposure weights across candidate commodities while keeping controls fixed at zero. The
two-sided mapping-permutation p-value is `0.03498`. This rejects arbitrary assignment of the current
weight multiset in the primary cell, but it cannot undo the fact that the weights were authored
after the event-study outcomes were known.

That primary result does not generalize across the specification family. No exposure coefficient
survives BH correction over the 20-cell grid, all uniform candidate-versus-control cells are null,
and two exposure-weighted control cells reject at raw 5%. Because the exposure weights were written
after the event-study results were known, the panel is exploratory evidence only.

## Bottom line

The recalibrated analysis is less permissive than the original centred bootstrap, as intended. It
leaves a small, coherent set of commodity associations, but both the phase-specificity tests and
the panel prevent an ENSO-specific interpretation. The highest-value next research step is to
construct exposure weights from external production geography and pre-estimated climate
teleconnections, freeze them, and rerun the panel without outcome-informed judgment.
