# W3 forecast-news run: the apparent response fails its own timing falsification

- Type: `finding`, `negative`
- Date: 2026-09-08
- Commodity data snapshot: `2026-09-06`
- Forecast archive snapshot: `2026-09-08`
- Follows: [the v3 implementation](2026-09-08-v3-implementation.md)
- Status: run completed; `void_lead_placebo_rejected`

## The archive is real, issue-dated, and narrower than the original hope

The live IRI archive exposes 136 consecutive monthly Quick Look pages with machine-readable
official CPC/IRI early-month consensus probability tables from January 2014 through April 2025.
From May 2025 onward those probabilities are images rather than HTML tables, and the current IRI
page says forecast data are no longer distributed. The acquisition therefore stops at the last
machine-readable official table. It does not transcribe chart pixels, infer probabilities from
model plumes, or use a current file as historical data.

The raw snapshot retains and hashes every HTML page. The processed archive has 1,223
issuance-target rows and 136 issue months. In 135 months the CPC section itself prints a valid
publication date. January 2020 carries a stale 2019 year in that section; for that page alone the
parser uses the independently printed January 21, 2020 Quick Look publication date, which is a
conservative explicit availability date. The February 2018 page's literal `Februrary` typo is
normalized. A few tables retain a just-completed season; those rows are excluded because the
frozen contract forbids a target month before issuance.

The processed probability file SHA-256 is
`358e4252e17a24283f0f0fcc34df82f76bb41235c301ba8e8e1a3c2a9d65bbd8`. It yields 135
successive-issuance revisions at the primary six-month lead, above the frozen minimum of 120.
The three-month sensitivity also yields 135 revisions. The prespecified nine-month sensitivity
is unavailable: the official tables expose at most eight forward target centres. The receipt
states that structurally rather than silently presenting an empty sensitivity as a result.

## The primary pattern is not interpretable as news

The six-month candidate-family maximum has wild-cluster-bootstrap `p=0.0575`, above both 5% and
the v3 program threshold of `0.0125`. Soybeans have coefficient `-0.1811`, bootstrap `p=0.0007`
and BH `q=0.0196`, the only candidate to survive within-family BH. Platinum, a negative control,
rejects at `p=0.0086`.

More importantly, four lead-placebo cells reject: Coconut oil (`p=0.0053`), DAP (`p=0.0181`),
Soybeans (`p=0.0226`) and Wheat (`p=0.0413`). Gold is also non-zero in the lead-placebo window
at `p=0.0368`. Seven lag-placebo cells reject. These are exactly the failures the frozen contract
said would void a contemporaneous response: the same revision series appears related to returns
before the forecast is published and after the news should already have been absorbed.

The correct result is therefore not "markets price ENSO forecasts." It is that this monthly
forecast-revision design does not isolate news in these data. The strongest apparent primary
response, Soybeans, also fails the lead placebo and cannot be rescued by its BH q-value. W3 does
not distinguish market efficiency from no material ENSO effect under this implementation.

No contract, sign, gate, or multiplicity rule was changed after observing this run.
