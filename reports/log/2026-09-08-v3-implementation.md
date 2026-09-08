# Implemented: the v3 findings program, five workstreams, none yet run on real data

- Type: `decision`, `finding`
- Date: 2026-09-08
- Follows: [the 2026-09-08 v3 findings program](2026-09-08-v3-findings-program.md)
- Git commit at authoring: `f2f0675`
- Status: implemented and unit-tested. **No stage has been run against a real-data snapshot.**
  The container this was written in has no snapshot, and rebuilding one would change the
  snapshot date and break the hash chain `current_results.md` reports against.

## What exists now

| Workstream | Contract | Modules | Runs on existing data? |
|---|---|---|---|
| W1 dispersion | `config/dispersion.yaml` | `dispersion.py`, `dispersion_analysis.py` | yes |
| W2 cold phase | `config/cold_phase.yaml`, `config/cold_phase_mechanism.yaml` | `cold_phase.py`, `cold_phase_analysis.py`, `disruption_chain.py` | price side yes; chain gated |
| W3 forecast news | `config/forecast_news.yaml`, `config/forecast_news_sources.yaml` | `forecast_news.py`, `news_analysis.py` | no — archive not acquired |
| W4 term structure | `config/term_structure.yaml` | `term_structure.py` | no — licensed inputs absent |
| W5 flavour | `config/flavour.yaml`, `config/flavour_sources.yaml` | `flavour.py`, `flavour_data.py`, `flavour_analysis.py` | yes, after one download |

`config/findings_v3.yaml` is the program-level register. It totals four primary tests, so the
program threshold is 0.05/4 = 0.0125, and every v3 receipt records its own p-value against it.

## Findings from building it

Three things were learned that were not in the plan, and two of them are calibration facts about
the designs themselves rather than about ENSO.

### The shift null is well calibrated against volatility clustering

This was the open question that decided whether W1 was worth having at all. A variance test on
commodity returns is exposed to volatility clustering: any null that treats months as
exchangeable finds "significant" dispersion differences everywhere, and the negative controls
would light up first.

Measured over 400 replications of a stationary process whose log variance follows an AR(1) with
coefficient 0.97 and no relationship to the episode mask, the stage rejects at **6.25%** against
a nominal 5%, with mean p-value 0.497 and median 0.492. The attainable levels are multiples of
1/65, so the largest level at or below 5% is 3/65 = 0.046; 6.25% over 400 draws is within Monte
Carlo error of it. The p-values are close to uniform. That is what makes the variance endpoint
usable here, and it is the single result most worth quoting from this commit.

### The worst-case control rule was rejecting at twice its nominal rate

W2 evaluates negative controls in whichever tail favours rejection, because they carry no
registered direction. As first written that took the smaller of two one-sided p-values and
compared it with alpha, which is the minimum of two tests and is not a p-value at the nominal
level. A pure-noise control voided the stage roughly 10% of the time; with eight controls it
would have voided essentially always, and the falsification would have been worthless — it would
have fired on noise every run and told a reader nothing.

It is now Bonferroni-doubled across the two tails, and a size test holds the rejection rate at
nominal over 200 replications of a pure-noise control. The correction is recorded in the
contract as `control_tail_correction: bonferroni_two_tails` rather than left in the code.

This was found because a test fixture happened to draw a control excursion large enough to void
a stage that should have passed. It is worth recording that the bug was a *falsification firing
too readily*, which is the direction that looks conservative and is not: a check that always
fires carries no information.

### The shift null assumes ENSO recurrence is irregular

If episodes recurred at a fixed period of p years, every whole-year shift that is a multiple of p
would map the mask onto itself, those draws would tie with the observed statistic, and the null
would lose that fraction of its resolution. The first test fixtures used exactly periodic onsets
and the null lost 12 of its 64 draws to exact ties, which is how this surfaced.

Real ENSO recurrence is two to seven years and irregular, so this does not bite in practice. It
is now stated where the shifts are generated, because it is an assumption the reference
distribution depends on and nothing in the code enforces it. A near-periodic treatment would
need a different null.

## The resolution floor, which constrains what W1 and W2 can ever conclude

A shift-based reference distribution with S shifts cannot produce a p-value below 1/(S+1).
Whole-year shifts are bounded by the sample length in years, so with 65 years the floor is
1/65 = 0.0154. The program threshold is 0.0125. **The floor is above the threshold**, which means
neither W1 nor W2 can reach a program-level finding on its shift statistic alone, however extreme
the observed value looks.

This is a property of the design and not of the data, so it is reported in every receipt as
`resolution_limited` rather than discovered afterwards. It also settles which workstream carries
the program: W3 has hundreds of monthly observations and a bootstrap that can resolve past the
threshold, W1 and W2 cannot. The plan already ranked W3 as the most valuable design; this is an
independent reason for the same ranking.

## What has not been done

- **Nothing has been run on real data.** Running W1, W2 and W5 against the `2026-09-06` snapshot
  is the next step and all three read hashes that the existing artifacts already satisfy.
- **W3's archive is not acquired.** `config/forecast_news_sources.yaml` deliberately carries no
  `sources:` block: the retrieval URL, layout and first available issuance of the CPC/IRI
  probabilistic archive must be verified against the live source first. The stage refuses to run
  without a hash-verified archive that certifies its issuance dates as published, because a
  back-filled probability would make every coefficient in W3 a look-ahead artifact.
- **W2's physical chain has no inputs.** Queensland rainfall, port throughput and dated export
  volumes are separate work. `disruption_chain.py` validates and estimates but reports nothing,
  and a partial chain is reported incomplete rather than as partial support.
- **W4 produces no number** and contributes zero primary tests, by construction.
- **`current_results.md` is untouched.** No v3 stage has a receipt, so there is nothing for the
  report generator to quote, and the existing hash chain is undisturbed.

## Standing commitments

The plan recorded four things that would make this program wrong. Two of them are now enforced
by code rather than by intention: W1 voids itself if its negative controls reject, and W3 voids
itself if its lead placebo rejects. The other two remain commitments a reader has to check
against the git history: that W2's signed hypotheses are not revised after seeing a result, and
that `config/findings_v3.yaml` is not edited after any stage runs. Both are visible as
configuration-hash changes in the receipts, which is the whole reason the hashes are recorded.
