# Research log

An append-only record of what was decided, tried, found and rejected.

## Why this exists

Three documents in this repository answer three different questions, and they rot
if the distinction is not held:

| File | Question | Lifecycle |
|---|---|---|
| `README.md` | What is this project and how do I run it? | Rewritten freely |
| `reports/current_results.md` | What does the *latest* run say? | **Overwritten** each run |
| `reports/log/*.md` | What did we learn, and when, and why did we change course? | **Append-only** |

`current_results.md` is a snapshot. Every time the pipeline is rerun it is
replaced, so the history of what was tried, what failed, and what argument
retired a specification survives only in commit messages. That is not good
enough for a study whose central claim is that its own negative results are
trustworthy: the credibility of "we did not go looking for this answer" rests
entirely on a dated record of what was specified before the numbers arrived.

## Conventions

- One file per entry, named `YYYY-MM-DD-slug.md`, dated when written.
- **Never edit an entry after committing it.** Corrections go in a new entry
  that links back. A superseded entry gets one line at the top pointing forward;
  that is the only permitted edit.
- State the git commit and, where a run is involved, the data snapshot date and
  the configuration SHA-256, so an entry can be tied to reproducible artifacts.
- Entry types, marked in the front matter line:
  - `decision` — a specification was frozen, changed or retired, and why.
  - `finding` — something was learned from a run or from reading the code.
  - `review` — a systematic pass over the design or the codebase.
  - `negative` — something was tried and did not work. These are the most
    valuable entries and the easiest to leave unwritten.
- Prefer being specific and falsifiable over being readable. An entry that
  cannot be checked against an artifact is a diary, not a log.

## Relationship to preregistration

Frozen contracts live in `config/` (`commodities.yaml` for the primary endpoint,
`panel.yaml` for the panel, `exposure_v2.yaml` for its external weights, and
`dose_response.yaml` for the secondary amplitude diagnostic), because a machine reads them and the
build refuses to run when they drift. The log records the *reasoning* around
those freezes, which YAML cannot carry: what alternative was considered, what
would have to be true for the choice to be wrong, and what was already known
about the outcomes at the moment the choice was made.

That last point is the one that matters most here. The retired v1 panel weights
were written after the event-study results were known; v2 records immutable
external source URLs, hashes, transformations and exclusions instead. Any future
weight set, endpoint or gate needs the same disclosure, and this is where the
argument for it lives.
