# Task 9 report: prequential walk-forward driver

## Status

Implemented Task 9 only.

- Added an injectable `PrequentialConfig` with the required production defaults
  and shortened-year/test storage overrides.
- Added tournament ordering by `(tourney_date, tourney_id)`, pre-tournament
  feature freezing and persistence, B1 scoring, and post-tournament Elo updates.
- Added once-per-year B0/B2 fitting from persisted feature rows whose original
  tournament dates fall in the prior rolling window.
- Added the required prediction output schema and explicit `NaN` values before
  each model's configured evaluation start year.
- Added optional fixed temperature application for B0 and B2 predictions.

## TDD evidence

The focused test first failed during collection because
`tml.validation.prequential` did not exist. After implementation:

```text
3 passed in 1.36s
```

The complete suite then passed:

```text
63 passed in 1.45s
```

## Critical regressions covered

- Shuffled input is processed chronologically by tournament.
- Every match in a tournament receives B1 from the same pre-tournament state,
  so earlier results in that tournament cannot affect later predictions.
- Frozen persisted rows retain their original cutoff Elo values as state
  advances.
- B0/B2 are unavailable before `supervised_from`; B1 independently starts at
  `elo_from`.
- Predictions contain exactly the required identifiers, probabilities,
  outcome, context, regime, and snapshot columns.

## Definition of Done

- No secrets, external input, SQL, HTML, shell construction, handlers, or UI
  are introduced.
- Driver boundary inputs and configuration are validated.
- Errors propagate with actionable messages; none are silently swallowed.
- Focused and full pytest suites, diff checks, and IDE diagnostics pass.
- No Task 10 or later work was started.

## Concerns

- The default feature-store path is append-only by design. Re-running the same
  snapshot at that path requires the caller to choose a fresh path or archive
  the prior store; silent overwrite would violate frozen-row guarantees.
- Supervised fitting requires both outcome classes in the configured prior-year
  window, as required by the existing B0/B2 estimators.
