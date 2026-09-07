# Task 8 report: antisymmetric supervised models and OOF temperature

## Status

Implemented Task 8 only.

- Added B0 ranking logistic on `rank_diff` and the antisymmetric
  `rank_missing_b - rank_missing_a` indicator.
- Added B2 L2 logistic with `C=1.0`, train-fitted median imputation,
  train-fitted non-centering scale, and `fit_intercept=False`.
- B2 accepts only difference columns and the Task 7 symmetry-safe
  `elo_x_bestof` interaction; raw `is_challenger` is rejected.
- Added yearly prequential OOF scoring, intercept-free temperature fitting,
  and score-to-probability temperature application.
- Added and locked the compatible scikit-learn dependency.

## TDD evidence

The focused tests initially failed during collection because
`tml.models.supervised` and `tml.models.calibration` did not exist. After
implementation:

```text
13 passed in 5.44s
```

The complete suite then passed:

```text
60 passed in 1.50s
```

`python -m compileall -q src tests`, `git diff --check`, and IDE diagnostics
also completed without errors.

## Critical regressions covered

- Negating every B2 difference feature complements probability to float
  tolerance.
- Swapping B0 rank slots, including missing-rank flags, complements
  probability.
- B0 and B2 expose a zero intercept and are always fit without one.
- Prediction rows cannot refit or alter median imputation and scaling.
- Raw Challenger context cannot become a free directional offset.
- Temperature scaling preserves antisymmetry.
- OOF year folds fit only on strictly earlier years; the initial year remains
  unscored and is excluded from temperature fitting.

## Definition of Done

- Focused and full test suites pass without warnings.
- Calibration has no Platt intercept and consumes only supplied OOF scores.
- No Task 9 or later work was started.

## Concerns

Median imputation is required by the brief. For a missing difference value,
the training median is not mathematically guaranteed to negate under a
hypothetical swapped missing row. Observed numeric differences remain exactly
antisymmetric because scaling does not center; B0 handles missingness with an
explicit antisymmetric indicator.
