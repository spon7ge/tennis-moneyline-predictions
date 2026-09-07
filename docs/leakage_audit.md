# Leakage audit

This document records how v1 prevents look-ahead bias and documents known information-set limitations. It maps design §3, §7–§10 to implemented checks.

## Estimand boundary

**Target:** `y_complete_win` = P(A wins | normal completion).

| Outcome | In labels? | Rationale |
|---------|------------|-----------|
| Normal completion | Yes | Defines the estimand |
| Retirement (`RET`) | No | Would shift toward book-settlement semantics |
| Walkover (`W/O`) | No | Match never starts |
| Default / abandoned | No | Not normal sporting completion |

Market no-vig probabilities may grade retirements differently — comparisons are **model–market probability gaps under mismatched estimands**, not proof of edge (see [model_card.md](model_card.md)).

## Orientation (not winner/loser)

Source CSVs are winner/loser ordered. Modeling never trains on that orientation.

1. Stable hash on `orientation_v1|match_id|sorted(player_ids)` assigns slots A/B
2. `y_complete_win = 1` iff A won
3. No mirrored duplicate rows in v1

**Audit:** `tests/data/test_orientation.py` — hash stability and independence from ID magnitude.

## Structural antisymmetry

Scores satisfy `s(A,B,c) = −s(B,A,c)` and `p̂ = logistic(s)`.

- Features enter as **differences** `x_A − x_B`
- Forbidden: free intercept, tour-level main effects that do not flip with A/B swap, asymmetric GBM probabilities without symmetrization
- Allowed calibration: temperature scaling `s' = s / T` (preserves antisymmetry)

**Audit:** `tests/models/test_symmetry.py`, `tests/models/test_supervised_symmetry.py` — `p̂(A,B) + p̂(B,A) ≈ 1`.

## Pre-tournament prediction regime

When only `tourney_date` exists (Sackmann files):

| Concept | v1 definition |
|---------|---------------|
| `prediction_regime` | `pre_tournament` |
| `prediction_cutoff` | Day before `tourney_date` (`prediction_cutoff()` in `tml.features.builders`) |
| Update batch | Entire tournament after all matches in that tournament are scored |
| Within-tournament features | **None** — no rest/load from same event without verified match timestamps |

**Rules enforced**

1. Features and predictions for all matches in tournament T use state frozen at T’s cutoff
2. Elo/form updates apply only after all predictions for T are recorded
3. Match calendar dates are **not** inferred from round labels in v1
4. Each persisted row stores `prediction_regime`, `prediction_cutoff`, and frozen feature vector

**Audit:** `tests/features/test_builders.py` — cutoff and prior-tournament-only history; `tests/features/test_feature_store.py` — regime and immutability.

## Ranking availability

| Source | `available_at` policy |
|--------|----------------------|
| Match-embedded ranks | `tourney_date` — usable at pre-tournament cutoff |
| Ranking snapshots | Usable iff `rank_as_of_date <= tourney_date` |
| Conflicts | Prefer snapshot; flag conflict |
| Missing | `rank_missing=True`; no invented ranks |

**Audit:** `tests/features/test_ranking.py` — embedded rank timing and antisymmetric `rank_diff`.

## Frozen feature store

Supervised models (B0, B2) fit on **persisted** feature rows from each match’s original cutoff. `FitSupervised` must never reconstruct training features from current or end-of-window Elo state.

**Audit:** `tests/features/test_feature_store.py`, `tests/validation/test_prequential.py`.

## Prequential walk-forward

| Model | First scored year | Notes |
|-------|-------------------|-------|
| B1 Elo | 2000 | Pre-2000 burn-in only |
| B0 / B2 | 2005 | Rolling 5-year coefficient windows |
| Dev era | 2005–2018 | Model selection |
| Final era | 2019+ | Primary Δ log-loss vs B1, untouched until report |

Year-Y coefficients are frozen from prior Dec 31 fit. Elo state still advances tournament-by-tournament within Y for **future** cutoffs only.

## Elo K-by-level (not tour dummy)

Shared player pool; `k_atp` and `k_challenger` control update reliability. A Challenger intercept cancels in `R_A − R_B` and is excluded.

**Audit:** `tests/features/test_elo.py` — K selection by `tour_level`.

## Odds timestamps (forward path)

Stored prices are **observed**, not executable.

Closing benchmark requires:

- Both sides in the same atomic record
- `last_update <= cutoff − 5 minutes`
- `collected_at <= cutoff − 5 minutes`
- `cutoff − last_update <= 30 minutes`

Schedule history is preserved; retrospective rewrites from later schedule versions alone are invalid.

**Audit:** `tests/odds/test_closing.py`, `tests/odds/test_storage.py`.

## Cross-regime comparisons

| Regime | Use |
|--------|-----|
| `pre_tournament` | Primary historical backtest |
| `pre_match` | Optional slice with verified match timestamp |
| `forward_live_info` | Forward path with intra-event info — must not compare 1:1 to primary backtest without labeling |

## Residual risks

| Risk | Mitigation |
|------|------------|
| `tourney_date` ≠ match date | Pre-tournament regime; no fake match-level claims |
| Post-match rank revisions in embedded fields | Manual QC on sample tournaments; conflict flags |
| CSV schema drift | Manifest hash `dataset_snapshot_id` |
| ID collisions | Versioned identity map + conflict queue |
