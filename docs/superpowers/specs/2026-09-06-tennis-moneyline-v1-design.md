# Tennis Moneyline Predictions — v1 Design

**Date:** 2026-09-06  
**Status:** Approved for implementation  
**Parent:** `project_spec.md` (full research vision)  
**Approach:** Thin vertical slice (Approach B)

## 1. Goal

Build a reproducible Python research package that estimates **conditional sporting-strength win probabilities** for professional tennis (ATP + Challenger), with explicit parameter uncertainty—not a tip sheet or profit guarantee.

v1 prioritizes: versioned canonical data, leakage-safe as-of features, calibrated baselines, prequential walk-forward validation under a **pre-tournament information set**, and a forward-looking odds snapshot pipeline via ParlayAPI.

## 2. Estimand (required)

### 2.1 Target

The v1 label and prediction target is:

\[
p = P(\text{player A beats player B} \mid \text{match starts and completes normally})
\]

**Not** unconditional pre-match win probability, and **not** sportsbook settlement probability.

| Outcome type | In label set? | Rationale |
|--------------|---------------|-----------|
| Normal completion | Yes | Defines the estimand |
| Retirement (`RET`) | **No** | Would change the estimand toward settlement-style outcomes |
| Walkover (`W/O`) | **No** | Match never starts |
| Default / abandoned | **No** | Not a normal sporting completion |

Rename in code and docs: `y_complete_win`, never `market_settlement_win`.

### 2.2 Market mismatch (documented, not “fixed” in v1)

Many books grade some retirements as wins for the leading/opponent player. Therefore:

- Model \(p\) is a **sporting-strength** probability conditional on completion.
- No-vig market probabilities estimate a **different** estimand when retirement grading differs.
- Forward notebooks may still compare model \(p\) to market-implied \(q\), but must label the comparison as **model–market probability gap under mismatched estimands**, not as proof of edge.
- True **CLV** (see §15) requires an explicit decision-time **observed** price and a later closing benchmark; it is separate from the research estimand.

Post-v1 option: model \(P(\text{completes})\) and settlement maps per book, or retrain on settlement labels.

## 3. Locked decisions

| Decision | Choice |
|----------|--------|
| Tours | ATP main tour + ATP Challenger only |
| Estimand | Conditional on normal completion (§2) |
| Historical information set | **Pre-tournament** when only `tourney_date` exists (§8) |
| Walkovers / defaults / retirements | Excluded from labels; retained in storage with flags |
| Metric years | Elo/B1 scored from **2000+**; B0/B2 scored from **2005+** (§10.2) |
| Pre-2000 data | Elo **burn-in only**; not in reported metrics; not used to fit B0/B2 |
| Validation | Prequential hybrid (§10): expanding Elo by tournament batch; rolling 5-year supervised coeffs |
| Model selection | Development era **2005–2018**; untouched **2019+** for final report (§10.3) |
| Score symmetry | **Structural** antisymmetric logit score (§7) |
| Tour-level Elo | Shared player pool; **different K by level** (§11) |
| Odds | Forward-only ParlayAPI; no deep historical odds dependency |
| Deliverable | Installable Python package + notebooks |
| Out of v1 training labels | WTA, ITF, quali |

## 4. Non-goals (explicit)

- Exact-set / total-games / point-level distributions
- Kelly / bankroll product
- Multi-year historical odds backtest
- Inferring match calendar dates from round labels (unless separately validated post-v1)
- Within-tournament rest/load features without verified match timestamps
- Calling API prices “executable” without verified limits/fills
- Treating model–market gap as CLV without decision-time observed prices
- Claiming profitability or certainty
- Unconditional or book-settlement win probability as the v1 estimand

## 5. Architecture

```
tml-data/*.csv  ──►  tml.data (ingest + manifest hash)  ──►  canonical tables
                                                            │
                                                            ▼
                                                  modeling table (A/B oriented)
                                                            │
                              feature rows frozen at prediction_cutoff
                                                            │
                         ┌──────────────────────────────────┼────────────────────┐
                         ▼                                  ▼                    ▼
                   Elo (expanding,                 supervised B0/B2         odds snapshots
                    tournament-batched)            (frozen feature rows)     (ParlayAPI)
                         │                                  │                    │
                         └──────────────────┬───────────────┘                    │
                                            ▼                                    ▼
                                    tml.validation (prequential)      gap vs CLV (distinct)
                                            │
                                            ▼
                                    notebooks + reports + viz
```

### Package layout

```
tml/
  data/         # ingest, IDs, identity maps, quality flags, manifest hashing
  features/     # as-of builders; persist frozen feature snapshots
  models/       # antisymmetric B0/B1/B2 (+ optional B2b/B3 with logit symmetrization)
  validation/   # prequential loop, paired metrics, out-of-time calibration
  odds/         # ParlayAPI client, atomic two-sided quotes, closing rules
  viz/          # intervals on p; joint ability draws from pipeline bootstrap
notebooks/
  01_ingest_and_qc.ipynb
  02_features_and_leakage_audit.ipynb
  03_baselines.ipynb
  04_walkforward_validation.ipynb
  05_uncertainty_plots.ipynb
  06_forward_odds.ipynb
docs/
  sources.md
  schema.md
  leakage_audit.md
  model_card.md
```

## 6. Data

### 6.1 Sources

| Source | Role | Notes |
|--------|------|-------|
| Local `tml-data/` | Results, embedded ranks, serve stats | ATP `YYYY.csv`; Challenger `YYYY_challenger.csv`; ranking CSVs when present |
| ParlayAPI | Forward moneylines | `tennis_atp`; confirm Challenger via `GET /v1/sports` before treating Challenger odds as a deliverable |
| ParlayAPI historical | Not required for v1 | Tier windows too short for 2000+ market backtests |

Before treating the odds archive as committed: document licensing, retention rights, rate limits, and Challenger availability in `docs/sources.md`.

### 6.2 Reproducibility / versioning

Each ingest run records a **sorted manifest**: one row per input file with `relative_path`, `size_bytes`, `sha256`. Then:

- `dataset_snapshot_id` = SHA-256 of the canonical serialized manifest (not raw byte concatenation of CSVs)
- `ingested_at` (wall clock)
- **Sapling** revision id and **dirty-worktree** boolean when available (do not assume git)

`ingested_at` alone is insufficient: experiments pin `dataset_snapshot_id`.

### 6.3 Player identity

- Versioned `player_identity_map` (map version id + effective dates)
- Rules for missing IDs, colliding IDs, and alias merges
- Unresolved collisions → exclude from modeling with a quality flag until resolved
- Odds name matching writes to the same map with review queue

### 6.4 Canonical entities

Stable IDs: `player`, `tournament`, `match`, `tour_level` (`atp` | `challenger`), surface, indoor/outdoor.

Match grain includes: tournament fields, `match_id`, round, `best_of`, raw winner/loser columns, score, minutes, serve columns, completion flags, lineage, and time fields in §8.

### 6.5 Completion flags

Derive `completion_status ∈ {completed, retirement, walkover, default, abandoned, unknown}` from score text. Only `completed` enters modeling labels.

## 7. Orientation and structural symmetry

Source CSVs are winner/loser ordered. **Never** train on that orientation.

### 7.1 Orientation

For each eligible completed match, assign slots A and B with a **stable hash** independent of outcome and of ID magnitude:

```
key = hash("orientation_v1" || match_id || player_id_1 || player_id_2)
# player_id_1, player_id_2 sorted only for hash input stability, NOT for slot assignment
if key odd: A = player_u, B = player_v
else:       A = player_v, B = player_u
```

Do **not** use `min(player_id)` (may correlate with cohort / ID assignment era).

Label \(y = 1\) if A won, else 0. **No mirrored duplicate rows** in training (v1).

### 7.2 Antisymmetric score (required)

Symmetry is **structural**, not merely tested:

\[
s(A,B,c) = -s(B,A,c), \qquad \hat{p}(A,B,c) = \operatorname{logistic}(s(A,B,c))
\]

where \(c\) is shared match context (surface, `best_of`, round, tour level, …).

Implications:

- Use **difference features** \(x_A - x_B\) (and antisymmetric interactions). Context \(c\) may enter only through terms that preserve \(s \mapsto -s\) when A/B swap (e.g. multiply differences by functions of \(c\); no additive directional intercept).
- **Forbidden:** free intercept that breaks swap symmetry; tournament/level main effects that do not flip with A/B; ordinary Platt scaling with nonzero intercept on \(\hat{p}\); raw GBM probabilities without symmetrization.
- **Allowed calibration:** temperature scaling \(s' = s / T\) (preserves antisymmetry). Diagnostic calibration intercept/slope on an evaluation year are **report-only**, not applied to production scores unless fit out-of-time on antisymmetric \(s\) without breaking the constraint.
- **GBM / ensembles:** predict a score or log-odds, then enforce \(s \leftarrow (s(A,B) - s(B,A)) / 2\) (or train only on differences). Never average asymmetric probabilities without this step.
- CI unit tests still assert \(\hat{p}(A,B) + \hat{p}(B,A) = 1\) within float tolerance—necessary but not sufficient without the structural form.

## 8. Time semantics: `tourney_date` and forecasting regime

### 8.1 What `tourney_date` is

In Jeff Sackmann–style files, `tourney_date` is generally the **tournament start date**, not each match’s calendar date. Batching on `tourney_date` alone therefore groups an entire tournament (and often multiple events starting that week)—**not** same-calendar-day matches.

Calling that “match-level prequential evaluation” would be misleading.

### 8.2 Historical prediction origin (v1 default)

When only `tourney_date` is available:

| Concept | Definition |
|---------|------------|
| `prediction_regime` | `pre_tournament` |
| `prediction_cutoff` | Instant just before tournament start: end of day prior to `tourney_date`, or `tourney_date` 00:00 with **no** results from that tournament yet |
| Update batch key | `(tourney_id)` — equivalently all matches sharing that tournament’s start date and id |
| Information set | All completed matches from **other** tournaments with cutoff strictly before this tournament’s start; plus rankings per §9 |

**Rules:**

1. Generate features and \(\hat{p}\) for **all** matches in a tournament using state frozen at `prediction_cutoff`.
2. After **all** matches in that tournament are predicted, update Elo/form using that tournament’s completed results (deterministic order by `match_id` for reproducibility only—not for within-tournament feature use).
3. **Do not** infer match dates from rounds unless a validated procedure is added later.
4. **Defer** within-tournament rest days and recent match-load features unless `match_timestamp_reliable = True`.
5. Persist each row’s `prediction_regime`, `prediction_cutoff`, and frozen feature vector (§10.5).

### 8.3 Reliable match timestamps (optional subset)

If a row has a verified actual start timestamp (e.g. from odds `commence_time` joined with high confidence):

- Set `match_timestamp_reliable = True`
- May use a finer cutoff for **that row only**
- Report metrics **separately** for this subset; do not mix silently with pre-tournament rows in the primary historical table without a slice label

### 8.4 Forward evaluation regimes

Forward ParlayAPI scoring often has true commence times and can use earlier-round results within an ongoing event. That is a **different forecasting regime**.

| Regime | When used | Label in reports |
|--------|-----------|------------------|
| `pre_tournament` | Historical Sackmann-only path | Primary backtest |
| `pre_match` | Verified match timestamp available | Optional slice |
| `forward_live_info` | Forward path intentionally using intra-event info | Must not be compared 1:1 to primary backtest without labeling |

To compare forward performance to historical backtests, either recreate the **pre-tournament** cutoff for those events or explicitly mark the comparison as cross-regime.

## 9. Ranking `available_at` policy

| Field | Meaning |
|-------|---------|
| `rank_value` | Rank or points |
| `rank_source` | `match_embedded` \| `ranking_snapshot` |
| `rank_as_of_date` | Official ranking date when known |
| `available_at` | Earliest date the value may enter features |
| `rank_staleness_days` | Days from `rank_as_of_date` to tournament start |
| `rank_missing` | Boolean |

**Policy:**

1. **Match-embedded pre-match ranks:** `available_at = tourney_date`; usable at the pre-tournament cutoff for that event. `rank_source = match_embedded`. Leakage audit must check these are not post-match revisions.
2. **Dated ranking snapshots:** usable iff `rank_as_of_date` is on or before the tournament start date under the pre-tournament cutoff rule (`available_at <= tourney_date`).
3. Prefer snapshot on conflict; set conflict flag.
4. Missing ranks: `rank_missing=True`; no invented ranks. Imputation only inside fold training on frozen features (§12).

## 10. Prequential walk-forward (executable)

### 10.1 Pseudocode

```
burn_in_end = 1999-12-31
elo_eval_start_year = 2000
supervised_eval_start_year = 2005
dev_era_years = 2005..2018
final_era_years = 2019..max_year

state = InitEloPriors()
WarmStartElo(state, tournaments with tourney_date <= burn_in_end)  # updates only

# Always persist frozen features at prediction time; never rebuild later from future state
feature_store = empty

For each tournament T in chronological order by (tourney_date, tourney_id):

  cutoff = PreTournamentCutoff(T)   # §8.2
  matches_T = completed modeling matches in T

  For each match m in matches_T:
    x = FeaturesAsOf(m, state_at=cutoff)   # no results from T; no within-T rest/load
    PersistFeatureRow(feature_store, m, cutoff, regime="pre_tournament", x)
    If year(T) >= elo_eval_start_year:
      p_B1[m] = EloWinProbAntisym(m, state_at=cutoff)

  # Predict supervised using coefs frozen for calendar year of T (see below)
  ...

  UpdateEloAndForm(state, matches_T)  # only after all predictions for T


For each calendar year Y in supervised_eval_start_year..max_year:

  train_end = Dec 31 of (Y-1)
  train_start = Jan 1 of (Y-5)

  # CRITICAL: Fit only on feature rows persisted when those matches were originally predicted
  train_rows = feature_store where tourney_date in [train_start, train_end]
  # Do NOT recompute x using state as of train_end

  oof_scores = PrequentialOutOfFoldScores(train_rows)  # for calibrator / ensemble
  coefs_Y, prep_Y, calib_Y, ens_Y = FitSupervisedOutOfTime(train_rows, oof_scores)

  For each tournament T with year(T) == Y:
    For each match m in T:
      x = feature_store[m]   # frozen at T's original cutoff
      p_B0[m], p_B2[m] = PredictAntisym(coefs_Y, prep_Y, calib_Y, ens_Y, x)
```

Year \(Y\) supervised **coefficients** are frozen from the prior Dec 31 fit. Elo/form state still advances tournament-by-tournament within the year for **future** tournaments’ cutoffs. Features for past matches remain those frozen at each match’s original cutoff.

### 10.2 First-fold resolution

| Model | First scored year | Notes |
|-------|-------------------|-------|
| B1 Elo | 2000 | State after prior tournaments only |
| B0 / B2 | 2005 | Fit on frozen feature rows from 2000–2004 |
| B2b / B3 | Selected on 2005–2018; **reported** on 2019+ |

No pre-2000 rows in B0/B2 fitting.

### 10.3 Nested / era-split model selection

1. **Development era (2005–2018):** choose {B2, B2b, B3} and tune Elo K / regularization using prequential results inside this era only.
2. **Final era (2019+):** freeze choices; report primary metrics once.
3. Applied **calibration and ensemble weights** must be fit from **prequential / out-of-fold scores inside the training period**, never from in-sample fitted probabilities on the same rows used to estimate coefficients.
4. Calibration intercept and slope on the evaluation year are **diagnostics only** (not applied back unless produced by the out-of-time procedure above, and only via temperature or other symmetry-preserving maps).

### 10.4 Frozen feature store (required)

`FitSupervised()` consumes **persisted** feature rows generated at each match’s original `prediction_cutoff`. It must **never** reconstruct historical training features from current or end-of-window Elo/form state.

Store at minimum: `match_id`, `prediction_cutoff`, `prediction_regime`, `dataset_snapshot_id`, feature vector, feature schema version.

## 11. Elo specification (B1) — tour level

Shared player pool across ATP and Challenger (Challenger informs players who later appear on main tour).

**v1 tour-level choice (single locked design):** different update reliability via **level-specific K-factors**:

- \(K_{\text{atp}}\) for main-tour completed matches  
- \(K_{\text{challenger}}\) for Challenger completed matches (typically smaller or tuned in dev era)  
- Same latent rating vector per player (hierarchical connection = players crossing tours)

A Challenger **main-effect dummy** on the match is useless for distinguishing A vs B. A constant additive offset applied to both players cancels in \(s(A,B)\). Those are **out**. Level may scale the **rating difference** (level-specific slope) only if introduced later via dev-era comparison; v1 ships K-by-level first.

| Element | v1 default |
|---------|------------|
| Surfaces | Separate hard / clay / grass ratings |
| Initialization | Prior \(\mu_0\) (e.g. 1500); new players at \(\mu_0\) |
| New-player | Higher effective K until \(n_{\min}\) matches |
| K-factor | \(K_{\text{atp}}\), \(K_{\text{challenger}}\) tuned in development era only |
| Inactivity | Optional regression to \(\mu_0\) after \(D\) days; if used, tune in dev era |
| Best-of-5 | Fixed BO5 adjustment on expectancy; freeze after dev era |
| Probability | \(\hat{p} = \operatorname{logistic}(s)\) with antisymmetric \(s \propto R_A - R_B\) (+ BO5 map) |
| Updates | After each **tournament** batch of completed matches (§8.2) |

## 12. Features (v1)

Built at `prediction_cutoff` under the row’s `prediction_regime`; A/B difference form.

1. Surface / overall Elo differentials (from state at cutoff)  
2. Ranking differentials + missing/staleness indicators  
3. Opponent-adjusted recency form (**prior tournaments only**)  
4. Opportunity-level serve/return rates from prior completed matches:
   - `1stIn / svpt`, `1stWon / 1stIn`, `2ndWon / (svpt - 1stIn)`
   - Breaks conceded `bpFaced - bpSaved` with shrinkage  
   - Not: treating `bpFaced` alone as hold probability  
5. Rest / load: **only** using prior-tournament end dates unless `match_timestamp_reliable`  
6. Age, hand, height, experience  
7. Context \(c\): tournament level, round, `best_of` (only via symmetry-safe interactions)  
8. H2H residual (not raw H2H):

\[
\mathrm{H2H\_resid} = \mathrm{shrink}\bigl(W^{\mathrm{hist}}_{AB} - \mathbb{E}[W_{AB} \mid \mathrm{Elo}]\bigr)
\]

**Missingness:** Within each fold, fit imputers/scalers on that fold’s **frozen** training feature rows only; apply to the prediction year’s frozen rows. Missingness indicators allowed.

## 13. Models

| Stage | Model | Role |
|-------|-------|------|
| B0 | Antisymmetric ranking logistic | Sanity baseline |
| B1 | Surface Elo → antisymmetric \(s\) | Primary comparison baseline |
| B2 | Rolling L2 logistic on difference features | Default supervised |
| B2b | GBM on differences + logit symmetrization | Dev-era candidate |
| B3 | Ensemble of antisymmetric scores | Dev-era candidate; weights from OOF |

Primary prediction: \(\hat{p} = P(A \text{ wins} \mid \text{completes})\).

## 14. Metrics and inference

### 14.1 Common eligible set

Compare models on the intersection of scoreable matches; report \(N\) and per-model coverage. Slice by `prediction_regime`.

### 14.2 Primary pre-registered comparison

- **Metric:** mean log-loss difference on final era (2019+), common eligible set, **`pre_tournament` regime**:

\[
\Delta = \overline{L}_{\mathrm{model}} - \overline{L}_{\mathrm{B1}}
\]

- Negative \(\Delta\) = improvement vs surface Elo.  
- Material only if \(\Delta \le -0.005\) nats (freeze before final-era peek) **and** time-blocked CI excludes 0.

### 14.3 Other metrics

Brier; calibration curve; **diagnostic** calibration intercept/slope on eval year; ECE secondary; AUC diagnostic only.

### 14.4 Metric confidence intervals

Paired, **moving multi-week blocks** (e.g. rolling 2–4 calendar weeks of tournaments), not IID matches and not single-tournament-as-independent without justification. Report CI for \(\Delta\).

## 15. Odds integration (ParlayAPI)

### 15.1 Storage

- Auth: `PARLAY_API_KEY`; never commit  
- Atomic two-sided quotes: `price_A`, `price_B`, `last_update`, `collected_at`  
- Preserve **schedule-version history** (each observed commence/schedule change with timestamp)—do not keep only the latest commence time  
- Track cancellations, suspensions, stale sides, API event id changes  
- Store API freshness metadata as provided (e.g. `stale_seconds`). If the API marks delayed re-serves of old prices, persist that boolean under a clear name such as `is_delayed_reserve` (ParlayAPI “topped_up”); do not use unexplained jargon in schema docs  

Call stored prices **observed**, not executable, unless execution and limits are verified.

### 15.2 Closing definition

Cutoff time = verified actual start if known, else best known scheduled start **as of the decision archive**, not a retrospective rewrite from a later schedule version alone.

Valid closing snapshot requires **all** of:

- `last_update <= cutoff - buffer` (default buffer **5 minutes**)  
- `collected_at <= cutoff - buffer`  
- `cutoff - last_update <= max_age` (default **30 minutes**)  
- Both sides present in the same atomic record  

Prefer Pinnacle when valid. Void cancelled / walkover events for CLV.

### 15.3 No-vig method (freeze before final eval)

Freeze one conversion before final-era market comparisons, e.g. **multiplicative margin removal** (normalize implied probs to sum to 1). Document in model card; do not switch during final reporting.

### 15.4 Matching and vocabulary

Match players with name + opponent + event + time context. Unmatched → queue.

| Term | Definition |
|------|------------|
| Model–market gap | \(\hat{p} - q_{\mathrm{no\text{-}vig}}\) at a stated time (estimands/regimes may differ) |
| CLV | Decision-time **observed** price vs later closing benchmark on same side—not merely \(\hat{p}\) vs close |

Report odds-join coverage and selection bias. Label forward regime explicitly (§8.4).

## 16. Uncertainty mechanics

### 16.1 Intervals on \(\hat{p}\)

Parameter/data uncertainty only. Match Bernoulli noise is **not** an interval around \(p\).

**Forecast interval algorithm (v1):** residual **block bootstrap of the training history** used to build state and coefficients:

1. Form moving multi-week blocks of tournaments in the relevant history.  
2. For each draw \(b = 1..B\): resample blocks with replacement (preserving approximate chronology by stitching blocks in sampled order, or use stationary block bootstrap—pick one and document).  
3. **Refit the full chronological pipeline** on the resampled history: Elo warm-start → tournament-batched updates → frozen features → supervised fit (and OOF calibrator if applicable).  
4. Score the target match(es) with draw \(b\)’s state/coefs at the appropriate cutoff.  
5. Interval = quantiles of \(\hat{p}^{(b)}\).

Bootstrapping only final coefficients while holding Elo/features fixed **understates** uncertainty and is not the v1 forecast-interval method.

### 16.2 Ability plots

From the same draws, keep joint \((R_A^{(b)}, R_B^{(b)})\). Plot paired differences / overlays from joint draws—not independent marginal densities.

### 16.3 Metric CIs

Moving multi-week blocks on the evaluation sample (§14.4).

## 17. Deliverables mapping (spec §H)

| Spec deliverable | v1 treatment |
|------------------|--------------|
| 1 Data-source registry | `docs/sources.md` (+ API license/limits) |
| 2 Canonical schema | `docs/schema.md` + typed tables |
| 3 Leakage audit | Orientation, pre-tournament cutoff, ranking policy, frozen features |
| 4 Baseline specs | B0–B2 + Elo K-by-level constants |
| 5 Walk-forward plan | §10 + `tml.validation` |
| 6 Calibration/backtest reports | Notebook 04; final \(\Delta\) on `pre_tournament` |
| 7 Feature pipeline | Builders + **persisted** feature store |
| 8 Model card | Estimand, regime, symmetry, no-vig method |
| 9 Dashboard spec | Deferred |
| 10 Roadmap | §18 |

## 18. Post-v1 roadmap

1. Validated match-date inference or external match timestamps → within-event features  
2. Settlement-aligned estimand  
3. Challenger odds if available; true CLV with decision-time observed prices  
4. Level-specific Elo slopes / hierarchical extensions if K-by-level is insufficient  
5. Set/game Monte Carlo; WTA; betting rules; dashboard  

## 19. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| `tourney_date` ≠ match date | Pre-tournament regime; no fake match-level claims |
| Cross-regime forward vs backtest | Explicit regime labels (§8.4) |
| Asymmetric models | Structural antisymmetric \(s\) (§7) |
| In-sample calibration | OOF / prequential calibrators only (§10.3) |
| Feature look-ahead via refit | Frozen feature store (§10.4) |
| Tour dummy cancels | K-by-level, not Challenger intercept (§11) |
| Retrospective odds | `collected_at` and `last_update` both ≤ cutoff; schedule history |
| Understated uncertainty | Full pipeline block bootstrap (§16) |
| CSV drift | Manifest hash snapshot id |
| ID collisions | Versioned identity map |

## 20. Success criteria for v1 complete

1. Manifest-hashed ingest; Sapling revision + dirty flag recorded when available  
2. Versioned player identity map; A/B hash orientation; structural symmetry tests  
3. Pre-tournament prediction regime implemented; within-tournament rest/load absent by default  
4. Frozen feature store used for all supervised fits  
5. B1 from 2000+; B0/B2 from 2005+; primary \(\Delta\) on 2019+ `pre_tournament` only  
6. OOF calibrators; diagnostic eval-year calibration intercept/slope  
7. Full-pipeline block-bootstrap intervals; joint ability draws  
8. Odds: atomic quotes, dual timestamp cutoff, schedule history, observed (not executable) wording, frozen no-vig method, coverage report  
9. Docs: sources, schema, leakage audit, model card  

---

*Second revision pass: `tourney_date` pre-tournament regime, structural antisymmetry, OOF calibration, K-by-level Challenger handling, odds `collected_at` cutoff, pipeline bootstrap uncertainty, frozen feature store, and minor reproducibility edits.*
