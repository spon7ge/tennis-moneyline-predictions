# Model card — Tennis Moneyline v1

**Purpose:** Research estimates of conditional sporting-strength win probabilities for ATP and Challenger singles. This is not a tip sheet, profit guarantee, or executable betting system.

## Estimand

\[
p = P(\text{player A beats player B} \mid \text{match starts and completes normally})
\]

- Label: `y_complete_win` (1 if A won among completed matches)
- **Not** unconditional pre-match win probability
- **Not** sportsbook settlement probability (retirements may be graded differently by books)

### Market mismatch (documented, not “fixed” in v1)

Books may grade some retirements as wins for the leading player. Therefore:

- Model \(p\) is sporting-strength conditional on completion
- No-vig market \(q\) estimates a **different** estimand when retirement grading differs
- Forward comparisons report **model–market probability gap under mismatched estimands** — not proof of edge
- **CLV** requires a decision-time **observed** price and a later closing benchmark on the same side; it is separate from the research estimand

## Intended use

- Offline research, calibration studies, and uncertainty visualization
- Forward odds snapshot comparison with explicit regime labels
- **Out of scope:** live trading, Kelly sizing, profitability claims

## Training data

| Aspect | Choice |
|--------|--------|
| Tours | ATP main + ATP Challenger |
| Excluded labels | WTA, ITF, qualifying; retirements; walkovers; defaults |
| Pre-2000 | Elo burn-in only — not in B0/B2 metrics |
| Snapshot pinning | `dataset_snapshot_id` from ingest manifest |

## Forecasting regime

**Primary historical regime:** `pre_tournament`

- Cutoff: end of day before `tourney_date`
- All matches in a tournament scored from the same frozen state
- Elo updates after the full tournament batch is predicted
- Within-tournament rest/load features absent unless `match_timestamp_reliable`

Forward ParlayAPI scoring may use `pre_match` or `forward_live_info` — report separately; do not mix silently with primary backtest metrics.

## Models

| Stage | Model | Role |
|-------|-------|------|
| B0 | Antisymmetric ranking logistic | Sanity baseline |
| B1 | Surface Elo → antisymmetric score | Primary comparison baseline |
| B2 | Rolling L2 logistic on difference features | Default supervised |
| B2b / B3 | GBM / ensemble candidates | **Out of v1; not implemented** |

### Structural symmetry

\[
s(A,B,c) = -s(B,A,c), \qquad \hat{p}(A,B,c) = \operatorname{logistic}(s(A,B,c))
\]

Difference features only; temperature scaling allowed; eval-year calibration intercept/slope are **diagnostics only** unless fit out-of-time on antisymmetric scores.

### Elo (B1) — K-by-level

Shared player pool across tours. Separate update K-factors:

- `k_atp` (default 32) for main-tour completed matches
- `k_challenger` (default 20) for Challenger completed matches

Surface-specific ratings; BO5 expectancy multiplier on rating differential. No Challenger main-effect dummy (cancels in antisymmetric form).

## Validation

- Prequential walk-forward: expanding Elo by tournament batch; rolling 5-year supervised coefficients
- Development era: 2005–2018 (selection)
- Final era: 2019+ (primary report, single peek)
- Primary metric: mean log-loss difference vs B1 on common eligible set, `pre_tournament` regime only:

\[
\Delta = \overline{L}_{\mathrm{model}} - \overline{L}_{\mathrm{B1}}
\]

Material improvement threshold: \(\Delta \le -0.005\) nats with time-blocked CI excluding 0.

Metric CIs use moving multi-week tournament blocks (not IID match assumption).

## Uncertainty

- Intervals on \(\hat{p}\): full **pipeline block bootstrap** (refit Elo → frozen features → supervised fit per draw)
- Match Bernoulli noise is **not** included in forecast intervals
- Ability plots use joint \((R_A^{(b)}, R_B^{(b)})\) draws from the same bootstrap

## Odds integration

| Topic | v1 choice |
|-------|-----------|
| Provider | ParlayAPI forward snapshots |
| Price type | **Observed** — not executable without verified fills/limits |
| No-vig method | **Multiplicative margin removal** (`tml.odds.devig.multiplicative_novig`) — frozen before final-era comparisons |
| Closing | Dual timestamp cutoff (`last_update`, `collected_at`) with 5m buffer, 30m max age; prefer Pinnacle |
| Vocabulary | Model–market gap ≠ CLV |

## Known limitations

- No historical odds backtest in v1
- No settlement-aligned estimand
- No within-tournament features without verified timestamps
- Challenger forward odds availability unconfirmed
- CSV `tourney_date` granularity limits match-level prequential claims
- Full multi-year corpus execution has not yet been completed; v1 verification uses automated tests and short smoke runs

## Ethical / honesty notes

Per project onboarding policy: never claim certainty or guaranteed profitability. Report coverage, regime labels, and estimand mismatches explicitly. If tests or data checks fail, show the output — do not describe unverified work as working.

## Version

- Package: `tml` 0.1.0
- Feature schema: `v1`
- Design reference: `docs/superpowers/specs/2026-09-06-tennis-moneyline-v1-design.md`
