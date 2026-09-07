# Canonical schema

v1 models **P(player A beats player B | match starts and completes normally)** — label column `y_complete_win`, never `market_settlement_win`. Retirements, walkovers, defaults, and abandoned matches are stored with `completion_status` flags but do not enter supervised labels.

## Match ingest (`tml.data.ingest`)

Raw Sackmann-style CSV rows are normalized to:

| Column | Type | Description |
|--------|------|-------------|
| `tourney_id` | string | Tournament identifier |
| `tourney_name` | string | Tournament name |
| `tourney_date` | date | Tournament **start** date (`YYYYMMDD` in source) |
| `surface` | string | Hard, Clay, Grass, … |
| `tourney_level` | string | Source tourney level code |
| `match_num` | string | Match number within tournament |
| `winner_id`, `loser_id` | string | Raw player IDs (preserved as strings) |
| `winner_name`, `loser_name` | string | Display names |
| `score` | string | Raw score text |
| `best_of` | int | 3 or 5 |
| `round` | string | Draw round |
| `winner_rank`, `loser_rank` | string/int | Embedded pre-match ranks when present |
| `tour_level` | string | `atp` or `challenger` (set at ingest) |
| `completion_status` | enum | `completed`, `retirement`, `walkover`, `default`, `abandoned`, `unknown` |
| `source_file` | string | Relative path under `TML_DATA_DIR` |
| `match_id` | string | `{tour_level}:{tourney_id}:{match_num}` |

Serve and return columns from source CSVs pass through when present (`svpt`, `1stIn`, `1stWon`, `2ndWon`, `bpFaced`, `bpSaved`, …).

## Modeling table (`tml.data.modeling_table`)

Completed matches only, A/B oriented via stable hash (not winner/loser order):

| Column | Description |
|--------|-------------|
| `match_id`, `tourney_id`, `tourney_date`, `surface`, `best_of`, `tour_level` | Match context |
| `player_a_id`, `player_b_id` | Hash-oriented slots (`orientation_v1` digest) |
| `y_complete_win` | `1` if A won, else `0` |
| `prediction_regime` | `pre_tournament` for historical v1 path |
| `identity_map_version` | Version of `PlayerIdentityMap` used |

## Player identity (`tml.data.identity`)

| Field | Description |
|-------|-------------|
| `version` | Map version id |
| `conflicts` | Unresolved name collisions queued for review |
| Canonical ID | Raw Sackmann ID string unless merged via future alias rules |

Unresolved collisions should exclude rows from modeling until resolved.

## Feature store (`tml.features.store`)

Append-only parquet; **frozen at prediction time**. Required columns:

| Column | Description |
|--------|-------------|
| `match_id` | Unique match key |
| `prediction_cutoff` | Instant before tournament start (day prior to `tourney_date`) |
| `prediction_regime` | `pre_tournament` in v1 |
| `dataset_snapshot_id` | Manifest hash from ingest |
| `feature_schema_version` | e.g. `v1` |
| `y_complete_win` | Label at persist time |
| Feature columns | Antisymmetric differences (`elo_surface_diff`, `rank_diff`, …) |

Supervised fits must consume persisted rows only — never rebuild historical features from end-of-window Elo state.

## Ranking features (`tml.features.ranking`)

| Field | Description |
|-------|-------------|
| `rank_value` | Rank integer |
| `rank_source` | `match_embedded` or `ranking_snapshot` |
| `rank_as_of_date` | Official ranking date when known |
| `available_at` | Earliest date the value may enter features |
| `rank_staleness_days` | Days from `rank_as_of_date` to tournament start |
| `rank_missing` | Boolean indicator |

Embedded ranks: `available_at = tourney_date`. Snapshot ranks: usable iff `rank_as_of_date <= tourney_date` under pre-tournament cutoff.

## Elo state (`tml.features.elo`)

Shared player pool across ATP and Challenger; **K-by-level**:

| Parameter | Default | Role |
|-----------|---------|------|
| `mu0` | 1500 | Prior for new players |
| `k_atp` | 32 | Update reliability on main tour |
| `k_challenger` | 20 | Update reliability on Challenger |
| Surface keys | hard, clay, grass, … | Separate ratings per `(player_id, surface)` |

Challenger main-effect dummies are **not** used — they cancel in antisymmetric scores. Level enters via K-factor only in v1.

## Prequential predictions (`tml.validation.prequential`)

Output columns include `p_b0`, `p_b1`, `p_b2`, `y`, `prediction_regime`, `dataset_snapshot_id`, plus context slices.

## Observed quotes (`tml.odds.storage.QuoteRecord`)

Atomic two-sided **observed** prices (not executable):

| Field | Description |
|-------|-------------|
| `event_id`, `sport_key`, `bookmaker` | API identifiers |
| `player_a`, `player_b` | Matched player names |
| `price_a`, `price_b` | Observed prices |
| `odds_format` | `american` or `decimal` |
| `commence_time` | Scheduled start for this schedule version |
| `schedule_observed_at` | When this schedule version was observed |
| `last_update` | Book-side last update timestamp |
| `collected_at` | Collection timestamp |
| `stale_seconds` | API freshness metadata |
| `is_delayed_reserve` | Delayed re-serve flag (ParlayAPI “topped_up”) |
| `is_cancelled`, `is_suspended` | Event state |
| `actual_start` | Verified start when known |

Closing selection requires both sides present, `last_update <= cutoff - 5m`, `collected_at <= cutoff - 5m`, and `cutoff - last_update <= 30m`. Prefer Pinnacle when valid.

## No-vig probabilities (`tml.odds.devig`)

Frozen v1 method: **multiplicative margin removal** — normalize implied probabilities from an atomic two-sided quote so they sum to 1. Do not switch methods during final-era reporting.

## Vocabulary

| Term | Definition |
|------|------------|
| Model–market gap | `p̂ − q_no-vig` at a stated time; estimands/regimes may differ |
| CLV | Decision-time **observed** price vs later closing benchmark on the same side — not merely `p̂` vs close |
| Observed price | API/collector snapshot; not guaranteed executable |
| Executable price | Requires verified limits, fills, and slippage — out of v1 scope |
