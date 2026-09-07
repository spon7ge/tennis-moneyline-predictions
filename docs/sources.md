# Data sources

This registry documents every external input used by the v1 research package. Pin experiments to `dataset_snapshot_id` (see [schema.md](schema.md)); `ingested_at` alone is not sufficient for reproducibility.

## Historical match results (Jeff Sackmann–style CSVs)

| Field | Value |
|-------|-------|
| Location | `TML_DATA_DIR` (default `./tml-data/`) |
| Tours in v1 | ATP main tour (`YYYY.csv`), ATP Challenger (`YYYY_challenger.csv`) |
| Out of v1 labels | WTA, ITF, qualifying draws |
| License | Sackmann datasets are widely used in open research; confirm current terms on the upstream repository before redistribution or commercial use |
| Retention | Local copies are gitignored; each ingest run records a manifest hash |

**Files consumed**

- `YYYY.csv` — ATP main-tour match results with embedded pre-match ranks and serve stats when present
- `YYYY_challenger.csv` — Challenger tour results (same column shape)
- Optional ranking snapshot CSVs when present (not required for v1 baseline path)

**Known limitations**

- `tourney_date` is the tournament **start date**, not each match’s calendar date (see [leakage_audit.md](leakage_audit.md))
- Retirements, walkovers, defaults, and abandoned matches are retained in storage but **excluded from modeling labels**

## Forward moneylines (ParlayAPI)

| Field | Value |
|-------|-------|
| Provider | [ParlayAPI](https://parlay-api.com) |
| Auth | `PARLAY_API_KEY` in `.env` — never commit |
| Sport key (v1) | `tennis_atp` |
| Historical archive | **Not required for v1** — tier windows are too short for 2000+ market backtests |
| Challenger odds | Confirm availability via `GET /v1/sports` before treating Challenger forward odds as a deliverable |

**Operational constraints**

- Respect provider rate limits and terms of service; do not scrape without reviewing robots.txt and licensing
- Store **observed** prices only — not executable fills unless execution and limits are verified separately
- Preserve schedule-version history (each commence/schedule change with timestamp); do not keep only the latest commence time
- Track API freshness metadata (`stale_seconds`, `is_delayed_reserve` when the API marks delayed re-serves)

**Before committing an odds archive to version control**, document licensing, retention rights, rate limits, and Challenger availability in this file.

## Internal derived artifacts

| Artifact | Path (default) | Role |
|----------|----------------|------|
| Feature store | `data/processed/features/features.parquet` | Frozen prediction-time feature rows |
| Quote store | `data/raw/odds/quotes.parquet` | Atomic two-sided observed quotes |

These paths are gitignored; rebuild from source inputs plus `dataset_snapshot_id`.

## Reproducibility metadata

Each ingest run records:

- Sorted manifest: `relative_path`, `size_bytes`, `sha256` per input file
- `dataset_snapshot_id` = SHA-256 of the canonical serialized manifest
- `ingested_at` (timezone-aware wall clock)
- Sapling revision id and dirty-worktree flag when available (do not assume git)

Experiments must pin `dataset_snapshot_id` alongside model and feature schema versions.
