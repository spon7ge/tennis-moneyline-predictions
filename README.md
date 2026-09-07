# Tennis Moneyline (tml)

Research package for estimating **P(A wins | match completes)** on ATP and Challenger tours using Sackmann-style historical match data and optional forward odds from ParlayAPI.

This is a personal research project — not a tip sheet or profit guarantee. See [docs/model_card.md](docs/model_card.md) for the estimand, `pre_tournament` regime, structural antisymmetry, and odds vocabulary.

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/sources.md](docs/sources.md) | Data providers, licensing notes, reproducibility |
| [docs/schema.md](docs/schema.md) | Canonical tables and column definitions |
| [docs/leakage_audit.md](docs/leakage_audit.md) | Orientation, cutoff, frozen features, ranking policy |
| [docs/model_card.md](docs/model_card.md) | Estimand, models, validation, no-vig method |

## Prerequisites

- Python 3.11+ (see `.python-version`)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

Copy the environment template and point `TML_DATA_DIR` at your Sackmann CSV tree:

```bash
cp .env.example .env
```

With **uv** (recommended):

```bash
uv sync --all-extras
```

With **pip**:

```bash
python3 -m pip install -e ".[dev]"
```

Place Sackmann-style CSVs under `./tml-data/` (gitignored). Expected layout:

- `YYYY.csv` — ATP main tour
- `YYYY_challenger.csv` — ATP Challenger

## Run tests

```bash
uv run pytest -q
# verbose: uv run pytest -v
```

All business logic lives in `src/tml/`; tests under `tests/` must pass before reporting results.

## Ingest historical matches

Ingest reads local CSVs, attaches completion flags and lineage, and returns a manifest-hashed snapshot id. Example for a single year (extend the glob for full history):

```python
from pathlib import Path

from tml.data.ingest import ingest_match_files
from tml.data.identity import PlayerIdentityMap
from tml.data.modeling_table import build_modeling_table
from tml.shared.config import get_settings

root = get_settings().tml_data_dir

atp = ingest_match_files(
    sorted(root.glob("2024.csv")),
    root=root,
    tour_level="atp",
)
challenger = ingest_match_files(
    sorted(root.glob("2024_challenger.csv")),
    root=root,
    tour_level="challenger",
)

print(atp.dataset_snapshot_id, len(atp.matches))

identity = PlayerIdentityMap()
modeling = build_modeling_table(atp.matches, identity)
print(modeling.head())
```

Pin `dataset_snapshot_id` in experiments — `ingested_at` alone is not sufficient.

## Research notebooks

Thin wrappers under `notebooks/` import `tml` and outline the v1 pipeline (no duplicated business logic):

| Notebook | Purpose |
|----------|---------|
| `01_ingest_and_qc.ipynb` | Manifest ingest and completion QC |
| `02_features_and_leakage_audit.ipynb` | Frozen features and cutoff checks |
| `03_baselines.ipynb` | B0/B1 antisymmetric baselines |
| `04_walkforward_validation.ipynb` | Prequential walk-forward and primary Δ |
| `05_uncertainty_plots.ipynb` | Bootstrap intervals and ability plots |
| `06_forward_odds.ipynb` | Observed quotes, no-vig, model–market gap |
| `07_positive_ev.ipynb` | Bettable EV board vs observed prices (research only) |

Run with Jupyter or VS Code; ensure `uv sync` has been run so `tml` is importable.

## Forward odds (optional)

Set `PARLAY_API_KEY` in `.env` for ParlayAPI snapshots. Stored prices are **observed**, not executable. See [docs/schema.md](docs/schema.md) for closing rules and frozen multiplicative no-vig.

## Deploy

Not applicable — local research package with no production deployment in v1.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
