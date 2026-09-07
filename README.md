# Tennis Moneyline (tml)

Research package for estimating **P(A wins | match completes)** on ATP and Challenger tours using Sackmann-style historical match data and optional forward odds from ParlayAPI.

## Prerequisites

- Python 3.11+ (see `.python-version`)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

Copy environment template and point `TML_DATA_DIR` at your Sackmann CSV tree:

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

## Run tests

```bash
uv run pytest
# or: pytest
```

## Deploy

Not applicable — this is a local research package with no production deployment in v1.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
