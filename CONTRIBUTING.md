# Contributing

## Setup

1. Clone the repo and create a branch from `main` (or the active feature branch).
2. Copy `.env.example` to `.env` and fill in values locally — never commit `.env`.
3. Install dependencies:

   ```bash
   uv sync --all-extras
   ```

   Or with pip: `python3 -m pip install -e ".[dev]"`.

4. Run tests before opening a PR:

   ```bash
   uv run pytest
   ```

## Branch naming

Use descriptive prefixes:

- `feat/` — new functionality
- `fix/` — bug fixes
- `docs/` — documentation only
- `refactor/` — behaviour-preserving restructuring

Example: `feat/tennis-moneyline-v1`

## Pull requests

- **One concern per PR.** Do not mix refactors with behaviour changes.
- Describe what changed, why, and how you verified it (test output, smoke commands).
- Keep commits small and coherent; imperative subject under 72 characters.

## Engineering guide

All code must follow the project contract in [skills.md](skills.md) — especially secrets handling, boundary validation, error handling, and tests in the same change as code.
