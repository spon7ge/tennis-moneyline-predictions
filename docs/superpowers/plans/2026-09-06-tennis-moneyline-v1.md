# Tennis Moneyline v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Also required:** read root `skills.md` before any code-producing step; follow Global Constraints → Engineering.

**Goal:** Build an installable Python research package that estimates \(P(A\text{ wins}\mid\text{completes})\) for ATP + Challenger under a pre-tournament information set, with antisymmetric scores, frozen features, walk-forward validation, and forward ParlayAPI odds snapshots.

**Architecture:** Ingest Sackmann-style CSVs into versioned canonical tables → hash-oriented A/B modeling rows → tournament-batched Elo + frozen feature store → antisymmetric B0/B1/B2 models with prequential yearly coefficient fits → metrics on common eligible sets → optional forward odds collector with dual-timestamp closing rules.

**Tech Stack:** Python 3.11+ (pinned), pandas, numpy, scikit-learn, pyarrow, pytest, matplotlib, pydantic (boundary validation), `requests` (+ optional `parlay-api`), python-dotenv; lock with **uv** (`uv.lock`)

**Spec:** `docs/superpowers/specs/2026-09-06-tennis-moneyline-v1-design.md`  
**Code contract:** Root `skills.md` (Project Engineering Guide) — bind before every code-producing change

## Global Constraints

### Domain (design)

- Estimand: `y_complete_win` only; retirements/walkovers/defaults excluded from labels
- Tours: ATP + Challenger only (ignore WTA/quali for modeling)
- Historical regime: `pre_tournament` when only `tourney_date` exists; never claim match-day prequential
- No within-tournament rest/load without `match_timestamp_reliable`
- Structural antisymmetry: \(s(A,B)=-s(B,A)\), \(p=\mathrm{logistic}(s)\); no directional intercept; temperature calib only for applied calibration
- Orientation: stable hash, not `min(player_id)`
- Elo: shared ratings; different \(K_{\mathrm{atp}}\) vs \(K_{\mathrm{challenger}}\)
- B1 metrics from 2000+; B0/B2 from 2005+; model selection 2005–2018; primary \(\Delta\) on 2019+ `pre_tournament` only
- `FitSupervised` uses persisted frozen feature rows only
- Odds prices are **observed**, not executable; closing needs `last_update` and `collected_at` ≤ cutoff

### Engineering (`skills.md` — adapted to this Python research package)

`skills.md` examples are TypeScript/Zod/Vitest. **This repo’s stack is Python.** Apply the *rules*; translate the *examples*:

| skills.md | This project |
|-----------|--------------|
| Zod schemas | **pydantic** models at I/O boundaries (ingest paths, odds API payloads, CLI args) |
| `Result<T,E>` in `src/shared/result.ts` | `tml.shared.result.Result` (`Ok` / `Err`) for expected failures |
| `invariant()` | `tml.shared.invariant.invariant` — throws; never soft-assert |
| `process.env` only in `config.ts` | **only** `tml.shared.config` reads env; everything else imports typed settings |
| Vitest/Jest | **pytest**; behaviour tests, not implementation-coupling |
| Co-located `*.test.ts` | Mirror package path under `tests/` (move/delete with the module) |
| Feature folders (`users/`) | Domain packages (`matches`, `features`, `models`, `backtest`, `odds`) + thin `shared/` |
| npm lockfile | **`uv.lock`** committed; `.python-version` pins 3.11+ |
| Web auth / a11y / e2e | **N/A** unless we add an HTTP UI (say so in §17 walk) |

**Hard stops (always):**

- Never commit or log secrets (`PARLAY_API_KEY`, tokens). Mask in logs.
- Never swallow errors (`except: pass` / bare `except`).
- Never build shell commands by string concatenation of untrusted input.
- Never invent libraries/APIs — check `pyproject.toml` / lockfile first. **Ask before adding a dependency.**
- Never mock the project’s own persisted stores with fake in-memory stand-ins that diverge from parquet/CSV behaviour — use temp files / real formats (§07).
- Tests ship in the **same change** as the code; no “tests follow-up.”
- Small coherent commits; imperative subjects; body explains *why*.
- Over-engineering is a failure mode (§00): no framework-for-future-DB, no 100% coverage gate.

**Per-task agent procedure (skills.md preamble):**

1. State which guide sections bind.
2. One–two sentence plan (files + applicable §17 Must items).
3. Write failing tests with the code.
4. Before “done,” walk §17 Must (N/A with reason where web-only).
5. Report honestly — show failing output if any; never claim unverified green.

**§17 Must for this repo (non-web):**

- [ ] No secrets committed; nothing sensitive in logs
- [ ] Inputs validated at every boundary (files, API JSON, CLI)
- [ ] Auth handlers — **N/A** (no HTTP app in v1)
- [ ] No string-concatenated SQL/HTML/shell of untrusted input
- [ ] Errors handled — `Result` for expected; throw for exceptional; never silent
- [ ] Tests written and passing (unit + integration smoke where I/O meets)
- [ ] Bug fixes include regression tests
- [ ] CI green — format/lint/typecheck/tests (Task 1 workflow)
- [ ] Keyboard/a11y — **N/A** (no UI product in v1)
- [ ] README / `.env.example` updated when setup/config changes

**§17 Should:** small functions; precise types (`mypy` clean on `src/tml`); invariants at impossible states; comments explain *why*; DRY at third occurrence; config over magic numbers; enough logging to debug a failed ingest/odds poll; small commits.

---

## File structure (create)

```
pyproject.toml
uv.lock                    # committed after first `uv lock`
.python-version            # e.g. 3.11
README.md
CONTRIBUTING.md
.env.example
.github/workflows/ci.yml
src/tml/
  __init__.py
  shared/
    __init__.py
    result.py              # Ok | Err
    invariant.py           # invariant(condition, message)
    config.py              # sole env reader → Settings
  data/                    # matches domain (ingest → modeling)
    __init__.py
    completion.py
    manifest.py
    ingest.py
    identity.py
    orientation.py
    modeling_table.py
  features/
    __init__.py
    ranking.py
    elo.py
    builders.py
    store.py
  models/
    __init__.py
    symmetry.py
    ranking_logit.py
    elo_prob.py
    supervised.py
    calibration.py
  validation/              # backtest domain
    __init__.py
    prequential.py
    metrics.py
    blocks.py
    uncertainty.py
  odds/
    __init__.py
    client.py
    storage.py
    closing.py
    matching.py
    devig.py
  viz/
    __init__.py
    plots.py
tests/                     # mirrors src/tml/* ; co-move with modules
  shared/
  data/
  features/
  models/
  validation/
  odds/
notebooks/                 # thin wrappers only — no business logic
docs/
  sources.md
  schema.md
  leakage_audit.md
  model_card.md
data/processed/            # gitignored
data/odds/                 # gitignored
```

---

### Task 1: Scaffold, shared primitives, onboarding, completion parser

**skills.md binds:** §00–§01, §04, §07–§10, §12–§13, §17 Must (secrets, tests, README/env, errors)

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `.env.example`, `README.md`, `CONTRIBUTING.md`
- Create: `.github/workflows/ci.yml`
- Create: `src/tml/__init__.py`
- Create: `src/tml/shared/__init__.py`, `result.py`, `invariant.py`, `config.py`
- Create: `src/tml/data/__init__.py`, `completion.py`
- Create: `tests/shared/test_result.py`, `tests/shared/test_invariant.py`, `tests/shared/test_config.py`
- Create: `tests/data/test_completion.py`

**Interfaces:**
- Consumes: none
- Produces:
  - `Result` = `Ok[T] | Err[E]` in `tml.shared.result`
  - `invariant(condition: object, message: str) -> None`
  - `get_settings() -> Settings` (pydantic); fields: `parlay_api_key: str | None`, `tml_data_dir: Path`
  - `parse_completion_status(score: str | None) -> str` ∈ `completed|retirement|walkover|default|abandoned|unknown`

- [ ] **Step 1: Write failing tests** (`tests/shared/*` + completion)

```python
# tests/shared/test_result.py
from tml.shared.result import Ok, Err

def test_ok_and_err_discriminate():
    assert Ok(1).ok is True and Ok(1).value == 1
    assert Err("x").ok is False and Err("x").error == "x"

# tests/shared/test_invariant.py
import pytest
from tml.shared.invariant import invariant

def test_invariant_passes():
    invariant(True, "ok")

def test_invariant_raises():
    with pytest.raises(RuntimeError, match="Invariant violated"):
        invariant(False, "ratings must be finite")

# tests/shared/test_config.py
from tml.shared.config import get_settings

def test_settings_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PARLAY_API_KEY", "test-key-not-real")
    monkeypatch.setenv("TML_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    s = get_settings()
    assert s.parlay_api_key == "test-key-not-real"
    assert s.tml_data_dir == tmp_path

# tests/data/test_completion.py
import pytest
from tml.data.completion import parse_completion_status

@pytest.mark.parametrize(
    "score,expected",
    [
        ("6-4 6-3", "completed"),
        ("6-4 3-6 6-2", "completed"),
        ("6-4 2-6 1-4 RET", "retirement"),
        ("W/O", "walkover"),
        ("Walkover", "walkover"),
        ("DEF", "default"),
        ("4-6 0-1 abd", "abandoned"),
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_parse_completion_status(score, expected):
    assert parse_completion_status(score) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/alexgonzalez/Documents/tennis moneyline predictions" && python -m pytest tests/shared tests/data/test_completion.py -v`  
Expected: FAIL (imports missing)

- [ ] **Step 3: Implement scaffold + shared + completion**

`pyproject.toml` extras: add `pydantic`, `pydantic-settings`; scripts `test`, `lint`/`typecheck` if ruff/mypy added — **ask before adding ruff/mypy**; minimum is pytest. Prefer: `uv init` lockflow.

`.python-version`: `3.11`

`.env.example`:

```bash
# ParlayAPI key from https://parlay-api.com — never commit the real value
PARLAY_API_KEY=
# Root of Sackmann-style CSV tree (default: ./tml-data)
TML_DATA_DIR=./tml-data
```

`src/tml/shared/result.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")

@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    value: T
    ok: bool = True

@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    error: E
    ok: bool = False

type Result[T, E] = Ok[T] | Err[E]
```

`src/tml/shared/invariant.py`:

```python
def invariant(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Invariant violated: {message}")
```

`src/tml/shared/config.py` — sole env reader via `pydantic_settings.BaseSettings`; `get_settings` cached; **never log `parlay_api_key`**.

`completion.py` as before (regex status parser).

`README.md`: what / prerequisites / `uv sync` or `pip install -e ".[dev]"` / `pytest` / no deploy yet.  
`CONTRIBUTING.md`: branch naming, one concern per PR, how to run tests, link `skills.md`.

CI workflow: install, `pytest`, (optional mypy later).

- [ ] **Step 4: Lock deps and run tests**

Run: `uv lock && uv sync --all-extras && uv run pytest tests/shared tests/data/test_completion.py -v`  
(If `uv` unavailable: `pip install -e ".[dev]" && pytest …` and note assumption; still create `uv.lock` when uv is available.)  
Expected: PASS. Commit `uv.lock`.

- [ ] **Step 5: §17 walk + commit**

Must: secrets (only `.env.example`), tests green, README/env present, errors via invariant/Result. Auth/a11y N/A.

```bash
git add pyproject.toml uv.lock .python-version .gitignore .env.example README.md CONTRIBUTING.md \
  .github/workflows/ci.yml src/tml tests/shared tests/data/test_completion.py
git commit -m "$(cat <<'EOF'
feat: scaffold tml with shared Result/invariant/config and completion parser

EOF
)"
```

---

### Task 2: Dataset manifest hashing

**Files:**
- Create: `src/tml/data/manifest.py`
- Create: `tests/data/test_manifest.py`

**Interfaces:**
- Consumes: filesystem paths
- Produces:
  - `file_sha256(path: Path) -> str`
  - `build_manifest(paths: list[Path], root: Path) -> list[dict]` with keys `relative_path`, `size_bytes`, `sha256`
  - `snapshot_id(manifest: list[dict]) -> str` (SHA-256 of canonical JSON)

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_manifest.py
import json
from pathlib import Path
from tml.data.manifest import build_manifest, snapshot_id

def test_snapshot_id_stable_and_order_independent(tmp_path: Path):
    a = tmp_path / "b.csv"
    b = tmp_path / "a.csv"
    a.write_text("x\n1\n")
    b.write_text("x\n2\n")
    m1 = build_manifest([a, b], root=tmp_path)
    m2 = build_manifest([b, a], root=tmp_path)
    assert snapshot_id(m1) == snapshot_id(m2)
    assert m1[0]["relative_path"] == "a.csv"
    assert "sha256" in m1[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_manifest.py -v`  
Expected: FAIL import/not found

- [ ] **Step 3: Implement**

```python
# src/tml/data/manifest.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(paths: list[Path], root: Path) -> list[dict]:
    rows = []
    for p in paths:
        p = p.resolve()
        rel = str(p.relative_to(root.resolve()))
        rows.append(
            {
                "relative_path": rel.replace("\\", "/"),
                "size_bytes": p.stat().st_size,
                "sha256": file_sha256(p),
            }
        )
    rows.sort(key=lambda r: r["relative_path"])
    return rows


def snapshot_id(manifest: list[dict]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/data/test_manifest.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tml/data/manifest.py tests/data/test_manifest.py
git commit -m "$(cat <<'EOF'
feat: add sorted dataset manifest and snapshot id hashing

EOF
)"
```

---

### Task 3: Ingest ATP + Challenger CSVs

**Files:**
- Create: `src/tml/data/ingest.py`
- Create: `tests/data/test_ingest.py`
- Create: `tests/data/fixtures/` with tiny synthetic CSVs (do not depend on full `tml-data` in unit tests)

**Interfaces:**
- Consumes: `build_manifest`, `snapshot_id`, `parse_completion_status`
- Produces: `ingest_match_files(paths, root, tour_level) -> IngestResult` dataclass with `matches: pd.DataFrame`, `manifest`, `dataset_snapshot_id`, `ingested_at`

Required columns after ingest (minimum): `tourney_id`, `tourney_name`, `tourney_date`, `surface`, `tourney_level`, `match_num`, `winner_id`, `loser_id`, `winner_name`, `loser_name`, `score`, `best_of`, `round`, `winner_rank`, `loser_rank`, serve cols if present, `tour_level`, `completion_status`, `source_file`, `match_id`

`match_id` = `f"{tour_level}:{tourney_id}:{match_num}"`  
`tourney_date` parsed as `datetime.date`

- [ ] **Step 1: Write fixture CSVs + failing test**

Create `tests/data/fixtures/atp_tiny.csv` and `challenger_tiny.csv` with header matching Sackmann columns and 2–3 rows including one `RET`.

```python
# tests/data/test_ingest.py
from pathlib import Path
from tml.data.ingest import ingest_match_files

FIXTURES = Path(__file__).parent / "fixtures"

def test_ingest_adds_completion_and_match_id():
    paths = [FIXTURES / "atp_tiny.csv"]
    result = ingest_match_files(paths, root=FIXTURES, tour_level="atp")
    df = result.matches
    assert "completion_status" in df.columns
    assert "match_id" in df.columns
    assert result.dataset_snapshot_id
    assert (df["tour_level"] == "atp").all()
    assert set(df["completion_status"]) >= {"completed", "retirement"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_ingest.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `ingest.py`**

Load with pandas; coerce ids to string; apply `parse_completion_status`; attach `tour_level`, `source_file`, `match_id`; parse `tourney_date` as `%Y%m%d` or pandas datetime → date; build manifest via Task 2.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/data/test_ingest.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tml/data/ingest.py tests/data/test_ingest.py tests/data/fixtures
git commit -m "$(cat <<'EOF'
feat: ingest ATP/Challenger match CSVs with lineage fields

EOF
)"
```

---

### Task 4: Hash orientation, identity map stub, and modeling table

**Files:**
- Create: `src/tml/data/orientation.py`
- Create: `src/tml/data/identity.py`
- Create: `src/tml/data/modeling_table.py`
- Create: `tests/data/test_orientation.py`
- Create: `tests/data/test_identity.py`
- Create: `tests/data/test_modeling_table.py`

**Interfaces:**
- Consumes: ingested matches
- Produces:
  - `PlayerIdentityMap(version: str = "v1")` with `resolve(raw_id: str, name: str | None = None) -> str` (v1: return `str(raw_id)`; record collisions in `conflicts`)
  - `assign_orientation(match_id: str, player_u: str, player_v: str) -> tuple[str, str]` → `(player_a_id, player_b_id)`
  - `build_modeling_table(matches: pd.DataFrame, identity: PlayerIdentityMap) -> pd.DataFrame` with only `completion_status == "completed"`, columns `player_a_id`, `player_b_id`, `y_complete_win`, `prediction_regime` default `pre_tournament`, `identity_map_version`, plus context fields

Orientation algorithm (exact):

```python
import hashlib

def assign_orientation(match_id: str, player_u: str, player_v: str) -> tuple[str, str]:
    u, v = sorted([str(player_u), str(player_v)])
    digest = hashlib.sha256(f"orientation_v1|{match_id}|{u}|{v}".encode()).hexdigest()
    if int(digest[-1], 16) % 2 == 1:
        return u, v
    return v, u
```

- [ ] **Step 1: Write failing tests**

```python
def test_orientation_independent_of_winner_loser_order():
    a1, b1 = assign_orientation("m1", "100", "200")
    a2, b2 = assign_orientation("m1", "200", "100")
    assert (a1, b1) == (a2, b2)

def test_identity_map_version_tagged():
    m = PlayerIdentityMap(version="v1")
    assert m.resolve("123") == "123"
    assert m.version == "v1"

def test_modeling_table_excludes_retirements_and_sets_label():
    import pandas as pd
    from tml.data.identity import PlayerIdentityMap
    from tml.data.modeling_table import build_modeling_table

    matches = pd.DataFrame(
        [
            {
                "match_id": "atp:T1:1",
                "tourney_id": "T1",
                "tourney_date": "2000-01-01",
                "winner_id": "100",
                "loser_id": "200",
                "completion_status": "completed",
                "surface": "Hard",
                "best_of": 3,
                "tour_level": "atp",
            },
            {
                "match_id": "atp:T1:2",
                "tourney_id": "T1",
                "tourney_date": "2000-01-01",
                "winner_id": "100",
                "loser_id": "300",
                "completion_status": "retirement",
                "surface": "Hard",
                "best_of": 3,
                "tour_level": "atp",
            },
        ]
    )
    out = build_modeling_table(matches, PlayerIdentityMap(version="v1"))
    assert len(out) == 1
    assert out.iloc[0]["y_complete_win"] in (0, 1)
    assert out.iloc[0]["identity_map_version"] == "v1"
    assert out.iloc[0]["prediction_regime"] == "pre_tournament"
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement `identity.py`, orientation, and modeling_table**

`y_complete_win = 1` iff resolved `winner_id == player_a_id`.

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/tml/data/orientation.py src/tml/data/identity.py src/tml/data/modeling_table.py tests/data/test_orientation.py tests/data/test_identity.py tests/data/test_modeling_table.py
git commit -m "$(cat <<'EOF'
feat: hash A/B orientation, identity map stub, completed-only modeling table

EOF
)"
```

---

### Task 5: Ranking availability helpers

**Files:**
- Create: `src/tml/features/__init__.py`
- Create: `src/tml/features/ranking.py`
- Create: `tests/features/test_ranking.py`

**Interfaces:**
- Produces: `rank_feature_pair(row, tourney_date) -> dict` with `rank_diff`, `rank_missing_a`, `rank_missing_b`, `rank_source`, `rank_staleness_days` using embedded ranks with `available_at = tourney_date` (allowed at pre-tournament cutoff)

- [ ] **Step 1: Failing test** — missing rank sets flags and leaves `rank_diff` as `None`/`NaN`; both ranks present → `log(rank_b) - log(rank_a)` or `rank_a - rank_b` (document: use `log_rank_a - log_rank_b` so higher-ranked favorite has positive signal toward A). Freeze: `rank_diff = log(rank_b) - log(rank_a)` when both ≥ 1.

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: ranking available_at helpers for embedded pre-match ranks

EOF
)"
```

---

### Task 6: Surface Elo with K by tour level

**Files:**
- Create: `src/tml/features/elo.py`
- Create: `src/tml/models/symmetry.py`
- Create: `src/tml/models/elo_prob.py`
- Create: `tests/features/test_elo.py`
- Create: `tests/models/test_symmetry.py`

**Interfaces:**
- Produces:
  - `EloState` with ratings keyed by `(player_id, surface)`
  - `expected_score(r_a, r_b, best_of: int) -> float`
  - `update_tournament(state, matches_df) -> EloState` (batch after predictions)
  - `rating_diff(state, player_a, player_b, surface) -> float`
  - `antisymmetric_logit_from_diff(diff: float, scale: float = 400/log(10) mapping) -> float`
  - `elo_win_prob(state, player_a, player_b, surface, best_of) -> float` with \(p=\mathrm{logistic}(s)\), \(s=-s_{\mathrm{swapped}}\)

Defaults: `mu0=1500`, `k_atp=32`, `k_challenger=20`, BO5: multiply rating diff by `1.1` before logistic (constant frozen; tune only in later validation task / config).

- [ ] **Step 1: Failing tests**

```python
def test_elo_probability_is_antisymmetric():
    state = EloState()
    state.set("1", "Hard", 1600)
    state.set("2", "Hard", 1500)
    p_ab = elo_win_prob(state, "1", "2", "Hard", best_of=3)
    p_ba = elo_win_prob(state, "2", "1", "Hard", best_of=3)
    assert abs(p_ab + p_ba - 1.0) < 1e-12

def test_challenger_uses_different_k():
    from copy import deepcopy
    from tml.features.elo import EloState, update_tournament

    base = EloState(mu0=1500, k_atp=32, k_challenger=16)
    base.set("1", "Hard", 1500)
    base.set("2", "Hard", 1500)
    row = {
        "player_a_id": "1",
        "player_b_id": "2",
        "y_complete_win": 1,
        "surface": "Hard",
        "best_of": 3,
    }
    atp = update_tournament(deepcopy(base), [{**row, "tour_level": "atp"}])
    ch = update_tournament(deepcopy(base), [{**row, "tour_level": "challenger"}])
    d_atp = atp.get("1", "Hard") - 1500
    d_ch = ch.get("1", "Hard") - 1500
    assert abs(d_atp / d_ch - 2.0) < 1e-9

def test_tournament_batch_uses_pre_batch_state_for_all_expected_scores():
    from tml.features.elo import EloState, rating_diffs_before_update, update_tournament

    state = EloState()
    state.set("1", "Hard", 1600)
    state.set("2", "Hard", 1500)
    state.set("3", "Hard", 1400)
    matches = [
        {"player_a_id": "1", "player_b_id": "2", "y_complete_win": 1, "surface": "Hard", "best_of": 3, "tour_level": "atp"},
        {"player_a_id": "1", "player_b_id": "3", "y_complete_win": 1, "surface": "Hard", "best_of": 3, "tour_level": "atp"},
    ]
    diffs = rating_diffs_before_update(state, matches)
    assert diffs[0] == 100.0
    assert diffs[1] == 200.0
    # second match must not see rating after first match within the same tournament batch
    update_tournament(state, matches)
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement Elo + symmetry helpers**

```python
# symmetry.py
import numpy as np

def logistic(s: float) -> float:
    return float(1.0 / (1.0 + np.exp(-s)))

def symmetrize_score(s_ab: float, s_ba: float) -> float:
    return 0.5 * (s_ab - s_ba)
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: surface Elo with tour-level K and antisymmetric win probs

EOF
)"
```

---

### Task 7: Frozen feature store and pre-tournament feature builder

**Files:**
- Create: `src/tml/features/builders.py`
- Create: `src/tml/features/store.py`
- Create: `tests/features/test_feature_store.py`
- Create: `tests/features/test_builders.py`

**Interfaces:**
- Produces:
  - `prediction_cutoff(tourney_date) -> date` = day before `tourney_date`
  - `build_feature_row(match_row, elo_state_at_cutoff, history_index) -> dict`
  - `FeatureStore.persist(rows) / load(path) / query(date_range) -> DataFrame`
  - Required persisted fields: `match_id`, `prediction_cutoff`, `prediction_regime`, `dataset_snapshot_id`, `feature_schema_version`, feature columns, `y_complete_win`

Feature vector v1 (difference form only): `elo_surface_diff`, `elo_overall_diff`, `rank_diff`, `rank_missing_a`, `rank_missing_b`, `form_diff`, `serve_1st_in_diff`, `serve_1st_won_diff`, `serve_2nd_won_diff`, `experience_diff`, `age_diff`, `best_of`, `is_challenger` **only as multiplier interaction later** — for v1 include `best_of` as context scaled into score via `best_of_centered * elo_surface_diff` interaction column `elo_x_bestof`, not a directional intercept.

**Do not** include within-tournament rest/load.

History index: prior **tournaments** only (`tourney_date`, `tourney_id`) strictly before current tournament start.

- [ ] **Step 1: Failing tests**

```python
def test_features_ignore_same_tournament_results():
    import pandas as pd
    from datetime import date
    from tml.features.builders import build_feature_row
    from tml.features.elo import EloState

    match = {
        "match_id": "atp:T2:1",
        "tourney_id": "T2",
        "tourney_date": date(2001, 1, 8),
        "player_a_id": "1",
        "player_b_id": "2",
        "surface": "Hard",
        "best_of": 3,
        "tour_level": "atp",
        "y_complete_win": 1,
    }
    # History incorrectly includes same tournament later round — builder must ignore it
    history = pd.DataFrame(
        [
            {
                "tourney_id": "T2",
                "tourney_date": date(2001, 1, 8),
                "player_a_id": "1",
                "player_b_id": "9",
                "y_complete_win": 1,
                "surface": "Hard",
                "tour_level": "atp",
                "completion_status": "completed",
            }
        ]
    )
    state = EloState()
    state.set("1", "Hard", 1500)
    state.set("2", "Hard", 1500)
    row_with = build_feature_row(match, state, history)
    row_without = build_feature_row(match, state, history.iloc[0:0])
    assert row_with["form_diff"] == row_without["form_diff"]
    assert row_with["experience_diff"] == row_without["experience_diff"]

def test_fit_uses_persisted_not_recomputed_state(tmp_path):
    from tml.features.store import FeatureStore

    store = FeatureStore(tmp_path / "features.parquet")
    store.persist(
        [
            {
                "match_id": "m1",
                "prediction_cutoff": "2000-12-31",
                "prediction_regime": "pre_tournament",
                "dataset_snapshot_id": "abc",
                "feature_schema_version": "v1",
                "elo_surface_diff": 10.0,
                "y_complete_win": 1,
            }
        ]
    )
    # Mutating any live Elo state must not change persisted value
    loaded = store.load()
    assert loaded.loc[loaded["match_id"] == "m1", "elo_surface_diff"].iloc[0] == 10.0
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement builders + parquet store under `data/processed/features/`**

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: pre-tournament feature builder and frozen parquet feature store

EOF
)"
```

---

### Task 8: Antisymmetric B0/B2 supervised models + OOF temperature

**Files:**
- Create: `src/tml/models/__init__.py`
- Create: `src/tml/models/ranking_logit.py`
- Create: `src/tml/models/supervised.py`
- Create: `src/tml/models/calibration.py`
- Create: `tests/models/test_supervised_symmetry.py`
- Create: `tests/models/test_calibration.py`

**Interfaces:**
- Produces:
  - `fit_b0(train_df) -> B0Model` using `rank_diff` (+ missing indicators); score `s = w·x` with no intercept (`fit_intercept=False`)
  - `fit_b2(train_df, feature_cols) -> B2Model` pipeline: median impute + scale fit on train; `LogisticRegression(C=1.0, penalty='l2', fit_intercept=False)`
  - `prequential_oof_scores(train_df, fit_fn) -> scores` by year folds inside training window
  - `fit_temperature(oof_scores, y) -> float T`
  - `predict_proba(model, X, T=1.0) -> p` via `logistic(s/T)`

- [ ] **Step 1: Failing tests**

```python
def test_b2_swap_players_complements_probability():
    import numpy as np
    import pandas as pd
    from tml.models.supervised import fit_b2, predict_proba

    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = (x > 0).astype(int)
    df = pd.DataFrame({"elo_surface_diff": x, "y_complete_win": y})
    model = fit_b2(df, feature_cols=["elo_surface_diff"])
    p = predict_proba(model, df[["elo_surface_diff"]])
    df_swap = df.assign(elo_surface_diff=-df["elo_surface_diff"])
    p_swap = predict_proba(model, df_swap[["elo_surface_diff"]])
    assert np.allclose(p + p_swap, 1.0, atol=1e-10)

def test_temperature_preserves_complement_symmetry():
    import numpy as np
    from tml.models.calibration import apply_temperature
    from tml.models.symmetry import logistic

    s = np.array([0.5, -1.0, 2.0])
    T = 1.7
    p = apply_temperature(s, T)
    p_swap = apply_temperature(-s, T)
    assert np.allclose(p + p_swap, 1.0, atol=1e-12)

def test_no_intercept_in_b2():
    import numpy as np
    import pandas as pd
    from tml.models.supervised import fit_b2

    df = pd.DataFrame(
        {
            "elo_surface_diff": [1.0, -1.0, 2.0, -2.0],
            "y_complete_win": [1, 0, 1, 0],
        }
    )
    model = fit_b2(df, feature_cols=["elo_surface_diff"])
    assert abs(float(model.intercept_)) < 1e-15
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: antisymmetric B0/B2 models with OOF temperature calibration

EOF
)"
```

---

### Task 9: Prequential walk-forward driver

**Files:**
- Create: `src/tml/validation/__init__.py`
- Create: `src/tml/validation/prequential.py`
- Create: `tests/validation/test_prequential.py`

**Interfaces:**
- Consumes: ingest → modeling table → Elo → feature store → B0/B1/B2
- Produces: `run_prequential(matches, config: PrequentialConfig) -> PrequentialResult` with predictions frame columns `match_id`, `year`, `p_b0`, `p_b1`, `p_b2`, `y`, `tour_level`, `surface`, `prediction_regime`, `dataset_snapshot_id`

Config defaults: burn_in_end=`1999-12-31`, elo_from=`2000`, supervised_from=`2005`, rolling_years=`5`, dev_era=`(2005,2018)`, final_era=`(2019, 2100)`.

Algorithm must match design §10: tournament order; persist features at cutoff; update Elo after tournament; fit supervised yearly on **frozen** rows.

- [ ] **Step 1: Failing test with synthetic timeline** (3 tournaments across 2 years; assert prediction for tournament 2 does not use its own results; assert B2 not produced before 2005 in config smoke with shortened years)

Use a mini config with years shifted for speed in unit tests (injectable year bounds).

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement `run_prequential`**

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: pre-tournament prequential walk-forward driver

EOF
)"
```

---

### Task 10: Metrics, common eligible set, block CIs

**Files:**
- Create: `src/tml/validation/metrics.py`
- Create: `src/tml/validation/blocks.py`
- Create: `tests/validation/test_metrics.py`
- Create: `tests/validation/test_blocks.py`

**Interfaces:**
- Produces:
  - `log_loss(y, p) -> float`
  - `brier(y, p) -> float`
  - `delta_log_loss(y, p_model, p_b1) -> float`
  - `common_eligible(df, cols) -> DataFrame` drop NA preds
  - `coverage_report(df, model_cols) -> dict`
  - `calibration_intercept_slope(y, p) -> tuple[float,float]` diagnostic
  - `moving_block_ci_delta(df, block_days=21, n_boot=500, seed=0) -> (low, high)`

Primary report helper: `final_era_primary_delta(preds, year_min=2019)` filters `prediction_regime=="pre_tournament"`.

- [ ] **Step 1: Failing tests** for Δ sign, eligible intersection, CI callable on tiny df

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: validation metrics, coverage, and moving-block CIs for delta log loss

EOF
)"
```

---

### Task 11: Full-pipeline uncertainty (thin v1)

**Files:**
- Create: `src/tml/validation/uncertainty.py`
- Create: `src/tml/viz/__init__.py`
- Create: `src/tml/viz/plots.py`
- Create: `tests/validation/test_uncertainty.py`

**Interfaces:**
- Produces: `pipeline_block_bootstrap_p(history_matches, target_match_ids, B=50, block_days=21, seed=0) -> dict[match_id, list[float]]`  
  Implementation may subsample history for speed in tests (`B=5`). Must refit Elo chronologically on resampled tournament blocks; supervised optional flag `include_supervised=False` for fast B1-only intervals in v1 default plots.

- [ ] **Step 1: Failing test** — bootstrap returns `B` probabilities in (0,1); swapped target complements within float tolerance for Elo-only mode

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement bootstrap + simple matplotlib helpers for interval strip and joint rating scatter from draws

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: pipeline block-bootstrap intervals and joint ability plot helpers

EOF
)"
```

---

### Task 12: Odds client, atomic storage, closing rules, multiplicative no-vig

**skills.md binds:** §04 (secrets), §06 (validate API JSON with pydantic), §07 (no fake store), §10 (`Result` for unmatched / missing close), §17

**Files:**
- Create: `src/tml/odds/__init__.py`
- Create: `src/tml/odds/devig.py`
- Create: `src/tml/odds/storage.py`
- Create: `src/tml/odds/closing.py`
- Create: `src/tml/odds/client.py`
- Create: `src/tml/odds/matching.py`
- Create: `tests/odds/test_devig.py`
- Create: `tests/odds/test_closing.py`
- Create: `tests/odds/test_matching.py`
- Modify: `.env.example` only if new vars appear

**Interfaces:**
- `multiplicative_novig(price_a: float, price_b: float, odds_format="american") -> Result[tuple[float, float], str]`
- `QuoteRecord` pydantic model: both sides + `last_update` + `collected_at` + schedule fields
- `select_closing_quote(...) -> Result[QuoteRecord, str]` — Err if none valid (expected)
- `fetch_h2h(sport_key="tennis_atp")` uses `get_settings().parlay_api_key`; **never logs the key**; returns `Result`; HTTP failures exceptional or Err with safe message
- `match_players(...) -> Result[MatchedEvent, list[UnmatchedPlayer]]`
- Persist quotes to real parquet/JSON under temp dirs in tests (not an in-memory fake schema)

- [ ] **Step 1: Failing tests** for no-vig sums to 1; closing rejects `collected_at > cutoff`; matching requires opponent context; settings-missing key → `Err` without raising secret material

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement** (monkeypatch HTTP transport only — not the quote store format)

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: §17 walk + commit**

```bash
git commit -m "$(cat <<'EOF'
feat: ParlayAPI odds snapshots with validated quotes and dual-timestamp closing

EOF
)"
```

---

### Task 13: Docs + notebook stubs + README

**Files:**
- Create: `docs/sources.md`
- Create: `docs/schema.md`
- Create: `docs/leakage_audit.md`
- Create: `docs/model_card.md`
- Create: `README.md`
- Create: `notebooks/01_ingest_and_qc.ipynb` (calls package; no duplicated business logic)
- Create: `notebooks/02_features_and_leakage_audit.ipynb`
- Create: `notebooks/03_baselines.ipynb`
- Create: `notebooks/04_walkforward_validation.ipynb`
- Create: `notebooks/05_uncertainty_plots.ipynb`
- Create: `notebooks/06_forward_odds.ipynb`

**Interfaces:** Docs must state estimand, `pre_tournament` regime, symmetry, K-by-level, frozen no-vig, observed-vs-executable prices, market mismatch.

- [ ] **Step 1: Write docs from design sections (no placeholders)**

- [ ] **Step 2: Add README with install, `pytest`, and how to run ingest on `tml-data/`**

- [ ] **Step 3: Create notebooks that import `tml` and outline cells**

- [ ] **Step 4: Run full unit suite**

Run: `python -m pytest -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
docs: add sources, schema, leakage audit, model card, and research notebooks

EOF
)"
```

---

### Task 14: Integration smoke on real `tml-data` (manual gate)

**Files:**
- Create: `scripts/smoke_ingest.py`
- Create: `scripts/smoke_prequential_sample.py`

**Interfaces:** CLI scripts that:
1. Ingest a small year range (e.g. 2018–2019 ATP + Challenger) from `tml-data/`
2. Run prequential with injectable bounds
3. Print coverage + Δ on a smoke window (not final-era claim)

- [ ] **Step 1: Write scripts using package APIs only**

- [ ] **Step 2: Run ingest smoke**

Run: `python scripts/smoke_ingest.py --start 2018 --end 2019`  
Expected: prints row counts, completion breakdown, snapshot id

- [ ] **Step 3: Run short prequential smoke**

Run: `python scripts/smoke_prequential_sample.py --start 2018 --end 2019`  
Expected: writes `data/processed/smoke_preds.parquet` and metric summary without error

- [ ] **Step 4: Commit scripts + any bugfixes**

```bash
git commit -m "$(cat <<'EOF'
feat: add real-data smoke scripts for ingest and short prequential run

EOF
)"
```

---

## Spec coverage checklist

| Spec area | Task(s) |
|-----------|---------|
| Estimand / completion labels | 1, 4 |
| Manifest snapshot hashing | 2, 3 |
| ATP+Challenger ingest + lineage | 3, 14 |
| Hash orientation + modeling table | 4 |
| Ranking `available_at` | 5 |
| Pre-tournament regime / no within-T rest | 7, 9 |
| Elo K-by-level + antisymmetric B1 | 6 |
| Frozen feature store | 7, 9 |
| B0/B2 + OOF temperature | 8 |
| Prequential loop | 9, 14 |
| Metrics, coverage, block CI, final-era Δ | 10 |
| Pipeline bootstrap uncertainty + viz | 11 |
| Odds observed prices, dual cutoff, no-vig | 12 |
| Docs / notebooks / model card | 13 |
| Identity map versioning | 4, 12 |
| Docs / notebooks / model card | 13 |

## Notes for implementers

- **`skills.md` wins on style** (Result, invariant, config boundary, tests-with-code, small commits, no silent errors). Design doc wins on domain semantics. If they conflict on a hard stop (secrets, etc.), stop and ask.
- Keep business logic out of notebooks.
- Do not tune on 2019+ during Tasks 1–14; leave hyperparameter search as a later optional task using 2005–2018 only.
- GBM/ensemble (B2b/B3) intentionally omitted from this plan (dev-era optional follow-up).
- If Sapling is unavailable, record `vcs_revision=None`, `dirty=None` rather than inventing git metadata.
- Ask before adding dependencies beyond those listed in Task 1 / Tech Stack.
- Prefer comments that explain *why* (e.g. why `tourney_date` batching is pre-tournament) — never narrate the diff.
