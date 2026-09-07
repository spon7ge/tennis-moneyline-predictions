from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tml.data.completion import parse_completion_status
from tml.data.manifest import build_manifest, snapshot_id


REQUIRED_COLUMNS = frozenset(
    {
        "tourney_id",
        "tourney_name",
        "tourney_date",
        "surface",
        "tourney_level",
        "match_num",
        "winner_id",
        "loser_id",
        "winner_name",
        "loser_name",
        "score",
        "best_of",
        "round",
        "winner_rank",
        "loser_rank",
    }
)
IDENTIFIER_COLUMNS = ("tourney_id", "winner_id", "loser_id")


class ValidationError(ValueError):
    """Raised when an ingest boundary receives invalid input."""


@dataclass(frozen=True, slots=True)
class IngestResult:
    matches: pd.DataFrame
    manifest: list[dict[str, Any]]
    dataset_snapshot_id: str
    ingested_at: datetime


def _validate_inputs(paths: list[Path], root: Path) -> tuple[list[Path], Path]:
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise ValidationError(f"Root directory does not exist: {root}")
    if not paths:
        raise ValidationError("At least one match file is required")

    resolved_paths: list[Path] = []
    for path in paths:
        resolved_path = path.resolve()
        if not resolved_path.is_file():
            raise ValidationError(f"Match file does not exist: {path}")
        if not resolved_path.is_relative_to(resolved_root):
            raise ValidationError(f"Match file is outside root directory: {path}")
        resolved_paths.append(resolved_path)
    return resolved_paths, resolved_root


def _parse_tourney_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    compact = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[compact] = pd.to_datetime(
        text.loc[compact], format="%Y%m%d", errors="coerce"
    )
    parsed.loc[~compact] = pd.to_datetime(text.loc[~compact], errors="coerce")
    if parsed.isna().any():
        invalid = sorted(text.loc[parsed.isna()].dropna().unique().tolist())
        raise ValidationError(f"Invalid tourney_date values: {invalid}")
    return parsed.dt.date


def _read_match_file(path: Path, root: Path, tour_level: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={column: "string" for column in IDENTIFIER_COLUMNS})
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValidationError(f"{path.name} is missing required columns: {missing}")

    frame["tourney_date"] = _parse_tourney_dates(frame["tourney_date"])
    for column in IDENTIFIER_COLUMNS:
        frame[column] = frame[column].astype("string")
    frame["tour_level"] = tour_level
    frame["completion_status"] = frame["score"].map(
        lambda score: parse_completion_status(None if pd.isna(score) else str(score))
    )
    frame["source_file"] = path.relative_to(root).as_posix()
    frame["match_id"] = (
        tour_level
        + ":"
        + frame["tourney_id"]
        + ":"
        + frame["match_num"].astype("string")
    )
    return frame


def ingest_match_files(
    paths: list[Path], root: Path, tour_level: str
) -> IngestResult:
    """Load Sackmann-style match CSVs and attach versioned lineage fields."""
    resolved_paths, resolved_root = _validate_inputs(paths, root)
    manifest = build_manifest(resolved_paths, resolved_root)
    matches = pd.concat(
        [
            _read_match_file(path, resolved_root, tour_level)
            for path in resolved_paths
        ],
        ignore_index=True,
    )
    return IngestResult(
        matches=matches,
        manifest=manifest,
        dataset_snapshot_id=snapshot_id(manifest),
        ingested_at=datetime.now().astimezone(),
    )
