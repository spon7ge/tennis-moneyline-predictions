from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_FEATURE_PATH = Path("data/processed/features/features.parquet")
REQUIRED_COLUMNS = {
    "match_id",
    "prediction_cutoff",
    "prediction_regime",
    "dataset_snapshot_id",
    "feature_schema_version",
    "y_complete_win",
}


class FeatureStore:
    """Append-only parquet storage for frozen prediction-time feature rows."""

    def __init__(self, path: str | Path = DEFAULT_FEATURE_PATH) -> None:
        self.path = Path(path)

    @staticmethod
    def _validate(frame: pd.DataFrame) -> None:
        missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"feature rows missing required fields: {', '.join(missing)}")
        if frame[list(REQUIRED_COLUMNS)].isna().any().any():
            null_columns = sorted(
                column
                for column in REQUIRED_COLUMNS
                if frame[column].isna().any()
            )
            raise ValueError(
                f"feature rows contain null required fields: {', '.join(null_columns)}"
            )
        if not frame["prediction_regime"].eq("pre_tournament").all():
            raise ValueError("prediction_regime must be pre_tournament")
        if frame["match_id"].duplicated().any():
            raise ValueError("match_id must be unique")

    def persist(self, rows: Iterable[Mapping[str, Any]] | pd.DataFrame) -> Path:
        incoming = (
            rows.copy(deep=True)
            if isinstance(rows, pd.DataFrame)
            else pd.DataFrame([dict(row) for row in rows])
        )
        self._validate(incoming)
        incoming["prediction_cutoff"] = pd.to_datetime(
            incoming["prediction_cutoff"], errors="raise"
        )

        if self.path.exists():
            stored = self.load()
            combined = pd.concat([stored, incoming], ignore_index=True)
            if combined["match_id"].duplicated().any():
                raise ValueError("persist would overwrite a frozen match_id")
        else:
            combined = incoming

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            combined.to_parquet(temporary, index=False, engine="pyarrow")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return self.path

    def load(self, path: str | Path | None = None) -> pd.DataFrame:
        source = self.path if path is None else Path(path)
        return pd.read_parquet(source, engine="pyarrow")

    def query(
        self, date_range: tuple[date | str, date | str]
    ) -> pd.DataFrame:
        if len(date_range) != 2:
            raise ValueError("date_range must contain (start, end)")
        start, end = (pd.Timestamp(value) for value in date_range)
        if start > end:
            raise ValueError("date_range start must be on or before end")
        frame = self.load()
        cutoffs = pd.to_datetime(frame["prediction_cutoff"], errors="raise")
        return frame.loc[cutoffs.between(start, end)].reset_index(drop=True)
