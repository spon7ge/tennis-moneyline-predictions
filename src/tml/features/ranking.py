from __future__ import annotations

import math
from datetime import date
from typing import Any, Mapping

import pandas as pd

RANK_SOURCE_MATCH_EMBEDDED = "match_embedded"


def embedded_rank_available_at(tourney_date: date) -> date:
    """Match-embedded pre-match ranks are usable at the tournament start date."""
    return tourney_date


def _parse_rank(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        value = stripped
    if pd.isna(value):
        return None
    try:
        rank = int(float(value))
    except (TypeError, ValueError):
        return None
    if rank < 1:
        return None
    return rank


def _row_value(row: Mapping[str, Any] | pd.Series, key: str) -> Any:
    if key in row:
        return row[key]
    return None


def rank_feature_pair(
    row: Mapping[str, Any] | pd.Series, tourney_date: date
) -> dict[str, Any]:
    """Build antisymmetric ranking features from oriented embedded ranks.

    ``rank_diff = log(rank_b) - log(rank_a)`` when both ranks are present and
    >= 1; otherwise ``rank_diff`` is ``NaN``. Embedded ranks use
    ``available_at = tourney_date`` (pre-tournament cutoff).
    """
    rank_a = _parse_rank(_row_value(row, "rank_a"))
    rank_b = _parse_rank(_row_value(row, "rank_b"))
    rank_missing_a = rank_a is None
    rank_missing_b = rank_b is None

    if rank_missing_a or rank_missing_b:
        rank_diff: float | None = math.nan
    else:
        rank_diff = math.log(rank_b) - math.log(rank_a)

    return {
        "rank_diff": rank_diff,
        "rank_missing_a": rank_missing_a,
        "rank_missing_b": rank_missing_b,
        "rank_source": RANK_SOURCE_MATCH_EMBEDDED,
        "rank_staleness_days": 0,
    }
