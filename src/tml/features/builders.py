from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from tml.features.elo import EloState, rating_diff
from tml.features.ranking import rank_feature_pair

FEATURE_SCHEMA_VERSION = "v1"
PREDICTION_REGIME = "pre_tournament"
OVERALL_SURFACE = "Overall"


def _as_date(value: object, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a valid date") from error


def prediction_cutoff(tourney_date: date) -> date:
    """Return the historical pre-tournament prediction cutoff."""
    return _as_date(tourney_date, "tourney_date") - timedelta(days=1)


def _records(history: Iterable[Mapping[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(history, pd.DataFrame):
        return history.to_dict(orient="records")
    records = list(history)
    if not all(isinstance(row, Mapping) for row in records):
        raise TypeError("history_index must contain row mappings")
    return [dict(row) for row in records]


def _prior_tournament_history(
    history: Iterable[Mapping[str, Any]] | pd.DataFrame,
    current_tourney_id: object,
    current_tourney_date: date,
) -> list[dict[str, Any]]:
    prior: list[dict[str, Any]] = []
    for row in _records(history):
        if row.get("completion_status", "completed") != "completed":
            continue
        if str(row.get("tourney_id")) == str(current_tourney_id):
            continue
        row_date = _as_date(row.get("tourney_date"), "history tourney_date")
        if row_date < current_tourney_date:
            prior.append(row)
    return prior


def _player_results(
    rows: list[dict[str, Any]], player_id: object
) -> list[tuple[dict[str, Any], str, float]]:
    player = str(player_id)
    results: list[tuple[dict[str, Any], str, float]] = []
    for row in rows:
        outcome = row.get("y_complete_win")
        if outcome not in (0, 1):
            continue
        if str(row.get("player_a_id")) == player:
            results.append((row, "a", float(outcome)))
        elif str(row.get("player_b_id")) == player:
            results.append((row, "b", 1.0 - float(outcome)))
    return results


def _first_present(row: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = row.get(name)
        if value is None or pd.isna(value):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _serve_rates(row: Mapping[str, Any], slot: str) -> tuple[float | None, ...]:
    direct_1st_in = _first_present(
        row, (f"serve_1st_in_{slot}", f"player_{slot}_serve_1st_in")
    )
    direct_1st_won = _first_present(
        row, (f"serve_1st_won_{slot}", f"player_{slot}_serve_1st_won")
    )
    direct_2nd_won = _first_present(
        row, (f"serve_2nd_won_{slot}", f"player_{slot}_serve_2nd_won")
    )
    points = _first_present(row, (f"{slot}_svpt", f"svpt_{slot}"))
    first_in = _first_present(row, (f"{slot}_1stIn", f"first_in_{slot}"))
    first_won = _first_present(row, (f"{slot}_1stWon", f"first_won_{slot}"))
    second_won = _first_present(row, (f"{slot}_2ndWon", f"second_won_{slot}"))

    if direct_1st_in is None and points and first_in is not None:
        direct_1st_in = first_in / points
    if direct_1st_won is None and first_in and first_won is not None:
        direct_1st_won = first_won / first_in
    second_points = None if points is None or first_in is None else points - first_in
    if direct_2nd_won is None and second_points and second_won is not None:
        direct_2nd_won = second_won / second_points
    return direct_1st_in, direct_1st_won, direct_2nd_won


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _player_history_features(
    rows: list[dict[str, Any]], player_id: object
) -> tuple[float, int, float, float, float]:
    results = _player_results(rows, player_id)
    wins = [result for _, _, result in results]
    serve_rates = [_serve_rates(row, slot) for row, slot, _ in results]
    serve_means = [
        _mean([rates[index] for rates in serve_rates if rates[index] is not None])
        for index in range(3)
    ]
    return _mean(wins) if wins else 0.0, len(results), *serve_means


def _difference(a: float, b: float) -> float:
    return a - b if math.isfinite(a) and math.isfinite(b) else math.nan


def _age(row: Mapping[str, Any], slot: str) -> float:
    value = _first_present(row, (f"age_{slot}", f"player_{slot}_age"))
    return value if value is not None else math.nan


def build_feature_row(
    match_row: Mapping[str, Any] | pd.Series,
    elo_state_at_cutoff: EloState,
    history_index: Iterable[Mapping[str, Any]] | pd.DataFrame,
) -> dict[str, Any]:
    """Build one frozen feature row using only prior-tournament information."""
    match = dict(match_row)
    required = {
        "match_id",
        "tourney_id",
        "tourney_date",
        "player_a_id",
        "player_b_id",
        "surface",
        "best_of",
        "tour_level",
        "y_complete_win",
    }
    missing = sorted(required.difference(match))
    if missing:
        raise ValueError(f"match_row missing required fields: {', '.join(missing)}")

    tourney_date = _as_date(match["tourney_date"], "tourney_date")
    best_of = int(match["best_of"])
    if best_of not in (3, 5):
        raise ValueError("best_of must be 3 or 5")
    prior = _prior_tournament_history(
        history_index, match["tourney_id"], tourney_date
    )
    player_a = match["player_a_id"]
    player_b = match["player_b_id"]
    a_history = _player_history_features(prior, player_a)
    b_history = _player_history_features(prior, player_b)
    surface_diff = rating_diff(
        elo_state_at_cutoff, player_a, player_b, str(match["surface"])
    )
    ranks = rank_feature_pair(match, tourney_date)

    return {
        "match_id": match["match_id"],
        "tourney_id": match["tourney_id"],
        "tourney_date": tourney_date,
        "prediction_cutoff": prediction_cutoff(tourney_date),
        "prediction_regime": PREDICTION_REGIME,
        "dataset_snapshot_id": match.get("dataset_snapshot_id"),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "elo_surface_diff": surface_diff,
        "elo_overall_diff": rating_diff(
            elo_state_at_cutoff, player_a, player_b, OVERALL_SURFACE
        ),
        "rank_diff": ranks["rank_diff"],
        "rank_missing_a": ranks["rank_missing_a"],
        "rank_missing_b": ranks["rank_missing_b"],
        "form_diff": a_history[0] - b_history[0],
        "serve_1st_in_diff": _difference(a_history[2], b_history[2]),
        "serve_1st_won_diff": _difference(a_history[3], b_history[3]),
        "serve_2nd_won_diff": _difference(a_history[4], b_history[4]),
        "experience_diff": a_history[1] - b_history[1],
        "age_diff": _difference(_age(match, "a"), _age(match, "b")),
        "best_of": best_of,
        "is_challenger": str(match["tour_level"]).casefold() == "challenger",
        "elo_x_bestof": (best_of - 3) * surface_diff,
        "y_complete_win": int(match["y_complete_win"]),
    }
