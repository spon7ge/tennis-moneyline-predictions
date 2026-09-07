from __future__ import annotations

import pandas as pd

from tml.data.identity import PlayerIdentityMap
from tml.data.orientation import assign_orientation

CONTEXT_COLUMNS = (
    "match_id",
    "tourney_id",
    "tourney_date",
    "surface",
    "best_of",
    "tour_level",
    "dataset_snapshot_id",
)
ORIENTED_FEATURE_COLUMNS = (
    ("rank", "winner_rank", "loser_rank"),
    ("age", "winner_age", "loser_age"),
    ("svpt", "w_svpt", "l_svpt"),
    ("1stIn", "w_1stIn", "l_1stIn"),
    ("1stWon", "w_1stWon", "l_1stWon"),
    ("2ndWon", "w_2ndWon", "l_2ndWon"),
)


def build_modeling_table(
    matches: pd.DataFrame, identity: PlayerIdentityMap
) -> pd.DataFrame:
    completed = matches.loc[matches["completion_status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame(
            columns=[
                *CONTEXT_COLUMNS,
                "player_a_id",
                "player_b_id",
                "y_complete_win",
                "prediction_regime",
                "identity_map_version",
            ]
        )

    rows: list[dict[str, object]] = []
    for _, match in completed.iterrows():
        winner_id = identity.resolve(
            str(match["winner_id"]),
            None if pd.isna(match.get("winner_name")) else str(match["winner_name"]),
        )
        loser_id = identity.resolve(
            str(match["loser_id"]),
            None if pd.isna(match.get("loser_name")) else str(match["loser_name"]),
        )
        player_a_id, player_b_id = assign_orientation(
            str(match["match_id"]), winner_id, loser_id
        )
        winner_is_a = winner_id == player_a_id
        row: dict[str, object] = {
            column: match[column]
            for column in CONTEXT_COLUMNS
            if column in match.index
        }
        row.update(
            {
                "player_a_id": player_a_id,
                "player_b_id": player_b_id,
                "y_complete_win": int(winner_is_a),
                "prediction_regime": "pre_tournament",
                "identity_map_version": identity.version,
            }
        )
        for target, winner_column, loser_column in ORIENTED_FEATURE_COLUMNS:
            if winner_column not in match.index or loser_column not in match.index:
                continue
            row[f"{target}_a" if target in {"rank", "age"} else f"a_{target}"] = (
                match[winner_column] if winner_is_a else match[loser_column]
            )
            row[f"{target}_b" if target in {"rank", "age"} else f"b_{target}"] = (
                match[loser_column] if winner_is_a else match[winner_column]
            )
        rows.append(row)

    return pd.DataFrame(rows)
