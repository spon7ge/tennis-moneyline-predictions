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
        row: dict[str, object] = {
            column: match[column]
            for column in CONTEXT_COLUMNS
            if column in match.index
        }
        row.update(
            {
                "player_a_id": player_a_id,
                "player_b_id": player_b_id,
                "y_complete_win": int(winner_id == player_a_id),
                "prediction_regime": "pre_tournament",
                "identity_map_version": identity.version,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)
