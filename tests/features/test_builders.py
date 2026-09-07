from datetime import date

import pandas as pd

from tml.features.builders import build_feature_row, prediction_cutoff
from tml.features.elo import EloState


def _match(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "match_id": "atp:T2:1",
        "tourney_id": "T2",
        "tourney_date": date(2001, 1, 8),
        "player_a_id": "1",
        "player_b_id": "2",
        "surface": "Hard",
        "best_of": 3,
        "tour_level": "atp",
        "rank_a": 10,
        "rank_b": 20,
        "age_a": 24.0,
        "age_b": 26.0,
        "dataset_snapshot_id": "snapshot-1",
        "y_complete_win": 1,
    }
    row.update(overrides)
    return row


def test_prediction_cutoff_is_day_before_tournament() -> None:
    assert prediction_cutoff(date(2001, 1, 1)) == date(2000, 12, 31)


def test_features_ignore_same_tournament_results() -> None:
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

    row_with = build_feature_row(_match(), state, history)
    row_without = build_feature_row(_match(), state, history.iloc[0:0])

    assert row_with["form_diff"] == row_without["form_diff"]
    assert row_with["experience_diff"] == row_without["experience_diff"]


def test_feature_row_uses_only_prior_tournaments_and_difference_features() -> None:
    history = pd.DataFrame(
        [
            {
                "tourney_id": "T1",
                "tourney_date": date(2001, 1, 1),
                "player_a_id": "1",
                "player_b_id": "3",
                "y_complete_win": 1,
                "completion_status": "completed",
            },
            {
                "tourney_id": "T3",
                "tourney_date": date(2001, 1, 15),
                "player_a_id": "2",
                "player_b_id": "3",
                "y_complete_win": 1,
                "completion_status": "completed",
            },
        ]
    )
    state = EloState()
    state.set("1", "Hard", 1600)
    state.set("2", "Hard", 1500)
    state.set("1", "Overall", 1550)
    state.set("2", "Overall", 1525)

    row = build_feature_row(_match(best_of=5), state, history)

    assert row["prediction_cutoff"] == date(2001, 1, 7)
    assert row["prediction_regime"] == "pre_tournament"
    assert row["feature_schema_version"] == "v1"
    assert row["elo_surface_diff"] == 100.0
    assert row["elo_overall_diff"] == 25.0
    assert row["elo_x_bestof"] == 200.0
    assert row["form_diff"] == 1.0
    assert row["experience_diff"] == 1
    assert row["age_diff"] == -2.0
    assert "rest_diff" not in row
    assert "load_diff" not in row
