import pandas as pd

from tml.data.identity import PlayerIdentityMap
from tml.data.modeling_table import build_modeling_table


def test_modeling_table_excludes_retirements_and_sets_label() -> None:
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


def test_modeling_table_orients_prediction_features_and_keeps_snapshot() -> None:
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
                "dataset_snapshot_id": "snapshot-1",
                "winner_rank": 10,
                "loser_rank": 20,
                "winner_age": 25.0,
                "loser_age": 27.0,
                "w_svpt": 60,
                "l_svpt": 70,
                "w_1stIn": 40,
                "l_1stIn": 45,
                "w_1stWon": 30,
                "l_1stWon": 25,
                "w_2ndWon": 10,
                "l_2ndWon": 12,
            }
        ]
    )

    row = build_modeling_table(
        matches, PlayerIdentityMap(version="v1")
    ).iloc[0]
    winner_is_a = row["y_complete_win"] == 1

    assert row["dataset_snapshot_id"] == "snapshot-1"
    assert row["rank_a"] == (10 if winner_is_a else 20)
    assert row["rank_b"] == (20 if winner_is_a else 10)
    assert row["age_a"] == (25.0 if winner_is_a else 27.0)
    assert row["age_b"] == (27.0 if winner_is_a else 25.0)
    assert row["a_svpt"] == (60 if winner_is_a else 70)
    assert row["b_svpt"] == (70 if winner_is_a else 60)
    assert row["a_1stIn"] == (40 if winner_is_a else 45)
    assert row["b_2ndWon"] == (12 if winner_is_a else 10)
