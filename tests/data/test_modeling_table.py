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
