from copy import deepcopy

import pytest

from tml.features.elo import (
    EloState,
    expected_score,
    rating_diff,
    rating_diffs_before_update,
    update_tournament,
)


def test_elo_state_defaults_and_surface_ratings():
    state = EloState()

    assert state.mu0 == 1500
    assert state.k_atp == 32
    assert state.k_challenger == 20
    assert state.get("new-player", "Hard") == 1500

    state.set("1", "Hard", 1600)
    assert state.get("1", "Hard") == 1600
    assert state.get("1", "Clay") == 1500


def test_expected_score_uses_bo5_adjustment():
    bo3 = expected_score(1600, 1500, best_of=3)
    bo5 = expected_score(1600, 1500, best_of=5)

    assert bo5 > bo3
    assert expected_score(1500, 1600, best_of=5) == pytest.approx(1.0 - bo5)


def test_rating_diff_uses_requested_surface():
    state = EloState()
    state.set("1", "Hard", 1600)
    state.set("2", "Hard", 1500)

    assert rating_diff(state, "1", "2", "Hard") == 100.0
    assert rating_diff(state, "2", "1", "Hard") == -100.0
    assert rating_diff(state, "1", "2", "Clay") == 0.0


def test_challenger_uses_different_k():
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


def test_tournament_update_moves_overall_ratings():
    state = EloState()
    row = {
        "player_a_id": "1",
        "player_b_id": "2",
        "y_complete_win": 1,
        "surface": "Hard",
        "best_of": 3,
        "tour_level": "atp",
    }

    update_tournament(state, [row])

    assert state.get("1", "Overall") > state.mu0
    assert state.get("2", "Overall") < state.mu0
    assert rating_diff(state, "1", "2", "Overall") > 0.0


def test_tournament_batch_uses_pre_batch_state_for_all_expected_scores():
    state = EloState()
    state.set("1", "Hard", 1600)
    state.set("2", "Hard", 1500)
    state.set("3", "Hard", 1400)
    matches = [
        {
            "player_a_id": "1",
            "player_b_id": "2",
            "y_complete_win": 1,
            "surface": "Hard",
            "best_of": 3,
            "tour_level": "atp",
        },
        {
            "player_a_id": "1",
            "player_b_id": "3",
            "y_complete_win": 1,
            "surface": "Hard",
            "best_of": 3,
            "tour_level": "atp",
        },
    ]
    diffs = rating_diffs_before_update(state, matches)
    assert diffs[0] == 100.0
    assert diffs[1] == 200.0

    expected_delta = 32 * (
        (1 - expected_score(1600, 1500, 3))
        + (1 - expected_score(1600, 1400, 3))
    )
    update_tournament(state, matches)
    assert state.get("1", "Hard") == pytest.approx(1600 + expected_delta)


def test_elo_state_rejects_non_finite_rating():
    state = EloState()

    with pytest.raises(RuntimeError, match="ratings must be finite"):
        state.set("1", "Hard", float("nan"))
