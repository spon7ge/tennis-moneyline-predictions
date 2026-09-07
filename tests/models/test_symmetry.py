import pytest

from tml.features.elo import EloState
from tml.models.elo_prob import (
    antisymmetric_logit_from_diff,
    elo_win_prob,
)
from tml.models.symmetry import logistic, symmetrize_score


def test_logistic_complements_for_negated_scores():
    assert logistic(2.5) + logistic(-2.5) == pytest.approx(1.0, abs=1e-15)


def test_symmetrize_score_is_antisymmetric_when_players_swap():
    score_ab = symmetrize_score(3.0, -1.0)
    score_ba = symmetrize_score(-1.0, 3.0)

    assert score_ab == 2.0
    assert score_ba == -score_ab


def test_elo_logit_has_standard_400_point_scale():
    assert logistic(antisymmetric_logit_from_diff(400)) == pytest.approx(10 / 11)


def test_elo_probability_is_antisymmetric():
    state = EloState()
    state.set("1", "Hard", 1600)
    state.set("2", "Hard", 1500)
    p_ab = elo_win_prob(state, "1", "2", "Hard", best_of=3)
    p_ba = elo_win_prob(state, "2", "1", "Hard", best_of=3)
    assert abs(p_ab + p_ba - 1.0) < 1e-12


def test_bo5_scales_elo_logit_by_frozen_factor():
    state = EloState()
    state.set("1", "Grass", 1600)
    state.set("2", "Grass", 1500)

    bo3_logit = antisymmetric_logit_from_diff(100)
    bo5_probability = elo_win_prob(state, "1", "2", "Grass", best_of=5)

    assert bo5_probability == pytest.approx(logistic(1.1 * bo3_logit))
