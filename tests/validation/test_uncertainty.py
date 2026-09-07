from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tml.validation.uncertainty import (
    pipeline_block_bootstrap_joint_ratings,
    pipeline_block_bootstrap_p,
)
from tml.viz.plots import (
    plot_joint_rating_scatter,
    plot_probability_interval_strip,
)


def _history_with_swapped_targets() -> pd.DataFrame:
    rows = [
        ("h1", "t1", "2020-01-01", "a", "b", 1),
        ("h2", "t2", "2020-01-22", "a", "c", 1),
        ("h3", "t3", "2020-02-12", "b", "c", 1),
        ("target-ab", "t4", "2020-03-04", "a", "b", 0),
        ("target-ba", "t4", "2020-03-04", "b", "a", 1),
    ]
    return pd.DataFrame(
        [
            {
                "match_id": match_id,
                "tourney_id": tourney_id,
                "tourney_date": tourney_date,
                "player_a_id": player_a,
                "player_b_id": player_b,
                "surface": "Hard",
                "best_of": 3,
                "tour_level": "atp",
                "y_complete_win": outcome,
            }
            for match_id, tourney_id, tourney_date, player_a, player_b, outcome in rows
        ]
    )


def test_pipeline_bootstrap_returns_b_strict_probabilities() -> None:
    draws = pipeline_block_bootstrap_p(
        _history_with_swapped_targets(),
        ["target-ab"],
        B=5,
        block_days=21,
        seed=4,
    )

    assert len(draws["target-ab"]) == 5
    assert all(0.0 < probability < 1.0 for probability in draws["target-ab"])


def test_elo_only_swapped_targets_are_drawwise_complements() -> None:
    draws = pipeline_block_bootstrap_p(
        _history_with_swapped_targets(),
        ["target-ab", "target-ba"],
        B=5,
        include_supervised=False,
        seed=7,
    )

    for p_ab, p_ba in zip(
        draws["target-ab"], draws["target-ba"], strict=True
    ):
        assert p_ab == pytest.approx(1.0 - p_ba)


def test_pipeline_bootstrap_joint_ratings_are_paired_and_length_b() -> None:
    ratings = pipeline_block_bootstrap_joint_ratings(
        _history_with_swapped_targets(),
        ["target-ab"],
        B=5,
        block_days=21,
        seed=4,
    )

    assert len(ratings["target-ab"]) == 5
    for rating_a, rating_b in ratings["target-ab"]:
        assert rating_a == pytest.approx(float(rating_a))
        assert rating_b == pytest.approx(float(rating_b))
        assert abs(rating_a - 1500.0) < 500.0
        assert abs(rating_b - 1500.0) < 500.0


def test_probability_interval_strip_plots_intervals_on_p() -> None:
    ax = plot_probability_interval_strip(
        {"match-a": [0.55, 0.60, 0.65], "match-b": [0.30, 0.40, 0.50]}
    )

    assert ax.get_xlabel() == "Win probability"
    assert tuple(round(limit, 8) for limit in ax.get_xlim()) == (0.0, 1.0)
    assert len(ax.lines) == 2
    plt.close(ax.figure)


def test_joint_rating_scatter_preserves_paired_draws() -> None:
    ax = plot_joint_rating_scatter(
        [(1510.0, 1490.0), (1530.0, 1470.0)],
        player_a_label="A",
        player_b_label="B",
    )

    offsets = ax.collections[0].get_offsets()
    assert offsets.tolist() == [[1510.0, 1490.0], [1530.0, 1470.0]]
    assert ax.get_xlabel() == "A rating"
    assert ax.get_ylabel() == "B rating"
    plt.close(ax.figure)
