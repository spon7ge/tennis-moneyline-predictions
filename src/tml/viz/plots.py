from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def _axes(ax: Axes | None) -> Axes:
    return ax if ax is not None else plt.subplots()[1]


def plot_probability_interval_strip(
    probability_draws: Mapping[object, Sequence[float]],
    *,
    alpha: float = 0.05,
    ax: Axes | None = None,
) -> Axes:
    """Plot bootstrap quantile intervals for p, without Bernoulli noise."""
    if not probability_draws:
        raise ValueError("probability_draws must not be empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    resolved = _axes(ax)
    labels = [str(label) for label in probability_draws]
    for position, draws in enumerate(probability_draws.values()):
        values = np.asarray(draws, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("each probability draw sequence must be non-empty")
        if not np.isfinite(values).all() or not (
            (values > 0.0) & (values < 1.0)
        ).all():
            raise ValueError("probability draws must be finite and between 0 and 1")
        low, median, high = np.quantile(
            values, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0]
        )
        resolved.plot([low, high], [position, position], marker="|")
        resolved.scatter([median], [position], color="black", zorder=3)

    resolved.set_yticks(range(len(labels)), labels)
    resolved.set_xlim(0.0, 1.0)
    resolved.set_xlabel("Win probability")
    resolved.set_title("Bootstrap uncertainty in estimated win probability")
    return resolved


def plot_joint_rating_scatter(
    rating_draws: Sequence[tuple[float, float]],
    *,
    player_a_label: str = "Player A",
    player_b_label: str = "Player B",
    ax: Axes | None = None,
) -> Axes:
    """Plot paired player ratings from the same pipeline bootstrap draws."""
    values = np.asarray(rating_draws, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 2:
        raise ValueError("rating_draws must be a non-empty sequence of pairs")
    if not np.isfinite(values).all():
        raise ValueError("rating draws must be finite")

    resolved = _axes(ax)
    resolved.scatter(values[:, 0], values[:, 1], alpha=0.65)
    resolved.set_xlabel(f"{player_a_label} rating")
    resolved.set_ylabel(f"{player_b_label} rating")
    resolved.set_title("Joint ability draws")
    return resolved
