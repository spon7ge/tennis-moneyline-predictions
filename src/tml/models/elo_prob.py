import math
from collections.abc import Hashable

from tml.features.elo import BO5_DIFF_MULTIPLIER, ELO_SCALE, EloState, rating_diff
from tml.models.symmetry import logistic
from tml.shared.invariant import invariant


def antisymmetric_logit_from_diff(
    diff: float,
    scale: float = ELO_SCALE,
) -> float:
    """Convert an Elo difference to an antisymmetric logistic score."""
    invariant(
        math.isfinite(diff) and math.isfinite(scale) and scale > 0,
        "Elo difference and positive scale must be finite",
    )
    return float(diff / scale)


def elo_win_prob(
    state: EloState,
    player_a: Hashable,
    player_b: Hashable,
    surface: str,
    best_of: int,
) -> float:
    """Return a structurally antisymmetric surface-Elo win probability."""
    invariant(best_of in (3, 5), "best_of must be 3 or 5")
    diff = rating_diff(state, player_a, player_b, surface)
    if best_of == 5:
        diff *= BO5_DIFF_MULTIPLIER
    return logistic(antisymmetric_logit_from_diff(diff))
