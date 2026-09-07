import math

from tml.shared.invariant import invariant


def logistic(s: float) -> float:
    """Map a finite score to a probability without overflowing."""
    invariant(math.isfinite(s), "logistic score must be finite")
    if s >= 0:
        return float(1.0 / (1.0 + math.exp(-s)))
    exp_s = math.exp(s)
    return float(exp_s / (1.0 + exp_s))


def symmetrize_score(s_ab: float, s_ba: float) -> float:
    """Project directional scores onto an antisymmetric score."""
    invariant(
        math.isfinite(s_ab) and math.isfinite(s_ba),
        "scores must be finite",
    )
    return 0.5 * (s_ab - s_ba)
