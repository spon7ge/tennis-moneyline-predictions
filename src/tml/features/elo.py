from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
import math
from typing import Any

from tml.models.symmetry import logistic
from tml.shared.invariant import invariant

ELO_SCALE = 400.0 / math.log(10.0)
BO5_DIFF_MULTIPLIER = 1.1


@dataclass
class EloState:
    """Surface-specific Elo ratings shared across ATP and Challenger."""

    mu0: float = 1500.0
    k_atp: float = 32.0
    k_challenger: float = 20.0
    ratings: dict[tuple[Hashable, str], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        invariant(math.isfinite(self.mu0), "mu0 must be finite")
        invariant(
            math.isfinite(self.k_atp) and self.k_atp >= 0,
            "ATP K-factor must be finite and non-negative",
        )
        invariant(
            math.isfinite(self.k_challenger) and self.k_challenger >= 0,
            "Challenger K-factor must be finite and non-negative",
        )

    def set(self, player_id: Hashable, surface: str, rating: float) -> None:
        invariant(math.isfinite(rating), "ratings must be finite")
        self.ratings[(player_id, surface)] = float(rating)

    def get(self, player_id: Hashable, surface: str) -> float:
        rating = self.ratings.get((player_id, surface), self.mu0)
        invariant(math.isfinite(rating), "ratings must be finite")
        return float(rating)


def _best_of_multiplier(best_of: int) -> float:
    invariant(best_of in (3, 5), "best_of must be 3 or 5")
    return BO5_DIFF_MULTIPLIER if best_of == 5 else 1.0


def expected_score(r_a: float, r_b: float, best_of: int) -> float:
    """Return player A's Elo expectancy."""
    invariant(
        math.isfinite(r_a) and math.isfinite(r_b),
        "ratings must be finite",
    )
    score = _best_of_multiplier(best_of) * (r_a - r_b) / ELO_SCALE
    return logistic(score)


def rating_diff(
    state: EloState,
    player_a: Hashable,
    player_b: Hashable,
    surface: str,
) -> float:
    return state.get(player_a, surface) - state.get(player_b, surface)


def _records(matches: Any) -> list[Mapping[str, Any]]:
    if hasattr(matches, "to_dict"):
        records = matches.to_dict(orient="records")
    else:
        records = list(matches)
    invariant(
        all(isinstance(row, Mapping) for row in records),
        "matches must contain row mappings",
    )
    return records


def rating_diffs_before_update(
    state: EloState,
    matches: Iterable[Mapping[str, Any]] | Any,
) -> list[float]:
    """Compute every match differential from the same frozen state."""
    return [
        rating_diff(
            state,
            row["player_a_id"],
            row["player_b_id"],
            row["surface"],
        )
        for row in _records(matches)
    ]


def _k_factor(state: EloState, tour_level: object) -> float:
    level = str(tour_level).casefold()
    invariant(level in {"atp", "challenger"}, "unknown tour level")
    return state.k_atp if level == "atp" else state.k_challenger


def update_tournament(
    state: EloState,
    matches_df: Iterable[Mapping[str, Any]] | Any,
) -> EloState:
    """Apply one simultaneous update after scoring a tournament batch."""
    matches = _records(matches_df)
    diffs = rating_diffs_before_update(state, matches)
    deltas: defaultdict[tuple[Hashable, str], float] = defaultdict(float)

    for row, diff in zip(matches, diffs, strict=True):
        outcome = row["y_complete_win"]
        invariant(outcome in (0, 1), "completed outcome must be 0 or 1")
        expected = expected_score(diff, 0.0, int(row["best_of"]))
        delta = _k_factor(state, row["tour_level"]) * (float(outcome) - expected)
        surface = row["surface"]
        deltas[(row["player_a_id"], surface)] += delta
        deltas[(row["player_b_id"], surface)] -= delta

    prior_ratings = {
        key: state.get(player_id=key[0], surface=key[1]) for key in deltas
    }
    for (player_id, surface), delta in deltas.items():
        state.set(player_id, surface, prior_ratings[(player_id, surface)] + delta)
    return state
