from __future__ import annotations

from collections.abc import Iterable
from typing import Hashable

import numpy as np
import pandas as pd

from tml.features.builders import build_feature_row
from tml.features.elo import EloState, update_tournament
from tml.models.elo_prob import elo_win_prob
from tml.models.supervised import B2Model, fit_b2, predict_proba
from tml.validation.prequential import DEFAULT_B2_FEATURE_COLUMNS

_REQUIRED_COLUMNS = {
    "match_id",
    "tourney_id",
    "tourney_date",
    "player_a_id",
    "player_b_id",
    "surface",
    "best_of",
    "tour_level",
    "y_complete_win",
}


def _validated_matches(history_matches: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(history_matches, pd.DataFrame):
        raise TypeError("history_matches must be a pandas DataFrame")
    missing = sorted(_REQUIRED_COLUMNS.difference(history_matches.columns))
    if missing:
        raise ValueError(
            f"history_matches missing required columns: {', '.join(missing)}"
        )
    if history_matches["match_id"].duplicated().any():
        raise ValueError("match_id must be unique")

    frame = history_matches.copy(deep=True)
    frame["tourney_date"] = pd.to_datetime(
        frame["tourney_date"], errors="coerce"
    )
    if frame["tourney_date"].isna().any():
        raise ValueError("tourney_date must contain valid dates")
    if not frame["best_of"].isin((3, 5)).all():
        raise ValueError("best_of must contain only 3 or 5")
    completed = frame.get(
        "completion_status",
        pd.Series("completed", index=frame.index),
    ).eq("completed")
    if not frame.loc[completed, "y_complete_win"].isin((0, 1)).all():
        raise ValueError("completed y_complete_win must contain only 0 and 1")
    return frame.sort_values(
        ["tourney_date", "tourney_id"], kind="stable"
    ).reset_index(drop=True)


def _calendar_blocks(
    history: pd.DataFrame, block_days: int
) -> list[pd.DataFrame]:
    if history.empty:
        return []
    origin = history["tourney_date"].min().normalize()
    offsets = (
        history["tourney_date"].dt.normalize() - origin
    ).dt.days.to_numpy(dtype=int)
    block_ids = offsets // block_days
    return [
        history.loc[block_ids == block_id].copy()
        for block_id in np.unique(block_ids)
    ]


def _sampled_tournaments(
    blocks: list[pd.DataFrame],
    rng: np.random.Generator,
) -> list[pd.DataFrame]:
    if not blocks:
        return []
    sampled_indices = rng.integers(0, len(blocks), size=len(blocks))
    tournaments: list[pd.DataFrame] = []
    synthetic_date = pd.Timestamp("2000-01-01")
    for block_index in sampled_indices:
        block = blocks[int(block_index)]
        grouped = block.groupby(
            ["tourney_date", "tourney_id"], sort=False, dropna=False
        )
        for _, tournament in grouped:
            replay = tournament.copy()
            replay["tourney_date"] = synthetic_date
            tournaments.append(replay)
            synthetic_date += pd.Timedelta(days=1)
    return tournaments


def _replay_pipeline(
    tournaments: list[pd.DataFrame],
    *,
    include_supervised: bool,
) -> tuple[EloState, pd.DataFrame, B2Model | None]:
    state = EloState()
    replay_history = pd.DataFrame()
    frozen_rows: list[dict[str, object]] = []
    for tournament in tournaments:
        if include_supervised:
            frozen_rows.extend(
                build_feature_row(match, state, replay_history)
                for _, match in tournament.iterrows()
            )
        update_tournament(state, tournament)
        replay_history = pd.concat(
            [replay_history, tournament], ignore_index=True
        )

    model = None
    if include_supervised:
        model = fit_b2(
            pd.DataFrame(frozen_rows),
            DEFAULT_B2_FEATURE_COLUMNS,
        )
    return state, replay_history, model


def _target_probability(
    target: pd.Series,
    state: EloState,
    replay_history: pd.DataFrame,
    model: B2Model | None,
) -> float:
    if model is None:
        return elo_win_prob(
            state,
            target["player_a_id"],
            target["player_b_id"],
            str(target["surface"]),
            int(target["best_of"]),
        )

    scoring_target = target.copy()
    scoring_target["tourney_date"] = (
        replay_history["tourney_date"].max() + pd.Timedelta(days=1)
        if not replay_history.empty
        else pd.Timestamp("2000-01-01")
    )
    features = pd.DataFrame(
        [build_feature_row(scoring_target, state, replay_history)]
    )
    return float(predict_proba(model, features)[0])


def _bootstrap_states(
    history_matches: pd.DataFrame,
    target_match_ids: Iterable[Hashable],
    B: int,
    block_days: int,
    seed: int,
    *,
    include_supervised: bool,
) -> tuple[
    list[Hashable],
    list[tuple[pd.Series, EloState, pd.DataFrame, B2Model | None]],
]:
    if isinstance(B, bool) or not isinstance(B, int) or B < 1:
        raise ValueError("B must be a positive integer")
    if (
        isinstance(block_days, bool)
        or not isinstance(block_days, int)
        or block_days < 1
    ):
        raise ValueError("block_days must be a positive integer")

    frame = _validated_matches(history_matches)
    ids = list(target_match_ids)
    if not ids:
        raise ValueError("target_match_ids must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError("target_match_ids must be unique")
    indexed = frame.set_index("match_id", drop=False)
    missing_targets = [
        match_id for match_id in ids if match_id not in indexed.index
    ]
    if missing_targets:
        raise ValueError(f"unknown target_match_ids: {missing_targets}")

    targets = indexed.loc[ids]
    if isinstance(targets, pd.Series):
        targets = targets.to_frame().T

    draws: list[tuple[pd.Series, EloState, pd.DataFrame, B2Model | None]] = []
    rng = np.random.default_rng(seed)

    for target_date, cutoff_targets in targets.groupby(
        "tourney_date", sort=True
    ):
        completed = frame.get(
            "completion_status",
            pd.Series("completed", index=frame.index),
        ).eq("completed")
        history = frame.loc[
            (frame["tourney_date"] < target_date) & completed
        ]
        blocks = _calendar_blocks(history, block_days)
        for _ in range(B):
            tournaments = _sampled_tournaments(blocks, rng)
            state, replay_history, model = _replay_pipeline(
                tournaments,
                include_supervised=include_supervised,
            )
            for _, target in cutoff_targets.iterrows():
                draws.append((target, state, replay_history, model))
    return ids, draws


def pipeline_block_bootstrap_p(
    history_matches: pd.DataFrame,
    target_match_ids: Iterable[Hashable],
    B: int = 50,
    block_days: int = 21,
    seed: int = 0,
    *,
    include_supervised: bool = False,
) -> dict[Hashable, list[float]]:
    """Refit chronological pipeline draws from resampled calendar blocks.

    Calendar blocks are sampled with replacement and stitched in draw order.
    Every tournament is replayed as a batch through a fresh Elo state.
    With ``include_supervised=True``, B2 is also refit from replay-frozen
    features; the default returns the faster Elo-only B1 draws.
    """
    ids, bootstrap_draws = _bootstrap_states(
        history_matches,
        target_match_ids,
        B,
        block_days,
        seed,
        include_supervised=include_supervised,
    )
    draws: dict[Hashable, list[float]] = {match_id: [] for match_id in ids}
    for target, state, replay_history, model in bootstrap_draws:
        probability = _target_probability(
            target, state, replay_history, model
        )
        draws[target["match_id"]].append(float(probability))
    return draws


def pipeline_block_bootstrap_joint_ratings(
    history_matches: pd.DataFrame,
    target_match_ids: Iterable[Hashable],
    B: int = 50,
    block_days: int = 21,
    seed: int = 0,
) -> dict[Hashable, list[tuple[float, float]]]:
    """Joint surface-Elo ratings (R_A, R_B) from the same pipeline bootstrap draws.

    Ratings come from the Elo state after each resampled history replay, on the
    target match's surface. Pairs are joint — not independent marginal samples.
    """
    ids, bootstrap_draws = _bootstrap_states(
        history_matches,
        target_match_ids,
        B,
        block_days,
        seed,
        include_supervised=False,
    )
    draws: dict[Hashable, list[tuple[float, float]]] = {
        match_id: [] for match_id in ids
    }
    for target, state, _replay_history, _model in bootstrap_draws:
        surface = str(target["surface"])
        rating_a = float(state.get(target["player_a_id"], surface))
        rating_b = float(state.get(target["player_b_id"], surface))
        draws[target["match_id"]].append((rating_a, rating_b))
    return draws
