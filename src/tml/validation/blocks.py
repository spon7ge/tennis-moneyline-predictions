from __future__ import annotations

import numpy as np
import pandas as pd

from tml.validation.metrics import delta_log_loss

_REQUIRED_COLUMNS = ("tourney_date", "y", "p_b1", "p_b2")


def _validated_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"df missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("df must not be empty")

    frame = df.copy(deep=True)
    frame["tourney_date"] = pd.to_datetime(frame["tourney_date"], errors="coerce")
    if frame["tourney_date"].isna().any():
        raise ValueError("tourney_date must contain valid dates")
    return frame.sort_values("tourney_date", kind="stable").reset_index(drop=True)


def _calendar_blocks(dates: pd.Series, block_days: int) -> list[np.ndarray]:
    if block_days < 1:
        raise ValueError("block_days must be at least 1")

    origin = dates.min().normalize()
    day_offsets = (dates.dt.normalize() - origin).dt.days.to_numpy(dtype=int)
    block_ids = day_offsets // block_days
    unique_blocks = np.unique(block_ids)
    return [
        np.flatnonzero(block_ids == block_id).astype(int) for block_id in unique_blocks
    ]


def moving_block_ci_delta(
    df: pd.DataFrame,
    *,
    block_days: int = 21,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Paired moving-block bootstrap CI for mean delta log loss (B2 minus B1)."""
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    frame = _validated_frame(df)
    blocks = _calendar_blocks(frame["tourney_date"], block_days)
    if not blocks:
        raise ValueError("no calendar blocks available")

    y = frame["y"].to_numpy(dtype=float)
    p_b1 = frame["p_b1"].to_numpy(dtype=float)
    p_b2 = frame["p_b2"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    n_blocks = len(blocks)
    for draw in range(n_boot):
        sampled = rng.integers(0, n_blocks, size=n_blocks)
        indices = np.concatenate([blocks[block_index] for block_index in sampled])
        deltas[draw] = delta_log_loss(y[indices], p_b2[indices], p_b1[indices])

    low, high = np.quantile(deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)
