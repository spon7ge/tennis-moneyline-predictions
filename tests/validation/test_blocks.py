from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tml.validation.blocks import moving_block_ci_delta


def _synthetic_eval_frame(n_blocks: int = 6, rows_per_block: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_blocks, freq="21D")
    rows: list[dict[str, object]] = []
    for block_idx, tourney_date in enumerate(dates):
        for row_idx in range(rows_per_block):
            offset = 0.02 * row_idx
            rows.append(
                {
                    "tourney_date": tourney_date,
                    "y": (block_idx + row_idx) % 2,
                    "p_b1": 0.50 + offset,
                    "p_b2": 0.55 + offset,
                }
            )
    return pd.DataFrame(rows)


def test_moving_block_ci_delta_returns_ordered_interval_on_tiny_frame() -> None:
    frame = _synthetic_eval_frame(n_blocks=3, rows_per_block=2)

    low, high = moving_block_ci_delta(
        frame,
        block_days=21,
        n_boot=50,
        seed=0,
    )

    assert low <= high
    assert np.isfinite(low)
    assert np.isfinite(high)


def test_moving_block_ci_delta_is_reproducible_with_seed() -> None:
    frame = _synthetic_eval_frame()

    first = moving_block_ci_delta(frame, block_days=21, n_boot=100, seed=7)
    second = moving_block_ci_delta(frame, block_days=21, n_boot=100, seed=7)

    assert first == pytest.approx(second)


def test_moving_block_ci_delta_requires_tourney_date() -> None:
    frame = pd.DataFrame(
        {
            "y": [1, 0],
            "p_b1": [0.5, 0.5],
            "p_b2": [0.6, 0.4],
        }
    )

    with pytest.raises(ValueError, match="tourney_date"):
        moving_block_ci_delta(frame)
