from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smoke_ingest import ingest_year_range
from smoke_prequential_sample import (
    select_smoke_tournaments,
    smoke_metric_summary,
)


REQUIRED_MATCH = {
    "tourney_name": "Tiny Open",
    "surface": "Hard",
    "tourney_level": "A",
    "winner_name": "Winner",
    "loser_name": "Loser",
    "score": "6-4 6-4",
    "best_of": 3,
    "round": "R32",
    "winner_rank": 10,
    "loser_rank": 20,
}


def _write_match(path: Path, tourney_id: str, match_num: int) -> None:
    pd.DataFrame(
        [
            {
                **REQUIRED_MATCH,
                "tourney_id": tourney_id,
                "tourney_date": 20180101,
                "match_num": match_num,
                "winner_id": f"{match_num}01",
                "loser_id": f"{match_num}02",
            }
        ]
    ).to_csv(path, index=False)


def test_ingest_year_range_combines_atp_and_challenger_snapshot(
    tmp_path: Path,
) -> None:
    _write_match(tmp_path / "2018.csv", "2018-001", 1)
    _write_match(tmp_path / "2018_challenger.csv", "2018-101", 2)

    result = ingest_year_range(tmp_path, 2018, 2018)

    assert len(result.matches) == 2
    assert set(result.matches["tour_level"]) == {"atp", "challenger"}
    assert len(result.manifest) == 2
    assert result.matches["dataset_snapshot_id"].eq(result.snapshot_id).all()


def test_ingest_year_range_rejects_reversed_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start year"):
        ingest_year_range(tmp_path, 2019, 2018)


def test_smoke_metric_summary_reports_coverage_and_non_final_delta() -> None:
    predictions = pd.DataFrame(
        {
            "y": [1, 0],
            "p_b0": [0.7, 0.3],
            "p_b1": [0.6, 0.4],
            "p_b2": [0.8, 0.2],
            "prediction_regime": ["pre_tournament", "pre_tournament"],
        }
    )

    summary = smoke_metric_summary(predictions)

    assert summary["n_common"] == 2
    assert summary["delta_log_loss_b2_minus_b1"] < 0
    assert summary["claim_scope"] == "smoke-window diagnostic; not final-era results"


def test_select_smoke_tournaments_bounds_each_year_and_level() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": f"{level}-{year}-{tournament}",
                "tourney_id": f"{year}-{tournament}",
                "tourney_date": f"{year}-01-{tournament + 1:02d}",
                "tour_level": level,
            }
            for year in (2018, 2019)
            for level in ("atp", "challenger")
            for tournament in range(3)
        ]
    )

    selected = select_smoke_tournaments(matches, per_level_year=2)

    counts = selected.groupby(
        [pd.to_datetime(selected["tourney_date"]).dt.year, "tour_level"]
    )["tourney_id"].nunique()
    assert counts.eq(2).all()
