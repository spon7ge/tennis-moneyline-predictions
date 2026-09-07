from datetime import date

import pytest

from tml.features.store import FeatureStore


def _row(match_id: str, cutoff: str, elo_diff: float) -> dict[str, object]:
    return {
        "match_id": match_id,
        "prediction_cutoff": cutoff,
        "prediction_regime": "pre_tournament",
        "dataset_snapshot_id": "abc",
        "feature_schema_version": "v1",
        "elo_surface_diff": elo_diff,
        "y_complete_win": 1,
    }


def test_fit_uses_persisted_not_recomputed_state(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    store = FeatureStore(path)
    rows = [_row("m1", "2000-12-31", 10.0)]
    store.persist(rows)

    rows[0]["elo_surface_diff"] = 999.0
    loaded = store.load()

    assert loaded.loc[loaded["match_id"] == "m1", "elo_surface_diff"].iloc[0] == 10.0
    assert path.read_bytes()[:4] == b"PAR1"


def test_load_can_read_an_explicit_parquet_path(tmp_path) -> None:
    first = FeatureStore(tmp_path / "first.parquet")
    first.persist([_row("m1", "2000-12-31", 10.0)])

    loaded = FeatureStore(tmp_path / "unused.parquet").load(first.path)

    assert loaded["match_id"].tolist() == ["m1"]


def test_query_filters_inclusive_prediction_cutoff_range(tmp_path) -> None:
    store = FeatureStore(tmp_path / "features.parquet")
    store.persist(
        [
            _row("m1", "2000-12-31", 10.0),
            _row("m2", "2001-01-07", 20.0),
            _row("m3", "2001-01-14", 30.0),
        ]
    )

    result = store.query((date(2001, 1, 1), date(2001, 1, 10)))

    assert result["match_id"].tolist() == ["m2"]


def test_persist_rejects_missing_required_metadata(tmp_path) -> None:
    store = FeatureStore(tmp_path / "features.parquet")

    with pytest.raises(ValueError, match="dataset_snapshot_id"):
        store.persist(
            [
                {
                    "match_id": "m1",
                    "prediction_cutoff": "2000-12-31",
                    "prediction_regime": "pre_tournament",
                    "feature_schema_version": "v1",
                    "elo_surface_diff": 10.0,
                    "y_complete_win": 1,
                }
            ]
        )


def test_persist_rejects_non_pre_tournament_rows(tmp_path) -> None:
    store = FeatureStore(tmp_path / "features.parquet")
    row = _row("m1", "2000-12-31", 10.0)
    row["prediction_regime"] = "pre_match"

    with pytest.raises(ValueError, match="pre_tournament"):
        store.persist([row])
