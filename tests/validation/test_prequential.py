from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import tml.validation.prequential as prequential
from tml.features.elo import EloState, update_tournament
from tml.features.store import FeatureStore
from tml.models.elo_prob import elo_win_prob
from tml.validation.prequential import PrequentialConfig, run_prequential


def _match(
    match_id: str,
    tourney_id: str,
    tourney_date: date,
    player_a: str,
    player_b: str,
    outcome: int,
    rank_a: int,
    rank_b: int,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "tourney_id": tourney_id,
        "tourney_date": tourney_date,
        "player_a_id": player_a,
        "player_b_id": player_b,
        "surface": "Hard",
        "best_of": 3,
        "tour_level": "atp",
        "rank_a": rank_a,
        "rank_b": rank_b,
        "y_complete_win": outcome,
        "dataset_snapshot_id": "synthetic-v1",
    }


def _timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _match("m1", "T1", date(2001, 1, 1), "A", "B", 1, 10, 20),
            _match("m2", "T2", date(2001, 2, 1), "A", "C", 1, 10, 30),
            _match("m3", "T2", date(2001, 2, 1), "A", "C", 0, 10, 30),
            _match("m4", "T3", date(2002, 1, 1), "A", "B", 1, 8, 25),
        ]
    )


def test_tournament_predictions_use_pre_tournament_elo_state(tmp_path) -> None:
    config = PrequentialConfig(
        burn_in_end=date(2000, 12, 31),
        elo_from=2001,
        supervised_from=2002,
        rolling_years=1,
        dev_era=(2002, 2002),
        final_era=(2003, 2100),
        feature_store_path=tmp_path / "features.parquet",
        b2_feature_cols=("elo_surface_diff",),
    )

    result = run_prequential(_timeline().sample(frac=1, random_state=4), config)
    predictions = result.predictions.set_index("match_id")

    state_after_t1 = EloState()
    update_tournament(state_after_t1, _timeline().iloc[[0]])
    expected_t2 = elo_win_prob(state_after_t1, "A", "C", "Hard", 3)

    assert predictions.loc["m2", "p_b1"] == pytest.approx(expected_t2)
    assert predictions.loc["m3", "p_b1"] == pytest.approx(expected_t2)
    assert np.isnan(predictions.loc["m2", "p_b2"])
    assert 0.0 < predictions.loc["m4", "p_b2"] < 1.0
    assert predictions.loc["m4", "year"] == 2002


def test_supervised_fit_uses_persisted_original_cutoff_features(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    config = PrequentialConfig(
        burn_in_end=date(2000, 12, 31),
        elo_from=2001,
        supervised_from=2002,
        rolling_years=1,
        feature_store_path=path,
        b2_feature_cols=("elo_surface_diff",),
    )

    result = run_prequential(_timeline(), config)
    frozen = FeatureStore(path).load().set_index("match_id")

    assert result.feature_store_path == path
    assert frozen.loc["m2", "elo_surface_diff"] == pytest.approx(16.0)
    assert frozen.loc["m3", "elo_surface_diff"] == pytest.approx(16.0)
    assert frozen.loc["m4", "elo_surface_diff"] != pytest.approx(16.0)


def test_yearly_supervised_fit_uses_oof_temperatures(tmp_path, monkeypatch) -> None:
    oof_calls: list[int] = []
    fitted_temperatures = iter((2.0, 3.0))
    prediction_temperatures: list[float] = []
    real_predict_proba = prequential.predict_proba

    def fake_oof_scores(train: pd.DataFrame, fit_fn: object) -> np.ndarray:
        oof_calls.append(len(train))
        return np.zeros(len(train), dtype=float)

    def fake_fit_temperature(scores: np.ndarray, y: np.ndarray) -> float:
        assert len(scores) == len(y)
        return next(fitted_temperatures)

    def recording_predict_proba(
        model: object, frame: pd.DataFrame, T: float = 1.0
    ) -> np.ndarray:
        prediction_temperatures.append(T)
        return real_predict_proba(model, frame, T=T)

    monkeypatch.setattr(prequential, "prequential_oof_scores", fake_oof_scores)
    monkeypatch.setattr(prequential, "fit_temperature", fake_fit_temperature)
    monkeypatch.setattr(prequential, "predict_proba", recording_predict_proba)

    run_prequential(
        _timeline(),
        PrequentialConfig(
            burn_in_end=date(2000, 12, 31),
            elo_from=2001,
            supervised_from=2002,
            rolling_years=1,
            feature_store_path=tmp_path / "features.parquet",
            b2_feature_cols=("elo_surface_diff",),
        ),
    )

    assert oof_calls == [3, 3]
    assert prediction_temperatures == [2.0, 3.0]


def test_output_schema_and_model_start_years_are_explicit(tmp_path) -> None:
    result = run_prequential(
        _timeline(),
        PrequentialConfig(
            burn_in_end=date(2000, 12, 31),
            elo_from=2002,
            supervised_from=2002,
            rolling_years=1,
            feature_store_path=tmp_path / "features.parquet",
            b2_feature_cols=("elo_surface_diff",),
        ),
    )

    assert result.predictions.columns.tolist() == [
        "match_id",
        "tourney_date",
        "year",
        "p_b0",
        "p_b1",
        "p_b2",
        "y",
        "tour_level",
        "surface",
        "prediction_regime",
        "dataset_snapshot_id",
    ]
    before_start = result.predictions["year"] < 2002
    unavailable = result.predictions.loc[before_start, ["p_b0", "p_b1", "p_b2"]]
    assert unavailable.isna().all().all()
    assert result.predictions["prediction_regime"].eq("pre_tournament").all()
