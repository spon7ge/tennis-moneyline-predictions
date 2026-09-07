import numpy as np
import pandas as pd
import pytest

from tml.models.calibration import (
    apply_temperature,
    fit_temperature,
    prequential_oof_scores,
)


def test_temperature_preserves_complement_symmetry() -> None:
    scores = np.array([0.5, -1.0, 2.0])

    p = apply_temperature(scores, 1.7)
    p_swapped = apply_temperature(-scores, 1.7)

    assert np.allclose(p + p_swapped, 1.0, atol=1e-12)


def test_fit_temperature_recovers_known_softening() -> None:
    scores = np.linspace(-3.0, 3.0, 20_001)
    probabilities = apply_temperature(scores, 2.0)
    rng = np.random.default_rng(4)
    outcomes = rng.binomial(1, probabilities)

    temperature = fit_temperature(scores, outcomes)

    assert temperature == pytest.approx(2.0, rel=0.08)


def test_prequential_oof_scores_use_prior_years_only() -> None:
    frame = pd.DataFrame(
        {
            "tourney_date": pd.to_datetime(
                ["2000-01-01", "2000-06-01", "2001-01-01", "2002-01-01"]
            ),
            "signal_diff": [-1.0, 1.0, 2.0, 3.0],
            "y_complete_win": [0, 1, 1, 1],
        }
    )
    training_years: list[tuple[int, ...]] = []

    class RecordingModel:
        def __init__(self, score: float) -> None:
            self.score = score

        def decision_function(self, rows: pd.DataFrame) -> np.ndarray:
            return np.full(len(rows), self.score)

    def fit_fn(train: pd.DataFrame) -> RecordingModel:
        years = tuple(sorted(pd.to_datetime(train["tourney_date"]).dt.year.unique()))
        training_years.append(years)
        return RecordingModel(float(len(train)))

    scores = prequential_oof_scores(frame, fit_fn)

    assert training_years == [(2000,), (2000, 2001)]
    assert np.isnan(scores[:2]).all()
    assert scores[2:].tolist() == [2.0, 3.0]


def test_fit_temperature_ignores_unscored_initial_fold() -> None:
    scores = np.array([np.nan, np.nan, -2.0, 2.0])
    y = np.array([0, 1, 0, 1])

    assert fit_temperature(scores, y) < 1.0


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.nan, np.inf])
def test_apply_temperature_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(np.array([0.0]), temperature)
