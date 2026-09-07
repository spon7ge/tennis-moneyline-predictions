from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tml.validation.metrics import (
    brier,
    calibration_intercept_slope,
    common_eligible,
    coverage_report,
    delta_log_loss,
    final_era_primary_delta,
    log_loss,
)


def test_log_loss_and_brier_on_known_probabilities() -> None:
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])

    expected_log_loss = float(
        np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
    )
    expected_brier = float(np.mean((p - y) ** 2))

    assert log_loss(y, p) == pytest.approx(expected_log_loss)
    assert brier(y, p) == pytest.approx(expected_brier)


def test_delta_log_loss_negative_when_model_beats_baseline() -> None:
    y = np.array([1, 0, 1, 0])
    p_b1 = np.array([0.5, 0.5, 0.5, 0.5])
    p_model = np.array([0.9, 0.1, 0.9, 0.1])

    delta = delta_log_loss(y, p_model, p_b1)

    assert delta < 0
    assert delta == pytest.approx(log_loss(y, p_model) - log_loss(y, p_b1))


def test_common_eligible_keeps_only_rows_with_all_model_predictions() -> None:
    frame = pd.DataFrame(
        {
            "p_b0": [0.6, np.nan, 0.7, np.nan],
            "p_b1": [0.5, 0.5, np.nan, 0.45],
            "p_b2": [0.55, 0.45, 0.65, 0.50],
            "y": [1, 0, 1, 0],
        }
    )

    eligible = common_eligible(frame, ["p_b0", "p_b1", "p_b2"])

    assert eligible.index.tolist() == [0]
    assert len(eligible) == 1


def test_coverage_report_counts_common_eligible_rows() -> None:
    frame = pd.DataFrame(
        {
            "p_b0": [0.6, np.nan, 0.7],
            "p_b1": [0.5, 0.5, 0.4],
            "p_b2": [0.55, 0.45, np.nan],
            "y": [1, 0, 1],
        }
    )

    report = coverage_report(frame, ["p_b0", "p_b1", "p_b2"])

    assert report["n_total"] == 3
    assert report["n_common"] == 1
    assert report["coverage"]["p_b0"] == pytest.approx(2 / 3)
    assert report["coverage"]["p_b1"] == pytest.approx(1.0)
    assert report["coverage"]["p_b2"] == pytest.approx(2 / 3)


def test_calibration_intercept_slope_recovers_near_identity() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 1.0, size=300)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    outcomes = rng.binomial(1, probabilities)

    intercept, slope = calibration_intercept_slope(outcomes, probabilities)

    assert intercept == pytest.approx(0.0, abs=0.35)
    assert slope == pytest.approx(1.0, abs=0.35)


def test_final_era_primary_delta_filters_regime_and_year() -> None:
    preds = pd.DataFrame(
        {
            "year": [2018, 2019, 2019, 2020],
            "prediction_regime": [
                "pre_tournament",
                "pre_tournament",
                "in_sample",
                "pre_tournament",
            ],
            "p_b0": [0.6, 0.6, 0.6, 0.6],
            "p_b1": [0.5, 0.5, 0.5, 0.5],
            "p_b2": [0.55, 0.55, 0.55, 0.55],
            "y": [1, 1, 1, 0],
        }
    )

    report = final_era_primary_delta(preds, year_min=2019)

    assert report["regime"] == "pre_tournament"
    assert report["year_min"] == 2019
    assert report["n"] == 2
    assert "delta_b2" in report
    assert report["delta_b2"] == pytest.approx(
        delta_log_loss(
            preds.loc[[1, 3], "y"].to_numpy(),
            preds.loc[[1, 3], "p_b2"].to_numpy(),
            preds.loc[[1, 3], "p_b1"].to_numpy(),
        )
    )


def test_log_loss_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        log_loss(np.array([1, 0]), np.array([1.1, 0.5]))
