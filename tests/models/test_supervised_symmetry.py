import numpy as np
import pandas as pd
import pytest

from tml.models.supervised import fit_b0, fit_b2, predict_proba


def test_b2_swap_players_complements_probability() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame(
        {"elo_surface_diff": x, "y_complete_win": (x > 0).astype(int)}
    )

    model = fit_b2(df, feature_cols=["elo_surface_diff"])
    p = predict_proba(model, df[["elo_surface_diff"]])
    swapped = df.assign(elo_surface_diff=-df["elo_surface_diff"])
    p_swapped = predict_proba(model, swapped[["elo_surface_diff"]])

    assert np.allclose(p + p_swapped, 1.0, atol=1e-10)


def test_no_intercept_in_b2() -> None:
    df = pd.DataFrame(
        {
            "elo_surface_diff": [1.0, -1.0, 2.0, -2.0],
            "y_complete_win": [1, 0, 1, 0],
        }
    )

    model = fit_b2(df, feature_cols=["elo_surface_diff"])

    assert abs(float(model.intercept_)) < 1e-15
    assert model.fit_intercept is False
    assert model.C == 1.0
    assert model.penalty == "l2"


def test_b2_imputer_and_scaler_are_fit_on_training_data_only() -> None:
    train = pd.DataFrame(
        {
            "elo_surface_diff": [1.0, 3.0, np.nan, -1.0],
            "y_complete_win": [1, 1, 0, 0],
        }
    )
    model = fit_b2(train, feature_cols=["elo_surface_diff"])

    predict_proba(model, pd.DataFrame({"elo_surface_diff": [10_000.0, np.nan]}))

    assert model.imputer.statistics_[0] == pytest.approx(1.0)
    assert model.scaler.with_mean is False
    assert model.scaler.scale_[0] < 10.0


def test_b2_rejects_raw_challenger_context_as_free_intercept() -> None:
    train = pd.DataFrame(
        {
            "elo_surface_diff": [-1.0, 1.0],
            "is_challenger": [False, True],
            "y_complete_win": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="antisymmetric"):
        fit_b2(train, feature_cols=["elo_surface_diff", "is_challenger"])


def test_b0_uses_antisymmetric_missing_rank_indicator() -> None:
    train = pd.DataFrame(
        {
            "rank_diff": [2.0, -2.0, np.nan, np.nan],
            "rank_missing_a": [False, False, True, False],
            "rank_missing_b": [False, False, False, True],
            "y_complete_win": [1, 0, 0, 1],
        }
    )
    model = fit_b0(train)

    original = train.drop(columns="y_complete_win")
    swapped = pd.DataFrame(
        {
            "rank_diff": -original["rank_diff"],
            "rank_missing_a": original["rank_missing_b"],
            "rank_missing_b": original["rank_missing_a"],
        }
    )

    assert model.fit_intercept is False
    assert np.allclose(
        predict_proba(model, original) + predict_proba(model, swapped),
        1.0,
        atol=1e-10,
    )
