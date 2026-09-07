from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit


class OOFModel(Protocol):
    def decision_function(self, frame: pd.DataFrame) -> np.ndarray: ...


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Apply intercept-free temperature scaling to antisymmetric scores."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = np.asarray(scores, dtype=float)
    return np.asarray(expit(values / temperature), dtype=float)


def prequential_oof_scores(
    train_df: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], OOFModel],
) -> np.ndarray:
    """Score each year using a model fit only on strictly earlier years.

    Rows in the initial year have no eligible prior-year fit and are returned
    as NaN. The output remains aligned to the input row order.
    """
    if "tourney_date" not in train_df:
        raise ValueError("missing fold column: tourney_date")
    dates = pd.to_datetime(train_df["tourney_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("tourney_date must contain valid dates")

    years = dates.dt.year.to_numpy()
    scores = np.full(len(train_df), np.nan, dtype=float)
    for year in np.sort(np.unique(years))[1:]:
        train_mask = years < year
        test_mask = years == year
        model = fit_fn(train_df.loc[train_mask].copy())
        fold_scores = np.asarray(
            model.decision_function(train_df.loc[test_mask].copy()), dtype=float
        )
        if fold_scores.shape != (int(test_mask.sum()),):
            raise ValueError("decision_function must return one score per row")
        if not np.isfinite(fold_scores).all():
            raise ValueError("OOF model returned non-finite scores")
        scores[test_mask] = fold_scores
    return scores


def fit_temperature(oof_scores: np.ndarray, y: np.ndarray) -> float:
    """Fit one positive temperature from prequential out-of-fold scores."""
    scores = np.asarray(oof_scores, dtype=float)
    outcomes = np.asarray(y)
    if scores.ndim != 1 or outcomes.ndim != 1 or scores.shape != outcomes.shape:
        raise ValueError("oof_scores and y must be aligned one-dimensional arrays")

    usable = np.isfinite(scores) & pd.notna(outcomes)
    scores = scores[usable]
    outcomes = outcomes[usable].astype(float)
    if scores.size == 0:
        raise ValueError("no finite OOF scores available")
    if not np.isin(outcomes, (0.0, 1.0)).all():
        raise ValueError("y must contain only 0 and 1")

    def log_loss(log_temperature: float) -> float:
        logits = scores / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0.0, logits) - outcomes * logits))

    result = minimize_scalar(
        log_loss,
        bounds=(np.log(1e-3), np.log(1e3)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    return float(np.exp(result.x))
