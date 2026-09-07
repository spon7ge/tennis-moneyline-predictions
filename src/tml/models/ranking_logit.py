from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def rank_difference_features(frame: pd.DataFrame) -> np.ndarray:
    """Return rank features that negate exactly when player slots are swapped."""
    required = {"rank_diff", "rank_missing_a", "rank_missing_b"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing B0 columns: {missing}")

    rank_diff = pd.to_numeric(frame["rank_diff"], errors="coerce").fillna(0.0)
    missing_diff = (
        frame["rank_missing_b"].astype(float)
        - frame["rank_missing_a"].astype(float)
    )
    return np.column_stack((rank_diff.to_numpy(), missing_diff.to_numpy()))


@dataclass(frozen=True)
class B0Model:
    """No-intercept ranking logistic model with antisymmetric features."""

    estimator: LogisticRegression

    @property
    def intercept_(self) -> float:
        return float(self.estimator.intercept_[0])

    @property
    def fit_intercept(self) -> bool:
        return bool(self.estimator.fit_intercept)

    def decision_function(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.estimator.decision_function(rank_difference_features(frame)),
            dtype=float,
        )
