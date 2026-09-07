from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tml.models.ranking_logit import B0Model, rank_difference_features

TARGET_COLUMN = "y_complete_win"
_ALLOWED_INTERACTION_COLUMNS = {"elo_x_bestof"}


class ScoreModel(Protocol):
    def decision_function(self, frame: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class B2Model:
    """Train-fitted preprocessing and no-intercept L2 logistic estimator."""

    feature_cols: tuple[str, ...]
    imputer: SimpleImputer
    scaler: StandardScaler
    estimator: LogisticRegression

    @property
    def intercept_(self) -> float:
        return float(self.estimator.intercept_[0])

    @property
    def fit_intercept(self) -> bool:
        return bool(self.estimator.fit_intercept)

    @property
    def C(self) -> float:
        return float(self.estimator.C)

    @property
    def penalty(self) -> str:
        return str(self.estimator.penalty)

    def decision_function(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = _feature_matrix(frame, self.feature_cols)
        transformed = self.scaler.transform(self.imputer.transform(matrix))
        return np.asarray(self.estimator.decision_function(transformed), dtype=float)


def _outcomes(train_df: pd.DataFrame) -> np.ndarray:
    if TARGET_COLUMN not in train_df:
        raise ValueError(f"missing target column: {TARGET_COLUMN}")
    outcomes = pd.to_numeric(train_df[TARGET_COLUMN], errors="raise").to_numpy()
    if not np.isin(outcomes, (0, 1)).all():
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1")
    if np.unique(outcomes).size < 2:
        raise ValueError("training outcomes must contain both classes")
    return outcomes.astype(int)


def _validate_feature_cols(feature_cols: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    columns = tuple(feature_cols)
    if not columns:
        raise ValueError("feature_cols must not be empty")
    if len(set(columns)) != len(columns):
        raise ValueError("feature_cols must be unique")
    asymmetric = [
        column
        for column in columns
        if not column.endswith("_diff")
        and column not in _ALLOWED_INTERACTION_COLUMNS
    ]
    if asymmetric:
        raise ValueError(
            "features must be antisymmetric differences or approved interactions; "
            f"got {asymmetric}"
        )
    return columns


def _feature_matrix(
    frame: pd.DataFrame, feature_cols: tuple[str, ...]
) -> pd.DataFrame:
    missing = sorted(set(feature_cols).difference(frame.columns))
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    try:
        return frame.loc[:, feature_cols].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("model features must be numeric") from exc


def fit_b0(train_df: pd.DataFrame) -> B0Model:
    """Fit the rank-difference baseline without an intercept."""
    estimator = LogisticRegression(
        C=1.0,
        penalty="l2",
        fit_intercept=False,
        solver="lbfgs",
    )
    estimator.fit(rank_difference_features(train_df), _outcomes(train_df))
    return B0Model(estimator=estimator)


def fit_b2(
    train_df: pd.DataFrame, feature_cols: list[str] | tuple[str, ...]
) -> B2Model:
    """Fit B2 using zero imputation and antisymmetric scaling."""
    columns = _validate_feature_cols(feature_cols)
    matrix = _feature_matrix(train_df, columns)
    imputer = SimpleImputer(
        strategy="constant", fill_value=0.0, keep_empty_features=True
    )
    scaler = StandardScaler(with_mean=False)
    transformed = scaler.fit_transform(imputer.fit_transform(matrix))
    estimator = LogisticRegression(
        C=1.0,
        penalty="l2",
        fit_intercept=False,
        solver="lbfgs",
    )
    estimator.fit(transformed, _outcomes(train_df))
    return B2Model(columns, imputer, scaler, estimator)


def predict_proba(
    model: ScoreModel, frame: pd.DataFrame, T: float = 1.0
) -> np.ndarray:
    """Map an antisymmetric model score through temperature-scaled logistic."""
    if not np.isfinite(T) or T <= 0:
        raise ValueError("temperature must be finite and positive")
    scores = np.asarray(model.decision_function(frame), dtype=float)
    return np.asarray(expit(scores / T), dtype=float)
