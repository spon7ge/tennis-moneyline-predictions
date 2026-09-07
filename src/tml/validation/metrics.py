from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_EPS = 1e-15
_PRIMARY_MODEL_COLS = ("p_b0", "p_b1", "p_b2")


def _as_arrays(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    outcomes = np.asarray(y, dtype=float)
    probabilities = np.asarray(p, dtype=float)
    if outcomes.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y and p must be one-dimensional arrays")
    if outcomes.shape != probabilities.shape:
        raise ValueError("y and p must have the same length")
    if outcomes.size == 0:
        raise ValueError("y and p must not be empty")
    if not np.isin(outcomes, (0.0, 1.0)).all():
        raise ValueError("y must contain only 0 and 1")
    if not np.isfinite(probabilities).all():
        raise ValueError("p must contain only finite values")
    if (probabilities <= 0.0).any() or (probabilities >= 1.0).any():
        raise ValueError("p must contain probabilities strictly between 0 and 1")
    return outcomes, probabilities


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary log loss on aligned outcome and probability arrays."""
    outcomes, probabilities = _as_arrays(y, p)
    losses = -(outcomes * np.log(probabilities) + (1.0 - outcomes) * np.log(1.0 - probabilities))
    return float(np.mean(losses))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean Brier score on aligned outcome and probability arrays."""
    outcomes, probabilities = _as_arrays(y, p)
    return float(np.mean((probabilities - outcomes) ** 2))


def delta_log_loss(y: np.ndarray, p_model: np.ndarray, p_b1: np.ndarray) -> float:
    """Mean paired log-loss difference model minus B1 baseline."""
    return log_loss(y, p_model) - log_loss(y, p_b1)


def common_eligible(df: pd.DataFrame, cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Return rows where every listed prediction column is finite and non-null."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not cols:
        raise ValueError("cols must not be empty")

    missing = [column for column in cols if column not in df.columns]
    if missing:
        raise ValueError(f"df missing required columns: {', '.join(missing)}")

    mask = df[list(cols)].notna().all(axis=1)
    for column in cols:
        mask &= np.isfinite(df[column].to_numpy(dtype=float))
    return df.loc[mask].copy()


def coverage_report(df: pd.DataFrame, model_cols: list[str] | tuple[str, ...]) -> dict[str, object]:
    """Report total rows, common eligible count, and per-model coverage."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if len(df) == 0:
        raise ValueError("df must not be empty")

    eligible = common_eligible(df, model_cols)
    coverage = {
        column: float(df[column].notna().mean()) for column in model_cols
    }
    report: dict[str, object] = {
        "n_total": int(len(df)),
        "n_common": int(len(eligible)),
        "coverage": coverage,
    }
    if "prediction_regime" in df.columns:
        report["by_regime"] = {
            str(regime): coverage_report(
                group.drop(columns=["prediction_regime"]), model_cols
            )
            for regime, group in df.groupby("prediction_regime", sort=False)
        }
    return report


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Fit diagnostic calibration logit(y) ~ intercept + slope * logit(p)."""
    outcomes, probabilities = _as_arrays(y, p)
    logits = np.log(probabilities / (1.0 - probabilities))
    model = LogisticRegression(
        fit_intercept=True,
        C=1e6,
        solver="lbfgs",
        max_iter=1_000,
    )
    model.fit(logits.reshape(-1, 1), outcomes.astype(int))
    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0, 0])
    return intercept, slope


def final_era_primary_delta(
    preds: pd.DataFrame,
    *,
    year_min: int = 2019,
    regime: str = "pre_tournament",
) -> dict[str, object]:
    """Primary final-era delta log loss versus B1 on the common eligible set."""
    required = {"year", "prediction_regime", "y", "p_b0", "p_b1", "p_b2"}
    missing = sorted(required.difference(preds.columns))
    if missing:
        raise ValueError(f"preds missing required columns: {', '.join(missing)}")

    filtered = preds.loc[
        (preds["year"] >= year_min) & (preds["prediction_regime"] == regime)
    ]
    eligible = common_eligible(filtered, _PRIMARY_MODEL_COLS)
    if eligible.empty:
        raise ValueError("no common eligible rows in final-era primary slice")

    y = eligible["y"].to_numpy(dtype=float)
    p_b1 = eligible["p_b1"].to_numpy(dtype=float)
    report: dict[str, object] = {
        "year_min": year_min,
        "regime": regime,
        "n": int(len(eligible)),
        "delta_b0": delta_log_loss(y, eligible["p_b0"].to_numpy(dtype=float), p_b1),
        "delta_b2": delta_log_loss(y, eligible["p_b2"].to_numpy(dtype=float), p_b1),
        "log_loss_b1": log_loss(y, p_b1),
        "log_loss_b0": log_loss(y, eligible["p_b0"].to_numpy(dtype=float)),
        "log_loss_b2": log_loss(y, eligible["p_b2"].to_numpy(dtype=float)),
        "brier_b1": brier(y, p_b1),
        "brier_b2": brier(y, eligible["p_b2"].to_numpy(dtype=float)),
    }
    intercept, slope = calibration_intercept_slope(y, eligible["p_b2"].to_numpy(dtype=float))
    report["calibration_intercept_b2"] = intercept
    report["calibration_slope_b2"] = slope
    return report
