from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path

import pandas as pd

from tml.features.builders import build_feature_row
from tml.features.elo import EloState, update_tournament
from tml.features.store import DEFAULT_FEATURE_PATH, FeatureStore
from tml.models.elo_prob import elo_win_prob
from tml.models.ranking_logit import B0Model
from tml.models.supervised import B2Model, fit_b0, fit_b2, predict_proba

PREDICTION_COLUMNS = [
    "match_id",
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

DEFAULT_B2_FEATURE_COLUMNS = (
    "elo_surface_diff",
    "elo_overall_diff",
    "rank_diff",
    "form_diff",
    "serve_1st_in_diff",
    "serve_1st_won_diff",
    "serve_2nd_won_diff",
    "experience_diff",
    "age_diff",
    "elo_x_bestof",
)

_REQUIRED_MATCH_COLUMNS = {
    "match_id",
    "tourney_id",
    "tourney_date",
    "player_a_id",
    "player_b_id",
    "surface",
    "best_of",
    "tour_level",
    "y_complete_win",
    "dataset_snapshot_id",
}


@dataclass(frozen=True)
class PrequentialConfig:
    burn_in_end: date = date(1999, 12, 31)
    elo_from: int = 2000
    supervised_from: int = 2005
    rolling_years: int = 5
    dev_era: tuple[int, int] = (2005, 2018)
    final_era: tuple[int, int] = (2019, 2100)
    feature_store_path: str | Path = DEFAULT_FEATURE_PATH
    b2_feature_cols: tuple[str, ...] = DEFAULT_B2_FEATURE_COLUMNS
    b0_temperature: float = 1.0
    b2_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.rolling_years < 1:
            raise ValueError("rolling_years must be at least 1")
        if self.elo_from < 1 or self.supervised_from < 1:
            raise ValueError("model start years must be positive")
        if not self.b2_feature_cols:
            raise ValueError("b2_feature_cols must not be empty")
        if len(set(self.b2_feature_cols)) != len(self.b2_feature_cols):
            raise ValueError("b2_feature_cols must be unique")
        for name, era in (("dev_era", self.dev_era), ("final_era", self.final_era)):
            if len(era) != 2 or era[0] > era[1]:
                raise ValueError(f"{name} must be an inclusive (start, end) pair")
        for name, value in (
            ("b0_temperature", self.b0_temperature),
            ("b2_temperature", self.b2_temperature),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class PrequentialResult:
    predictions: pd.DataFrame
    feature_store_path: Path


def _validated_matches(matches: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(matches, pd.DataFrame):
        raise TypeError("matches must be a pandas DataFrame")
    missing = sorted(_REQUIRED_MATCH_COLUMNS.difference(matches.columns))
    if missing:
        raise ValueError(f"matches missing required columns: {', '.join(missing)}")
    if matches["match_id"].duplicated().any():
        raise ValueError("match_id must be unique")
    if matches[list(_REQUIRED_MATCH_COLUMNS)].isna().any().any():
        null_columns = sorted(
            column
            for column in _REQUIRED_MATCH_COLUMNS
            if matches[column].isna().any()
        )
        raise ValueError(
            f"matches contain null required fields: {', '.join(null_columns)}"
        )

    frame = matches.copy(deep=True)
    frame["tourney_date"] = pd.to_datetime(frame["tourney_date"], errors="coerce")
    if frame["tourney_date"].isna().any():
        raise ValueError("tourney_date must contain valid dates")
    if not frame["y_complete_win"].isin((0, 1)).all():
        raise ValueError("y_complete_win must contain only 0 and 1")
    if not frame["best_of"].isin((3, 5)).all():
        raise ValueError("best_of must contain only 3 or 5")
    return frame.sort_values(
        ["tourney_date", "tourney_id"], kind="stable"
    ).reset_index(drop=True)


def _training_rows(
    store: FeatureStore,
    year: int,
    rolling_years: int,
) -> pd.DataFrame:
    frozen = store.load()
    dates = pd.to_datetime(frozen["tourney_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("persisted tourney_date must contain valid dates")
    start = pd.Timestamp(year=year - rolling_years, month=1, day=1)
    end = pd.Timestamp(year=year - 1, month=12, day=31)
    return frozen.loc[dates.between(start, end)].copy()


def _prediction_row(
    match: pd.Series,
    year: int,
    p_b0: float,
    p_b1: float,
    p_b2: float,
) -> dict[str, object]:
    return {
        "match_id": match["match_id"],
        "year": year,
        "p_b0": p_b0,
        "p_b1": p_b1,
        "p_b2": p_b2,
        "y": int(match["y_complete_win"]),
        "tour_level": match["tour_level"],
        "surface": match["surface"],
        "prediction_regime": "pre_tournament",
        "dataset_snapshot_id": match["dataset_snapshot_id"],
    }


def run_prequential(
    matches: pd.DataFrame,
    config: PrequentialConfig | None = None,
) -> PrequentialResult:
    """Run tournament-batched walk-forward scoring from frozen feature rows."""
    resolved = config or PrequentialConfig()
    ordered = _validated_matches(matches)
    store = FeatureStore(resolved.feature_store_path)
    state = EloState()
    history = ordered.iloc[0:0].copy()
    predictions: list[dict[str, object]] = []
    supervised_models: dict[int, tuple[B0Model, B2Model]] = {}

    grouped = ordered.groupby(["tourney_date", "tourney_id"], sort=False)
    for (timestamp, _), tournament in grouped:
        tournament_date = pd.Timestamp(timestamp).date()
        year = tournament_date.year

        if tournament_date <= resolved.burn_in_end:
            update_tournament(state, tournament)
            history = pd.concat([history, tournament], ignore_index=True)
            continue

        frozen_rows = pd.DataFrame(
            [
                build_feature_row(match, state, history)
                for _, match in tournament.iterrows()
            ]
        )
        store.persist(frozen_rows)

        models: tuple[B0Model, B2Model] | None = None
        if year >= resolved.supervised_from:
            if year not in supervised_models:
                train_rows = _training_rows(store, year, resolved.rolling_years)
                supervised_models[year] = (
                    fit_b0(train_rows),
                    fit_b2(train_rows, resolved.b2_feature_cols),
                )
            models = supervised_models[year]

        for (_, match), (_, features) in zip(
            tournament.iterrows(), frozen_rows.iterrows(), strict=True
        ):
            p_b1 = math.nan
            if year >= resolved.elo_from:
                p_b1 = elo_win_prob(
                    state,
                    match["player_a_id"],
                    match["player_b_id"],
                    str(match["surface"]),
                    int(match["best_of"]),
                )

            p_b0 = math.nan
            p_b2 = math.nan
            if models is not None:
                feature_frame = pd.DataFrame([features])
                p_b0 = float(
                    predict_proba(
                        models[0], feature_frame, T=resolved.b0_temperature
                    )[0]
                )
                p_b2 = float(
                    predict_proba(
                        models[1], feature_frame, T=resolved.b2_temperature
                    )[0]
                )
            predictions.append(_prediction_row(match, year, p_b0, p_b1, p_b2))

        update_tournament(state, tournament)
        history = pd.concat([history, tournament], ignore_index=True)

    prediction_frame = pd.DataFrame(predictions, columns=PREDICTION_COLUMNS)
    return PrequentialResult(prediction_frame, Path(resolved.feature_store_path))
