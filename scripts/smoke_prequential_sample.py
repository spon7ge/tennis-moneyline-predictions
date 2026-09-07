from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from tml.data.identity import PlayerIdentityMap
from tml.data.modeling_table import build_modeling_table
from tml.validation.metrics import (
    common_eligible,
    coverage_report,
    delta_log_loss,
    log_loss,
)
from tml.validation.prequential import PrequentialConfig, run_prequential

if __package__:
    from .smoke_ingest import DEFAULT_DATA_ROOT, PROJECT_ROOT, ingest_year_range
else:
    from smoke_ingest import DEFAULT_DATA_ROOT, PROJECT_ROOT, ingest_year_range


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data/processed/smoke_preds.parquet"
DEFAULT_FEATURE_PATH = PROJECT_ROOT / "data/processed/smoke_features.parquet"
MODEL_COLUMNS = ("p_b0", "p_b1", "p_b2")


def select_smoke_tournaments(
    matches: pd.DataFrame, per_level_year: int
) -> pd.DataFrame:
    """Keep the earliest bounded tournaments for every year and tour level."""
    if per_level_year < 1:
        raise ValueError("per_level_year must be at least 1")
    frame = matches.copy()
    frame["tourney_date"] = pd.to_datetime(frame["tourney_date"], errors="raise")
    frame["_year"] = frame["tourney_date"].dt.year
    key_columns = ["_year", "tour_level", "tourney_date", "tourney_id"]
    selected_keys = (
        frame[key_columns]
        .drop_duplicates()
        .sort_values(key_columns, kind="stable")
        .groupby(["_year", "tour_level"], sort=False)
        .head(per_level_year)
    )
    return (
        frame.merge(selected_keys, on=key_columns, how="inner", validate="many_to_one")
        .drop(columns=["_year"])
        .reset_index(drop=True)
    )


def smoke_metric_summary(predictions: pd.DataFrame) -> dict[str, object]:
    """Summarize diagnostic coverage and paired deltas on the smoke window."""
    coverage = coverage_report(predictions, MODEL_COLUMNS)
    eligible = common_eligible(predictions, MODEL_COLUMNS)
    summary: dict[str, object] = {
        **coverage,
        "claim_scope": "smoke-window diagnostic; not final-era results",
    }
    if eligible.empty:
        summary["delta_status"] = "unavailable: no common eligible rows"
        return summary

    y = eligible["y"].to_numpy(dtype=float)
    p_b1 = eligible["p_b1"].to_numpy(dtype=float)
    summary.update(
        {
            "log_loss_b1": log_loss(y, p_b1),
            "delta_log_loss_b0_minus_b1": delta_log_loss(
                y, eligible["p_b0"].to_numpy(dtype=float), p_b1
            ),
            "delta_log_loss_b2_minus_b1": delta_log_loss(
                y, eligible["p_b2"].to_numpy(dtype=float), p_b1
            ),
        }
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a short real-data prequential integration smoke."
    )
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--feature-store", type=Path, default=DEFAULT_FEATURE_PATH
    )
    parser.add_argument(
        "--tournaments-per-level-year",
        type=int,
        default=8,
        help="Earliest tournaments retained per tour level and year.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    ingested = ingest_year_range(args.data_root, args.start, args.end)
    matches = build_modeling_table(
        ingested.matches, PlayerIdentityMap(version="smoke-v1")
    )
    matches["dataset_snapshot_id"] = ingested.snapshot_id
    completed_rows_available = len(matches)
    matches = select_smoke_tournaments(
        matches, args.tournaments_per_level_year
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.feature_store.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    args.feature_store.unlink(missing_ok=True)

    config = PrequentialConfig(
        burn_in_end=date(args.start - 1, 12, 31),
        elo_from=args.start,
        supervised_from=args.start + 1,
        rolling_years=1,
        dev_era=(args.start, args.end),
        final_era=(args.end + 1, 2100),
        feature_store_path=args.feature_store,
    )
    result = run_prequential(matches, config)
    result.predictions.to_parquet(args.output, index=False, engine="pyarrow")
    summary = smoke_metric_summary(result.predictions)

    print("SMOKE WINDOW DIAGNOSTIC ONLY — not final-era results")
    print(f"completed_rows_available={completed_rows_available}")
    print(f"rows_modeled={len(matches)}")
    print(f"predictions_written={len(result.predictions)}")
    print(f"output_path={args.output}")
    print(f"n_common={summary['n_common']}")
    for model, value in summary["coverage"].items():
        print(f"coverage_{model}={value:.6f}")
    if "delta_status" in summary:
        print(f"delta_status={summary['delta_status']}")
    else:
        print(f"log_loss_b1={summary['log_loss_b1']:.6f}")
        print(
            "delta_log_loss_b0_minus_b1="
            f"{summary['delta_log_loss_b0_minus_b1']:.6f}"
        )
        print(
            "delta_log_loss_b2_minus_b1="
            f"{summary['delta_log_loss_b2_minus_b1']:.6f}"
        )


if __name__ == "__main__":
    main()
