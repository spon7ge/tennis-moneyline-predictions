from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tml.data.ingest import ingest_match_files
from tml.data.manifest import snapshot_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "tml-data"


@dataclass(frozen=True, slots=True)
class SmokeIngestResult:
    matches: pd.DataFrame
    manifest: list[dict[str, object]]
    snapshot_id: str


def ingest_year_range(
    data_root: Path, start: int, end: int
) -> SmokeIngestResult:
    """Ingest ATP and Challenger files for an inclusive year range."""
    if start > end:
        raise ValueError("start year must be on or before end year")

    years = range(start, end + 1)
    atp_paths = [data_root / f"{year}.csv" for year in years]
    challenger_paths = [
        data_root / f"{year}_challenger.csv" for year in years
    ]
    atp = ingest_match_files(atp_paths, root=data_root, tour_level="atp")
    challenger = ingest_match_files(
        challenger_paths, root=data_root, tour_level="challenger"
    )

    manifest = sorted(
        [*atp.manifest, *challenger.manifest],
        key=lambda row: str(row["relative_path"]),
    )
    combined_snapshot_id = snapshot_id(manifest)
    matches = pd.concat([atp.matches, challenger.matches], ignore_index=True)
    matches["dataset_snapshot_id"] = combined_snapshot_id
    return SmokeIngestResult(matches, manifest, combined_snapshot_id)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-ingest real ATP and Challenger match files."
    )
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = ingest_year_range(args.data_root, args.start, args.end)

    print(f"rows_total={len(result.matches)}")
    print("rows_by_source:")
    for source, count in result.matches["source_file"].value_counts().sort_index().items():
        print(f"  {source}: {count}")
    print("completion_breakdown:")
    for status, count in (
        result.matches["completion_status"].value_counts().sort_index().items()
    ):
        print(f"  {status}: {count}")
    print(f"dataset_snapshot_id={result.snapshot_id}")


if __name__ == "__main__":
    main()
