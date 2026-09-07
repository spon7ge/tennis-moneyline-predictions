from datetime import date
from pathlib import Path

import pytest

from tml.data.ingest import ingest_match_files


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("fixture_name", "tour_level"),
    [("atp_tiny.csv", "atp"), ("challenger_tiny.csv", "challenger")],
)
def test_ingest_adds_completion_lineage_and_match_id(
    fixture_name: str, tour_level: str
) -> None:
    path = FIXTURES / fixture_name

    result = ingest_match_files([path], root=FIXTURES, tour_level=tour_level)

    df = result.matches
    assert set(df["completion_status"]) >= {"completed", "retirement"}
    assert (df["tour_level"] == tour_level).all()
    assert df["match_id"].tolist() == [
        f"{tour_level}:{tourney_id}:{match_num}"
        for tourney_id, match_num in zip(
            df["tourney_id"], df["match_num"], strict=True
        )
    ]
    assert (df["source_file"] == fixture_name).all()
    assert all(isinstance(value, date) for value in df["tourney_date"])
    assert result.manifest[0]["relative_path"] == fixture_name
    assert result.dataset_snapshot_id
    assert result.ingested_at.tzinfo is not None


def test_ingest_preserves_identifier_strings() -> None:
    result = ingest_match_files(
        [FIXTURES / "atp_tiny.csv"], root=FIXTURES, tour_level="atp"
    )

    assert result.matches["winner_id"].tolist() == ["1001", "1002"]
    assert result.matches["loser_id"].tolist() == ["2001", "2002"]


def test_ingest_rejects_missing_paths(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(ValueError, match="does not exist"):
        ingest_match_files([missing], root=tmp_path, tour_level="atp")
