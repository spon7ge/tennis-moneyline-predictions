import math
from datetime import date

import pandas as pd
import pytest

from tml.features.ranking import rank_feature_pair


def test_rank_diff_when_both_ranks_present() -> None:
    row = {"rank_a": 1, "rank_b": 10}
    tourney_date = date(2000, 1, 10)

    out = rank_feature_pair(row, tourney_date)

    expected = math.log(10) - math.log(1)
    assert out["rank_diff"] == pytest.approx(expected)
    assert out["rank_missing_a"] is False
    assert out["rank_missing_b"] is False
    assert out["rank_source"] == "match_embedded"
    assert out["rank_staleness_days"] == 0


def test_rank_diff_positive_when_a_is_higher_ranked() -> None:
    row = {"rank_a": 5, "rank_b": 50}
    out = rank_feature_pair(row, date(2000, 6, 1))
    assert out["rank_diff"] > 0


def test_missing_rank_a_sets_flag_and_nan_diff() -> None:
    row = {"rank_a": None, "rank_b": 20}
    out = rank_feature_pair(row, date(2000, 1, 10))

    assert out["rank_missing_a"] is True
    assert out["rank_missing_b"] is False
    assert out["rank_diff"] is None or (
        isinstance(out["rank_diff"], float) and math.isnan(out["rank_diff"])
    )
    assert out["rank_source"] == "match_embedded"


def test_missing_rank_b_sets_flag_and_nan_diff() -> None:
    row = {"rank_a": 8, "rank_b": pd.NA}
    out = rank_feature_pair(row, date(2000, 1, 10))

    assert out["rank_missing_a"] is False
    assert out["rank_missing_b"] is True
    assert out["rank_diff"] is None or (
        isinstance(out["rank_diff"], float) and math.isnan(out["rank_diff"])
    )


def test_both_ranks_missing_sets_both_flags() -> None:
    row = {"rank_a": "", "rank_b": None}
    out = rank_feature_pair(row, date(2000, 1, 10))

    assert out["rank_missing_a"] is True
    assert out["rank_missing_b"] is True
    assert out["rank_diff"] is None or (
        isinstance(out["rank_diff"], float) and math.isnan(out["rank_diff"])
    )


def test_rank_below_one_treated_as_missing() -> None:
    row = {"rank_a": 0, "rank_b": 15}
    out = rank_feature_pair(row, date(2000, 1, 10))

    assert out["rank_missing_a"] is True
    assert out["rank_missing_b"] is False
    assert out["rank_diff"] is None or (
        isinstance(out["rank_diff"], float) and math.isnan(out["rank_diff"])
    )


def test_accepts_pandas_series_row() -> None:
    row = pd.Series({"rank_a": 3, "rank_b": 12})
    out = rank_feature_pair(row, date(2001, 3, 15))

    assert out["rank_diff"] == pytest.approx(math.log(12) - math.log(3))
    assert out["rank_staleness_days"] == 0
