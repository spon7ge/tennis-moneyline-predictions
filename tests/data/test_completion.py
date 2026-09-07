import pytest
from tml.data.completion import parse_completion_status


@pytest.mark.parametrize(
    "score,expected",
    [
        ("6-4 6-3", "completed"),
        ("6-4 3-6 6-2", "completed"),
        ("6-4 2-6 1-4 RET", "retirement"),
        ("W/O", "walkover"),
        ("Walkover", "walkover"),
        ("DEF", "default"),
        ("4-6 0-1 abd", "abandoned"),
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_parse_completion_status(score, expected):
    assert parse_completion_status(score) == expected
