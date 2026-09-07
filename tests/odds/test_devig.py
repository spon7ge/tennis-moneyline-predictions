import math

from tml.odds.devig import multiplicative_novig
from tml.shared.result import Err, Ok


def test_multiplicative_novig_probabilities_sum_to_one() -> None:
    result = multiplicative_novig(-110, -110)

    assert isinstance(result, Ok)
    assert math.isclose(sum(result.value), 1.0, rel_tol=0.0, abs_tol=1e-15)
    assert result.value == (0.5, 0.5)


def test_multiplicative_novig_supports_decimal_observed_prices() -> None:
    result = multiplicative_novig(1.8, 2.2, odds_format="decimal")

    assert isinstance(result, Ok)
    assert math.isclose(sum(result.value), 1.0, rel_tol=0.0, abs_tol=1e-15)
    assert result.value[0] > result.value[1]


def test_multiplicative_novig_returns_err_for_invalid_observed_price() -> None:
    result = multiplicative_novig(0, -110)

    assert isinstance(result, Err)
    assert "observed" in result.error
