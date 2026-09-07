import math

from tml.shared.result import Result
from tml.shared.result import Err, Ok


def _implied_probability(price: float, odds_format: str) -> float:
    if not math.isfinite(price):
        raise ValueError("observed price must be finite")
    if odds_format == "american":
        if -100 < price < 100:
            raise ValueError("observed American price must be <= -100 or >= 100")
        return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)
    if odds_format == "decimal":
        if price <= 1:
            raise ValueError("observed decimal price must be greater than 1")
        return 1.0 / price
    raise ValueError("odds_format must be 'american' or 'decimal'")


def multiplicative_novig(
    price_a: float,
    price_b: float,
    odds_format: str = "american",
) -> Result[tuple[float, float], str]:
    """Convert two observed prices to normalized implied probabilities."""
    try:
        implied_a = _implied_probability(float(price_a), odds_format)
        implied_b = _implied_probability(float(price_b), odds_format)
    except (TypeError, ValueError) as error:
        return Err(str(error))

    probability_a = implied_a / (implied_a + implied_b)
    # Computing the second side as the complement preserves the two-way invariant.
    return Ok((probability_a, 1.0 - probability_a))
