from collections.abc import Iterable
from datetime import datetime, timedelta

from tml.odds.storage import QuoteRecord
from tml.shared.result import Err, Ok, Result


def select_closing_quote(
    quotes: Iterable[QuoteRecord],
    cutoff: datetime,
    *,
    buffer: timedelta = timedelta(minutes=5),
    max_age: timedelta = timedelta(minutes=30),
) -> Result[QuoteRecord, str]:
    """Select a valid closing benchmark from observed two-sided quotes."""
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("cutoff must be timezone-aware")
    if buffer < timedelta(0):
        raise ValueError("buffer cannot be negative")
    if max_age <= timedelta(0):
        raise ValueError("max_age must be positive")

    boundary = cutoff - buffer
    eligible = [
        quote
        for quote in quotes
        if not quote.is_cancelled
        and not quote.is_suspended
        and quote.last_update <= boundary
        and quote.collected_at <= boundary
        and cutoff - quote.last_update <= max_age
    ]
    if not eligible:
        return Err("no valid observed closing quote")

    # A valid Pinnacle observation is preferred; within a book use the latest pair.
    eligible.sort(
        key=lambda quote: (
            quote.bookmaker.casefold() == "pinnacle",
            quote.last_update,
            quote.collected_at,
        ),
        reverse=True,
    )
    return Ok(eligible[0])
