from datetime import datetime, timedelta, timezone

from tml.odds.closing import select_closing_quote
from tml.odds.storage import QuoteRecord
from tml.shared.result import Err, Ok

UTC = timezone.utc
CUTOFF = datetime(2026, 9, 6, 20, 0, tzinfo=UTC)


def _quote(
    *,
    bookmaker: str = "fanduel",
    last_update: datetime = CUTOFF - timedelta(minutes=10),
    collected_at: datetime = CUTOFF - timedelta(minutes=9),
    commence_time: datetime = CUTOFF,
) -> QuoteRecord:
    return QuoteRecord(
        event_id="evt-1",
        sport_key="tennis_atp",
        bookmaker=bookmaker,
        player_a="Player A",
        player_b="Player B",
        price_a=-110,
        price_b=-105,
        odds_format="american",
        commence_time=commence_time,
        schedule_observed_at=collected_at,
        last_update=last_update,
        collected_at=collected_at,
    )


def test_closing_rejects_quote_collected_after_buffer_boundary() -> None:
    quote = _quote(collected_at=CUTOFF - timedelta(minutes=4))

    result = select_closing_quote([quote], CUTOFF)

    assert isinstance(result, Err)


def test_closing_requires_last_update_before_buffer_and_within_max_age() -> None:
    too_recent = _quote(
        last_update=CUTOFF - timedelta(minutes=4),
        collected_at=CUTOFF - timedelta(minutes=3),
    )
    too_old = _quote(last_update=CUTOFF - timedelta(minutes=31))

    assert isinstance(select_closing_quote([too_recent], CUTOFF), Err)
    assert isinstance(select_closing_quote([too_old], CUTOFF), Err)


def test_closing_prefers_valid_pinnacle_quote() -> None:
    fanduel = _quote(bookmaker="fanduel")
    pinnacle = _quote(
        bookmaker="pinnacle",
        last_update=CUTOFF - timedelta(minutes=12),
        collected_at=CUTOFF - timedelta(minutes=11),
    )

    result = select_closing_quote([fanduel, pinnacle], CUTOFF)

    assert isinstance(result, Ok)
    assert result.value.bookmaker == "pinnacle"


def test_quote_record_names_delayed_reserve_and_keeps_schedule_version() -> None:
    quote = _quote()
    delayed = quote.model_copy(update={"is_delayed_reserve": True})

    assert delayed.is_delayed_reserve is True
    assert delayed.commence_time == CUTOFF
    assert delayed.schedule_observed_at == delayed.collected_at
