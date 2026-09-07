from datetime import datetime, timedelta, timezone

from tml.odds.storage import QuoteRecord, QuoteStore

UTC = timezone.utc


def _quote(commence_time: datetime, collected_at: datetime) -> QuoteRecord:
    return QuoteRecord(
        event_id="evt-1",
        sport_key="tennis_atp",
        bookmaker="pinnacle",
        player_a="Player A",
        player_b="Player B",
        price_a=-110,
        price_b=-105,
        odds_format="american",
        commence_time=commence_time,
        schedule_observed_at=collected_at,
        last_update=collected_at - timedelta(seconds=10),
        collected_at=collected_at,
        stale_seconds=10,
        is_delayed_reserve=False,
    )


def test_quote_store_writes_real_parquet_and_preserves_schedule_history(tmp_path) -> None:
    path = tmp_path / "quotes.parquet"
    store = QuoteStore(path)
    first_collected = datetime(2026, 9, 6, 18, 0, tzinfo=UTC)
    second_collected = first_collected + timedelta(minutes=15)

    store.append([_quote(first_collected + timedelta(hours=2), first_collected)])
    store.append([_quote(second_collected + timedelta(hours=3), second_collected)])
    loaded = store.load()

    assert path.read_bytes()[:4] == b"PAR1"
    assert len(loaded) == 2
    assert loaded[0].commence_time != loaded[1].commence_time
    assert loaded[0].price_a is not None and loaded[0].price_b is not None
