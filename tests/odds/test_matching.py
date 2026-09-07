from datetime import datetime, timezone

from tml.odds.matching import OddsEvent, match_players
from tml.shared.result import Err, Ok


def _event(
    event_id: str,
    player_a: str,
    player_b: str,
    event_name: str = "US Open",
) -> OddsEvent:
    return OddsEvent(
        event_id=event_id,
        player_a=player_a,
        player_b=player_b,
        event_name=event_name,
        commence_time=datetime(2026, 9, 6, 20, 0, tzinfo=timezone.utc),
    )


def test_match_players_requires_opponent_context() -> None:
    events = [
        _event("wrong", "Alex Smith", "Wrong Opponent"),
        _event("right", "Alex Smith", "Taylor Jones"),
    ]

    result = match_players("Alex Smith", "Taylor Jones", events)

    assert isinstance(result, Ok)
    assert result.value.event_id == "right"


def test_match_players_returns_unmatched_queue_item() -> None:
    result = match_players(
        "Alex Smith",
        "Taylor Jones",
        [_event("wrong", "Alex Smith", "Wrong Opponent")],
    )

    assert isinstance(result, Err)
    assert result.error[0].player_name == "Alex Smith"
    assert result.error[0].opponent_name == "Taylor Jones"


def test_match_players_uses_event_context_to_disambiguate() -> None:
    events = [
        _event("indian-wells", "Alex Smith", "Taylor Jones", "Indian Wells"),
        _event("us-open", "Taylor Jones", "Alex Smith", "US Open"),
    ]

    result = match_players(
        "Alex Smith",
        "Taylor Jones",
        events,
        event_name="US Open",
    )

    assert isinstance(result, Ok)
    assert result.value.event_id == "us-open"
    assert result.value.player_a == "Alex Smith"
    assert result.value.player_b == "Taylor Jones"
