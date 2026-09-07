from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
import re
import unicodedata

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from tml.shared.result import Err, Ok, Result


class OddsEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    player_a: str = Field(min_length=1)
    player_b: str = Field(min_length=1)
    event_name: str | None = None
    commence_time: AwareDatetime | None = None


class MatchedEvent(OddsEvent):
    pass


class UnmatchedPlayer(BaseModel):
    model_config = ConfigDict(frozen=True)

    player_name: str
    opponent_name: str
    reason: str


def _normalize_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value.casefold()))


def match_players(
    player_name: str,
    opponent_name: str,
    events: Iterable[OddsEvent],
    *,
    event_name: str | None = None,
    commence_time: datetime | None = None,
    time_tolerance: timedelta = timedelta(hours=12),
) -> Result[MatchedEvent, list[UnmatchedPlayer]]:
    """Match a player only in the context of the named opponent and event."""
    player_key = _normalize_name(player_name)
    opponent_key = _normalize_name(opponent_name)
    if not player_key or not opponent_key or player_key == opponent_key:
        return Err(
            [
                UnmatchedPlayer(
                    player_name=player_name,
                    opponent_name=opponent_name,
                    reason="player and opponent context must be distinct non-empty names",
                )
            ]
        )
    if commence_time is not None and (
        commence_time.tzinfo is None or commence_time.utcoffset() is None
    ):
        raise ValueError("commence_time must be timezone-aware")
    if time_tolerance < timedelta(0):
        raise ValueError("time_tolerance cannot be negative")

    candidates: list[OddsEvent] = []
    for raw_event in events:
        event = OddsEvent.model_validate(raw_event)
        sides = {_normalize_name(event.player_a), _normalize_name(event.player_b)}
        if sides != {player_key, opponent_key}:
            continue
        if event_name is not None and _normalize_name(event.event_name or "") != _normalize_name(
            event_name
        ):
            continue
        if commence_time is not None:
            if event.commence_time is None:
                continue
            if abs(event.commence_time - commence_time) > time_tolerance:
                continue
        candidates.append(event)

    if len(candidates) != 1:
        reason = "no contextual event match" if not candidates else "ambiguous contextual event match"
        return Err(
            [
                UnmatchedPlayer(
                    player_name=player_name,
                    opponent_name=opponent_name,
                    reason=reason,
                )
            ]
        )

    match = candidates[0]
    return Ok(
        MatchedEvent(
            event_id=match.event_id,
            player_a=player_name,
            player_b=opponent_name,
            event_name=match.event_name,
            commence_time=match.commence_time,
        )
    )
