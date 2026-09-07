from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from tml.odds.storage import QuoteRecord
from tml.shared.config import get_settings
from tml.shared.result import Err, Ok, Result


class _ApiOutcome(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    price: float


class _ApiMarket(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str = Field(min_length=1)
    outcomes: list[_ApiOutcome]


class _ApiBookmaker(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str = Field(min_length=1)
    last_update: AwareDatetime | None = None
    last_update_ms: int | None = None
    stale_seconds: float | None = Field(default=None, ge=0)
    topped_up: bool = False
    markets: list[_ApiMarket]

    @model_validator(mode="after")
    def require_freshness_timestamp(self) -> _ApiBookmaker:
        if self.last_update is None and self.last_update_ms is None:
            raise ValueError("bookmaker requires last_update or last_update_ms")
        return self

    def observed_at(self) -> datetime:
        if self.last_update is not None:
            return self.last_update
        assert self.last_update_ms is not None
        return datetime.fromtimestamp(self.last_update_ms / 1000, tz=timezone.utc)


class _ApiEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    sport_key: str = Field(min_length=1)
    commence_time: AwareDatetime | None
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    bookmakers: list[_ApiBookmaker]
    cancelled: bool = False
    suspended: bool = False


_EVENTS = TypeAdapter(list[_ApiEvent])


def _atomic_quotes(event: _ApiEvent, collected_at: datetime) -> list[QuoteRecord]:
    quotes: list[QuoteRecord] = []
    for bookmaker in event.bookmakers:
        for market in bookmaker.markets:
            if market.key != "h2h":
                continue
            prices = {outcome.name: outcome.price for outcome in market.outcomes}
            if set(prices) != {event.home_team, event.away_team}:
                raise ValueError("h2h must contain exactly both event sides")
            quotes.append(
                QuoteRecord(
                    event_id=event.id,
                    sport_key=event.sport_key,
                    bookmaker=bookmaker.key,
                    player_a=event.home_team,
                    player_b=event.away_team,
                    price_a=prices[event.home_team],
                    price_b=prices[event.away_team],
                    odds_format="american",
                    commence_time=event.commence_time,
                    schedule_observed_at=collected_at,
                    last_update=bookmaker.observed_at(),
                    collected_at=collected_at,
                    stale_seconds=bookmaker.stale_seconds,
                    is_delayed_reserve=bookmaker.topped_up,
                    is_cancelled=event.cancelled,
                    is_suspended=event.suspended,
                )
            )
    return quotes


def fetch_h2h(
    sport_key: str = "tennis_atp",
    *,
    collected_at: datetime | None = None,
) -> Result[list[QuoteRecord], str]:
    """Fetch and validate atomic observed h2h prices from ParlayAPI."""
    settings = get_settings()
    api_key = settings.parlay_api_key
    if api_key is None or not api_key.strip():
        return Err("PARLAY_API_KEY is required to fetch observed odds")
    if not re.fullmatch(r"[a-z0-9_]+", sport_key):
        return Err("invalid sport key")

    observed_at = collected_at or datetime.now(timezone.utc)
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        return Err("collected_at must be timezone-aware")

    base_url = str(settings.parlay_base_url).rstrip("/")
    path_key = urllib.parse.quote(sport_key, safe="")
    query = urllib.parse.urlencode({"markets": "h2h", "oddsFormat": "american"})
    request = urllib.request.Request(
        f"{base_url}/sports/{path_key}/odds?{query}",
        headers={"X-API-Key": api_key},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=settings.parlay_timeout_seconds
        ) as response:
            payload: Any = json.loads(response.read())
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        return Err("unable to fetch observed odds")

    try:
        events = _EVENTS.validate_python(payload)
        quotes = [
            quote
            for event in events
            for quote in _atomic_quotes(event, observed_at)
        ]
    except (ValidationError, ValueError):
        return Err("invalid odds response from provider")
    return Ok(quotes)
