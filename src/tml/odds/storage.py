from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

DEFAULT_QUOTE_PATH = Path("data/raw/odds/quotes.parquet")


class QuoteRecord(BaseModel):
    """One atomic pair of observed prices and its schedule version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    sport_key: str = Field(min_length=1)
    bookmaker: str = Field(min_length=1)
    player_a: str = Field(min_length=1)
    player_b: str = Field(min_length=1)
    price_a: float
    price_b: float
    odds_format: Literal["american", "decimal"] = "american"
    commence_time: AwareDatetime | None
    schedule_observed_at: AwareDatetime
    last_update: AwareDatetime
    collected_at: AwareDatetime
    stale_seconds: float | None = Field(default=None, ge=0)
    is_delayed_reserve: bool = False
    is_cancelled: bool = False
    is_suspended: bool = False
    actual_start: AwareDatetime | None = None

    @field_validator("price_a", "price_b")
    @classmethod
    def price_must_be_finite(cls, value: float) -> float:
        if not pd.notna(value) or value in (float("inf"), float("-inf")):
            raise ValueError("observed prices must be finite")
        return value

    @model_validator(mode="after")
    def validate_atomic_record(self) -> QuoteRecord:
        if self.player_a.casefold().strip() == self.player_b.casefold().strip():
            raise ValueError("the two quote sides must identify different players")
        if self.last_update > self.collected_at:
            raise ValueError("last_update cannot be after collected_at")
        if self.schedule_observed_at > self.collected_at:
            raise ValueError("schedule_observed_at cannot be after collected_at")
        return self


class QuoteStore:
    """Append-only parquet storage for atomic observed-price records."""

    def __init__(self, path: str | Path = DEFAULT_QUOTE_PATH) -> None:
        self.path = Path(path)

    def append(self, quotes: Iterable[QuoteRecord]) -> Path:
        incoming = [QuoteRecord.model_validate(quote) for quote in quotes]
        if not incoming:
            raise ValueError("at least one atomic quote is required")

        records = self.load() if self.path.exists() else []
        records.extend(incoming)
        frame = pd.DataFrame(
            [record.model_dump(mode="python") for record in records]
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            frame.to_parquet(temporary, engine="pyarrow", index=False)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return self.path

    def load(self) -> list[QuoteRecord]:
        frame = pd.read_parquet(self.path, engine="pyarrow")
        return [
            QuoteRecord.model_validate(record)
            for record in frame.to_dict(orient="records")
        ]
