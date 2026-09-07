from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings read from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    parlay_api_key: str | None = Field(default=None, validation_alias="PARLAY_API_KEY")
    parlay_base_url: HttpUrl = Field(
        default=HttpUrl("https://parlay-api.com/v1"),
        validation_alias="PARLAY_BASE_URL",
    )
    parlay_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        validation_alias="PARLAY_TIMEOUT_SECONDS",
    )
    tml_data_dir: Path = Field(default=Path("./tml-data"), validation_alias="TML_DATA_DIR")


@lru_cache
def get_settings() -> Settings:
    return Settings()
