import json
from datetime import datetime, timezone

from tml.odds.client import fetch_h2h
from tml.shared.config import get_settings
from tml.shared.result import Err, Ok


class _Response:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_fetch_h2h_missing_key_returns_safe_err(monkeypatch) -> None:
    # An explicit empty environment value must override a developer's .env file.
    monkeypatch.setenv("PARLAY_API_KEY", "")
    get_settings.cache_clear()

    result = fetch_h2h()

    assert isinstance(result, Err)
    assert "PARLAY_API_KEY" in result.error


def test_fetch_h2h_validates_api_json_and_uses_header_key(monkeypatch) -> None:
    secret = "unit-test-secret"
    monkeypatch.setenv("PARLAY_API_KEY", secret)
    get_settings.cache_clear()
    captured_request = None

    def fake_urlopen(request, timeout):
        nonlocal captured_request
        captured_request = request
        assert timeout == 30.0
        return _Response(
            [
                {
                    "id": "evt-1",
                    "sport_key": "tennis_atp",
                    "commence_time": "2026-09-06T20:00:00Z",
                    "home_team": "Player A",
                    "away_team": "Player B",
                    "bookmakers": [
                        {
                            "key": "pinnacle",
                            "last_update": "2026-09-06T19:50:00Z",
                            "last_update_ms": 1788724200000,
                            "stale_seconds": 10,
                            "topped_up": True,
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Player A", "price": -110},
                                        {"name": "Player B", "price": -105},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = fetch_h2h(collected_at=datetime(2026, 9, 6, 19, 51, tzinfo=timezone.utc))

    assert isinstance(result, Ok)
    assert len(result.value) == 1
    assert result.value[0].is_delayed_reserve is True
    assert result.value[0].price_a == -110
    assert captured_request.get_header("X-api-key") == secret
    assert secret not in captured_request.full_url


def test_fetch_h2h_matches_outcome_names_case_insensitively(monkeypatch) -> None:
    monkeypatch.setenv("PARLAY_API_KEY", "unit-test-secret")
    get_settings.cache_clear()

    def fake_urlopen(request, timeout):
        return _Response(
            [
                {
                    "id": "evt-case",
                    "sport_key": "tennis_atp",
                    "commence_time": "2026-09-06T20:00:00Z",
                    "home_team": "Arthur Gea",
                    "away_team": "Botic Van de Zandschulp",
                    "bookmakers": [
                        {
                            "key": "prophetx",
                            "last_update": "2026-09-06T19:50:00Z",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Arthur Gea", "price": 154},
                                        {
                                            "name": "Botic Van De Zandschulp",
                                            "price": -158,
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = fetch_h2h(collected_at=datetime(2026, 9, 6, 19, 51, tzinfo=timezone.utc))

    assert isinstance(result, Ok)
    assert len(result.value) == 1
    assert result.value[0].price_a == 154
    assert result.value[0].price_b == -158


def test_fetch_h2h_invalid_api_json_returns_safe_err(monkeypatch) -> None:
    monkeypatch.setenv("PARLAY_API_KEY", "unit-test-secret")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda _request, timeout: _Response([{"id": "missing-required-fields"}]),
    )

    result = fetch_h2h()

    assert isinstance(result, Err)
    assert "invalid odds response" in result.error.lower()
