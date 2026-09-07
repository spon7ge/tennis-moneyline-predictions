from __future__ import annotations

import re

COMPLETION_STATUSES = frozenset(
    {"completed", "retirement", "walkover", "default", "abandoned", "unknown"}
)

_RETIREMENT = re.compile(r"\bRET\b", re.IGNORECASE)
_WALKOVER = re.compile(r"\bW/O\b|\bwalkover\b", re.IGNORECASE)
_DEFAULT = re.compile(r"\bDEF\b", re.IGNORECASE)
_ABANDONED = re.compile(r"\babd\b", re.IGNORECASE)
_SCORE_PATTERN = re.compile(r"\d-\d")


def parse_completion_status(score: str | None) -> str:
    """Derive completion status from Sackmann-style score text."""
    if score is None or score.strip() == "":
        return "unknown"

    text = score.strip()

    if _RETIREMENT.search(text):
        return "retirement"
    if _WALKOVER.search(text):
        return "walkover"
    if _DEFAULT.search(text):
        return "default"
    if _ABANDONED.search(text):
        return "abandoned"
    if _SCORE_PATTERN.search(text):
        return "completed"

    return "unknown"
