from __future__ import annotations

import hashlib


def assign_orientation(
    match_id: str, player_u: str, player_v: str
) -> tuple[str, str]:
    u, v = sorted([str(player_u), str(player_v)])
    digest = hashlib.sha256(f"orientation_v1|{match_id}|{u}|{v}".encode()).hexdigest()
    if int(digest[-1], 16) % 2 == 1:
        return u, v
    return v, u
