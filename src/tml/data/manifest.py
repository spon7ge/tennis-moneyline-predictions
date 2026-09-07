from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(paths: list[Path], root: Path) -> list[dict]:
    rows = []
    for p in paths:
        p = p.resolve()
        rel = str(p.relative_to(root.resolve()))
        rows.append(
            {
                "relative_path": rel.replace("\\", "/"),
                "size_bytes": p.stat().st_size,
                "sha256": file_sha256(p),
            }
        )
    rows.sort(key=lambda r: r["relative_path"])
    return rows


def snapshot_id(manifest: list[dict]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
