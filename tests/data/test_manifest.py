import json
from pathlib import Path
from tml.data.manifest import build_manifest, snapshot_id

def test_snapshot_id_stable_and_order_independent(tmp_path: Path):
    a = tmp_path / "b.csv"
    b = tmp_path / "a.csv"
    a.write_text("x\n1\n")
    b.write_text("x\n2\n")
    m1 = build_manifest([a, b], root=tmp_path)
    m2 = build_manifest([b, a], root=tmp_path)
    assert snapshot_id(m1) == snapshot_id(m2)
    assert m1[0]["relative_path"] == "a.csv"
    assert "sha256" in m1[0]
