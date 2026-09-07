import pytest
from tml.shared.invariant import invariant


def test_invariant_passes():
    invariant(True, "ok")


def test_invariant_raises():
    with pytest.raises(RuntimeError, match="Invariant violated"):
        invariant(False, "ratings must be finite")
