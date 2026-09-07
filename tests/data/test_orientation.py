from tml.data.orientation import assign_orientation


def test_orientation_independent_of_winner_loser_order() -> None:
    a1, b1 = assign_orientation("m1", "100", "200")
    a2, b2 = assign_orientation("m1", "200", "100")
    assert (a1, b1) == (a2, b2)
