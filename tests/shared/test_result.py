from tml.shared.result import Ok, Err


def test_ok_and_err_discriminate():
    assert Ok(1).ok is True and Ok(1).value == 1
    assert Err("x").ok is False and Err("x").error == "x"
