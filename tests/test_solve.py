from inference.predict import solve_equation


def test_solves_plain_expression():
    result, err = solve_equation("(4+3)*8")
    assert err is None
    assert result == "56"


def test_solves_decimal_expression():
    result, err = solve_equation("5.25*4")
    assert err is None
    assert float(result) == 21.0


def test_equality_correct():
    result, err = solve_equation("2+2=4")
    assert err is None
    assert "correct" in result


def test_equality_incorrect():
    result, err = solve_equation("2+2=5")
    assert err is None
    assert "incorrect" in result


def test_empty_input_returns_error():
    result, err = solve_equation("")
    assert result is None
    assert err is not None


def test_malformed_expression_returns_error_not_crash():
    result, err = solve_equation("4++")
    assert result is None
    assert err is not None
