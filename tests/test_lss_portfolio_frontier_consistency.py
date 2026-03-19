from psmn_rl.analysis.lss_portfolio_frontier_consistency import _check


def test_check_pass() -> None:
    row = _check("x", True, "ok")
    assert row["status"] == "pass"


def test_check_fail() -> None:
    row = _check("x", False, "bad")
    assert row["status"] == "fail"
