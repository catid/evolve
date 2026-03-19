from psmn_rl.analysis.lss_portfolio_structural_probe import _support_status


def test_support_status_clean() -> None:
    assert _support_status(1.0, 1.0) == "measured_clean_support"


def test_support_status_regression() -> None:
    assert _support_status(0.625, 1.0) == "measured_support_regression"
