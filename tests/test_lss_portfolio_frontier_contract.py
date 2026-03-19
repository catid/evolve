from psmn_rl.analysis.lss_portfolio_frontier_contract import _singleton


def test_singleton_returns_only_value() -> None:
    assert _singleton(["round7"]) == "round7"


def test_singleton_returns_none_for_empty() -> None:
    assert _singleton([]) is None


def test_singleton_returns_none_for_multiple() -> None:
    assert _singleton(["round7", "round10"]) is None
