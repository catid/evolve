from psmn_rl.analysis.lss_portfolio_seed_pack import _tier


def test_tier_restart_default() -> None:
    assert _tier("round7", ["round7"], ["round10", "round5"], ["door3_post5"]) == "restart_default"


def test_tier_reserve() -> None:
    assert _tier("round10", ["round7"], ["round10", "round5"], ["door3_post5"]) == "reserve"


def test_tier_retired() -> None:
    assert _tier("door3_post5", ["round7"], ["round10", "round5"], ["door3_post5"]) == "retired"
