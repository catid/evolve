from psmn_rl.analysis.lss_portfolio_frontier_replay import ALIAS_MAP


def test_alias_map_contains_measured_structural_lines() -> None:
    assert ALIAS_MAP["round10_carry2_post4"] == "carry2_post4"
    assert ALIAS_MAP["round10_door2_post4"] == "door2_post4"
    assert ALIAS_MAP["round10_post_unlock_x5"] == "post_unlock_x5"
