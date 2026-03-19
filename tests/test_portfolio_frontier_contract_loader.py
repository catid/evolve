from psmn_rl.analysis.portfolio_frontier_contract_loader import load_frontier_contract


def test_load_frontier_contract_smoke() -> None:
    contract = load_frontier_contract()
    assert contract.benchmark.active_candidate == "round6"
    assert contract.frontier_roles.default_restart_prior == "round7"
    assert contract.frontier_roles.replay_validated_alternate == "round10"
    assert contract.frontier_roles.hold_only_priors == ("round5",)
    assert contract.frontier_roles.retired_priors == ("door3_post5", "post_unlock_x5")


def test_load_frontier_contract_seed_roles() -> None:
    contract = load_frontier_contract()
    assert contract.role_seed("sentinel").lane == "prospective_c"
    assert contract.role_seed("sentinel").seed == 193
    assert contract.role_seed("ranking_support").lane == "prospective_f"
    assert contract.role_seed("ranking_support").seed == 233
    assert contract.differentiator == ("prospective_f:233",)
    assert contract.global_hard == ("prospective_c:193",)
