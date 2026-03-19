from psmn_rl.analysis.portfolio_frontier_contract_loader import load_frontier_contract


def test_frontier_kit_contract_smoke() -> None:
    contract = load_frontier_contract()
    assert contract.frontier_roles.default_restart_prior == "round7"
    assert contract.frontier_roles.replay_validated_alternate == "round10"
