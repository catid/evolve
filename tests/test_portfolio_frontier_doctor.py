from psmn_rl.analysis.portfolio_frontier_doctor import evaluate_frontier_state


def test_evaluate_frontier_state_pass() -> None:
    result = evaluate_frontier_state(
        active_candidate="round6",
        default_restart_prior="round7",
        replay_validated_alternate="round10",
        consistency_overall="pass",
    )
    assert result["overall"] == "pass"


def test_evaluate_frontier_state_fail() -> None:
    result = evaluate_frontier_state(
        active_candidate="round7",
        default_restart_prior="round7",
        replay_validated_alternate="round10",
        consistency_overall="pass",
    )
    assert result["overall"] == "fail"
