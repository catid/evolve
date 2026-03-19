from psmn_rl.analysis.portfolio_frontier_doctor import evaluate_frontier_state


def test_evaluate_frontier_state_pass() -> None:
    result = evaluate_frontier_state(
        active_candidate="round6",
        active_candidate_pack="outputs/reports/portfolio_candidate_pack.json",
        archived_frozen_pack="outputs/reports/frozen_benchmark_pack.json",
        default_restart_prior="round7",
        replay_validated_alternate="round10",
        consistency_overall="pass",
        docs_audit_overall="pass",
    )
    assert result["overall"] == "pass"


def test_evaluate_frontier_state_fail() -> None:
    result = evaluate_frontier_state(
        active_candidate="round7",
        active_candidate_pack="outputs/reports/portfolio_candidate_pack.json",
        archived_frozen_pack="outputs/reports/frozen_benchmark_pack.json",
        default_restart_prior="round7",
        replay_validated_alternate="round10",
        consistency_overall="pass",
        docs_audit_overall="pass",
    )
    assert result["overall"] == "fail"
