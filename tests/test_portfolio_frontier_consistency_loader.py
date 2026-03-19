from psmn_rl.analysis.portfolio_frontier_consistency_loader import load_frontier_consistency_report


def test_load_frontier_consistency_report_snapshot() -> None:
    report = load_frontier_consistency_report()
    assert report.overall == "pass"
    assert tuple(check.label for check in report.checks) == (
        "active_benchmark_round6",
        "restart_default_round7",
        "retired_priors_match",
        "round7_scorer_advances",
        "round10_replay_advances",
        "round5_hold_only",
        "door3_retired_consistent",
        "post_unlock_x5_retired_consistent",
    )
    assert all(check.status == "pass" for check in report.checks)


def test_consistency_check_by_label() -> None:
    report = load_frontier_consistency_report()
    check = report.check_by_label("round10_replay_advances")
    assert check.status == "pass"
    assert "round10=advance_for_broader_dev" in check.detail
