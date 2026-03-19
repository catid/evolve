from psmn_rl.analysis.portfolio_frontier_guard_report_loader import load_frontier_guard_report


def test_load_frontier_guard_report_snapshot() -> None:
    report = load_frontier_guard_report()
    assert report.overall == "pass"
    assert report.active_candidate == "round6"
    assert report.active_candidate_pack == "outputs/reports/portfolio_candidate_pack.json"
    assert report.archived_frozen_pack == "outputs/reports/frozen_benchmark_pack.json"
    assert report.default_restart_prior == "round7"
    assert report.replay_validated_alternate == "round10"
    assert tuple(check.label for check in report.checks) == (
        "consistency_pass",
        "docs_audit_pass",
        "doctor_pass",
    )
    assert all(check.status == "pass" for check in report.checks)


def test_check_by_label() -> None:
    report = load_frontier_guard_report()
    check = report.check_by_label("doctor_pass")
    assert check.status == "pass"
    assert check.detail == "doctor_overall=pass"
