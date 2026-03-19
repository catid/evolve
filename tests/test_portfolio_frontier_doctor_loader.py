from psmn_rl.analysis.portfolio_frontier_doctor_loader import load_frontier_doctor_report


def test_load_frontier_doctor_report_snapshot() -> None:
    report = load_frontier_doctor_report()
    assert report.overall == "pass"
    assert tuple(check.label for check in report.checks) == (
        "active_candidate_round6",
        "active_pack_current",
        "archived_pack_frozen",
        "default_restart_round7",
        "replay_alternate_round10",
        "consistency_pass",
        "docs_audit_pass",
    )
    assert all(check.status == "pass" for check in report.checks)


def test_doctor_check_by_label() -> None:
    report = load_frontier_doctor_report()
    check = report.check_by_label("active_pack_current")
    assert check.status == "pass"
    assert check.detail == "active_candidate_pack=outputs/reports/portfolio_candidate_pack.json"
