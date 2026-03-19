from psmn_rl.analysis.portfolio_frontier_docs_audit_loader import load_frontier_docs_audit_report


def test_load_frontier_docs_audit_report_snapshot() -> None:
    report = load_frontier_docs_audit_report()
    assert report.overall == "pass"
    assert report.active_candidate == "round6"
    assert report.active_candidate_pack == "outputs/reports/portfolio_candidate_pack.json"
    assert report.default_restart_prior == "round7"
    assert report.replay_validated_alternate == "round10"
    assert tuple(check.path for check in report.checks) == (
        "README.md",
        "summary.md",
        "report.md",
        "outputs/reports/claim_ledger.md",
    )
    assert all(check.status == "pass" for check in report.checks)
    assert all(check.missing == () for check in report.checks)


def test_check_by_path() -> None:
    report = load_frontier_docs_audit_report()
    check = report.check_by_path("README.md")
    assert check.status == "pass"
    assert check.missing == ()
