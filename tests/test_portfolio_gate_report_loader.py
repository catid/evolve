from psmn_rl.analysis.portfolio_gate_report_loader import load_portfolio_gate_report


def test_load_portfolio_gate_report_snapshot() -> None:
    report = load_portfolio_gate_report()
    assert report.candidate_pack == "outputs/reports/portfolio_candidate_pack.json"
    assert report.frozen_pack == "outputs/reports/round6_current_benchmark_pack.json"
    assert report.mode == "pack"
    assert report.verdict == "PASS: thaw consideration allowed"
    assert report.candidate_pack_validation[0].name == "pack_type"
    assert report.candidate_pack_validation[0].status == "PASS"
    assert report.frozen_pack_validation[0].name == "pack_type"
    assert report.frozen_pack_validation[0].status == "PASS"
    assert tuple(check.name for check in report.checks[:3]) == (
        "evaluation_shape",
        "task",
        "evaluation_path",
    )


def test_gate_report_check_by_name() -> None:
    report = load_portfolio_gate_report()
    check = report.check_by_name("combined_picture_mean")
    assert check.status == "PASS"
    assert "candidate combined SARE mean" in check.detail
