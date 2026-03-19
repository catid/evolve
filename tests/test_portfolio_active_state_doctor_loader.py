from psmn_rl.analysis.portfolio_active_state_doctor_loader import load_active_state_doctor_report


def test_load_active_state_doctor_report_snapshot() -> None:
    report = load_active_state_doctor_report()
    assert report.overall == "pass"
    assert tuple(check.label for check in report.checks) == (
        "candidate_round6",
        "candidate_pack_current",
        "candidate_archived_pack_frozen",
        "candidate_eval_doorkey_external64",
        "gate_targets_current_active_pack",
        "gate_pack_mode_pass",
        "gate_combined_picture_pass",
        "contract_matches_active_roles",
        "frontier_doctor_pass",
    )
    assert all(check.status == "pass" for check in report.checks)


def test_active_state_doctor_check_by_label() -> None:
    report = load_active_state_doctor_report()
    check = report.check_by_label("gate_pack_mode_pass")
    assert check.status == "pass"
    assert "PASS: thaw consideration allowed" in check.detail
