from psmn_rl.analysis.portfolio_seed_pack_doctor_loader import load_seed_pack_doctor_report


def test_load_seed_pack_doctor_report_snapshot() -> None:
    report = load_seed_pack_doctor_report()
    assert report.overall == "pass"
    assert tuple(check.label for check in report.checks) == (
        "active_candidate_round6",
        "active_pack_matches_contract",
        "archived_pack_matches_contract",
        "restart_default_round7",
        "reserve_priors_match_contract_roles",
        "retired_priors_match_contract_roles",
        "support_status_matches_contract",
        "generated_from_active_pack_current",
        "seed_roles_match_contract_thresholds",
        "validation_buckets_match_frontier_roles",
        "scorer_verdicts_match_frontier_roles",
    )
    assert all(check.status == "pass" for check in report.checks)


def test_seed_pack_doctor_check_by_label() -> None:
    report = load_seed_pack_doctor_report()
    check = report.check_by_label("restart_default_round7")
    assert check.status == "pass"
    assert check.detail == "restart_default=('round7',)"
