from psmn_rl.analysis.portfolio_seed_pack_validation_loader import load_seed_pack_validation_report


def test_load_seed_pack_validation_report_snapshot() -> None:
    report = load_seed_pack_validation_report()
    assert report.needs_review == ()
    assert report.validated_restart_default == ("round7",)
    assert report.validated_reserve == ("round10", "round5")
    assert report.validated_retired == ("door3_post5", "post_unlock_x5")
    assert tuple(row.candidate for row in report.rows) == (
        "round7",
        "round10",
        "round5",
        "door3_post5",
        "post_unlock_x5",
    )


def test_seed_pack_validation_row_by_candidate() -> None:
    report = load_seed_pack_validation_report()
    row = report.row_by_candidate("round10")
    assert row.validation_bucket == "validated_reserve"
    assert row.policy_bucket == "reserve_same_signal_higher_cost"
    assert row.support_233 == 1.0
