from psmn_rl.analysis.portfolio_seed_pack_scorer_loader import load_seed_pack_scorer_report


def test_load_seed_pack_scorer_report_snapshot() -> None:
    report = load_seed_pack_scorer_report()
    assert report.grouped == {
        "advance_for_broader_dev": ("round7", "round10"),
        "hold_seed_clean_but_below_incumbent": ("round5",),
        "prune_guardrail_regression": ("post_unlock_x5",),
        "prune_support_regression": ("door3_post5",),
    }
    assert report.incumbent_269 == 0.984375
    assert tuple(row.candidate for row in report.rows) == (
        "round7",
        "round10",
        "round5",
        "door3_post5",
        "post_unlock_x5",
    )


def test_seed_pack_scorer_row_by_candidate() -> None:
    report = load_seed_pack_scorer_report()
    row = report.row_by_candidate("round5")
    assert row.verdict == "hold_seed_clean_but_below_incumbent"
    assert row.dev_mean == 0.8368055555555556
    assert row.guardrail_277 == 1.0
