from psmn_rl.analysis.portfolio_frontier_schedule_loader import load_frontier_schedule


def test_load_frontier_schedule_snapshot() -> None:
    report = load_frontier_schedule()
    assert report.active_benchmark == "round6"
    assert tuple(row.candidate for row in report.rows) == (
        "round7",
        "round10",
        "round5",
        "door3_post5",
        "post_unlock_x5",
    )
    assert report.seed_contract["ranking_support"].required_min_success == 1.0
    assert report.seed_contract["ranking_weakness"].required_strictly_above == 0.984375001
    assert report.seed_contract["sentinel"].use == "track_only"


def test_schedule_row_by_candidate() -> None:
    report = load_frontier_schedule()
    row = report.row_by_candidate("round7")
    assert row.action == "run_first"
    assert row.bucket == "default_restart"
    assert row.priority == 1
