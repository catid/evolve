from psmn_rl.analysis.portfolio_frontier_manifest_loader import load_frontier_manifest


def test_load_frontier_manifest_snapshot() -> None:
    report = load_frontier_manifest()
    assert report.active == ("round6",)
    assert report.default_restart == ("round7",)
    assert report.hold_only == ("round5",)
    assert report.replay_validated_alternates == ("round10",)
    assert report.retired == ("door3_post5", "post_unlock_x5")
    assert report.seed_clean_unconfirmed == ()


def test_manifest_row_by_candidate() -> None:
    report = load_frontier_manifest()
    row = report.row_by_candidate("round10")
    assert row.manifest_bucket == "replay_validated_alternate"
    assert row.replay_verdict == "advance_for_broader_dev"
    assert row.scorer_verdict == "advance_for_broader_dev"
