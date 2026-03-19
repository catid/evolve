from psmn_rl.analysis.portfolio_frontier_kit_loader import load_frontier_kit


def test_load_frontier_kit_snapshot() -> None:
    kit = load_frontier_kit()
    assert kit.benchmark.active_candidate == "round6"
    assert kit.benchmark.active_candidate_pack == "outputs/reports/portfolio_candidate_pack.json"
    assert kit.benchmark.archived_frozen_pack == "outputs/reports/frozen_benchmark_pack.json"
    assert kit.frontier_roles.default_restart_prior == "round7"
    assert kit.frontier_roles.replay_validated_alternate == "round10"
    assert kit.next_restart.primary == "round7"
    assert kit.next_restart.secondary == "round10"
    assert tuple(row.candidate for row in kit.ordered_queue) == (
        "round7",
        "round10",
        "round5",
        "door3_post5",
        "post_unlock_x5",
    )
    assert kit.promotion_rules.broader_dev_only_after_seed_clear is True
    assert kit.seed_contract["sentinel"].mode == "track_only"


def test_queue_row_by_candidate() -> None:
    kit = load_frontier_kit()
    row = kit.queue_row_by_candidate("round10")
    assert row.action == "run_second_if_needed"
    assert row.bucket == "validated_alternate"
    assert row.priority == 2
