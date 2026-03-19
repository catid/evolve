from psmn_rl.analysis.portfolio_frontier_replay_loader import load_frontier_replay


def test_load_frontier_replay_snapshot() -> None:
    report = load_frontier_replay()
    assert report.alias_map == {
        "round10": "round10",
        "round10_carry2_post4": "carry2_post4",
        "round10_door2_post4": "door2_post4",
        "round10_post_unlock_x5": "post_unlock_x5",
    }
    assert report.grouped == {
        "advance_for_broader_dev": ("round10",),
        "prune_guardrail_regression": (
            "round10_carry2_post4",
            "round10_door2_post4",
            "round10_post_unlock_x5",
        ),
    }
    assert tuple(row.case_candidate for row in report.rows) == (
        "round10",
        "round10_carry2_post4",
        "round10_door2_post4",
        "round10_post_unlock_x5",
    )


def test_replay_row_by_candidate() -> None:
    report = load_frontier_replay()
    row = report.row_by_candidate("round10_post_unlock_x5")
    assert row.verdict == "prune_guardrail_regression"
    assert row.guardrail_277 == 0.890625
    assert row.support_233 == 1.0
