from psmn_rl.analysis.lss_forensic_atlas import _build_decision_memo, _build_scorecard


def test_scorecard_ends_with_bounded_retry_not_justified_for_split_signals() -> None:
    casebook_rows = [
        {
            "row_type": "episode_summary",
            "group": "weak",
            "seed": "47",
            "label": "kl_lss_sare",
            "success_rate": "0.0",
        },
        {
            "row_type": "episode_summary",
            "group": "weak",
            "seed": "47",
            "label": "kl_lss_single_expert",
            "success_rate": "0.5",
        },
    ]
    round_rows = [
        {
            "group": "weak",
            "round": "4",
            "collection_disagreement_rate": "0.9",
            "collection_unique_state_ratio": "0.003",
            "route_pair_dominant_mean": "0.6",
        },
        {
            "group": "strong",
            "round": "4",
            "collection_disagreement_rate": "0.05",
            "collection_unique_state_ratio": "0.08",
            "route_pair_dominant_mean": "0.3",
        },
    ]
    route_rows = [
        {
            "row_type": "weak_single_gap",
            "teacher_match_gap": "0.2",
        },
        {
            "row_type": "counterfactual_phase",
            "fixed_router_action_change_rate": "0.7",
        },
    ]
    report, rows = _build_scorecard(casebook_rows, round_rows, route_rows)
    assert "bounded retry not justified" in report
    assert any(row.get("resume_gate_status") == "bounded retry not justified" for row in rows)


def test_decision_memo_stays_frozen_when_scorecard_blocks_retry() -> None:
    memo = _build_decision_memo([{"row_type": "verdict", "resume_gate_status": "bounded retry not justified"}])
    assert "stay frozen as-is" in memo
    assert "Retry status: not run" in memo
