from psmn_rl.analysis.portfolio_active_state_doctor import evaluate_active_state


def test_evaluate_active_state_pass() -> None:
    result = evaluate_active_state()
    assert result["overall"] == "pass"


def test_evaluate_active_state_has_expected_labels() -> None:
    result = evaluate_active_state()
    assert [check["label"] for check in result["checks"]] == [
        "candidate_round6",
        "candidate_pack_current",
        "candidate_archived_pack_frozen",
        "candidate_eval_doorkey_external64",
        "gate_targets_current_active_pack",
        "gate_pack_mode_pass",
        "gate_combined_picture_pass",
        "contract_matches_active_roles",
        "frontier_doctor_pass",
    ]
