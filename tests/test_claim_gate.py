from psmn_rl.analysis.claim_gate import evaluate_claim_gate


def _manifest() -> dict:
    return {
        "thaw_gate": {
            "required_controls": [
                "recovered_token_dense",
                "kl_lss_token_dense",
                "kl_lss_single_expert",
                "baseline_sare",
            ],
            "required_evaluation": {
                "path_key": "external_policy_diagnostics",
                "episodes": 64,
                "task": "DoorKey",
            },
            "retry_block": {
                "candidate_sare_mean_must_exceed": 0.3125,
                "candidate_sare_complete_seed_failures_must_be_lte": 1,
            },
            "combined_picture": {
                "candidate_sare_mean_must_be_gte": 0.7122,
                "candidate_sare_complete_seed_failures_must_be_lte": 1,
            },
            "disallowed_claim_keys": [
                "ppo_only_routed_win",
                "specifically_multi_expert_routed_advantage",
            ],
        }
    }


def test_claim_gate_fails_for_current_frozen_baseline_like_candidate() -> None:
    candidate = {
        "task": "DoorKey",
        "evaluation": {"path_key": "external_policy_diagnostics", "episodes": 64},
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": [
            "recovered_token_dense",
            "kl_lss_token_dense",
            "kl_lss_single_expert",
            "baseline_sare",
            "kl_lss_sare",
        ],
        "metrics": {
            "retry_block": {
                "kl_lss_sare": {"mean": 0.3125, "complete_seed_failures": 1},
                "kl_lss_single_expert": {"mean": 0.4635, "complete_seed_failures": 0},
            },
            "combined": {
                "kl_lss_sare": {"mean": 0.7122, "complete_seed_failures": 1},
                "kl_lss_single_expert": {"mean": 0.6862, "complete_seed_failures": 1},
            },
        },
    }
    verdict, checks = evaluate_claim_gate(_manifest(), candidate)
    assert verdict == "FAIL: claim remains frozen"
    assert any(check.name == "retry_block_improvement" and check.status == "FAIL" for check in checks)
    assert any(check.name == "retry_block_vs_single_expert" and check.status == "FAIL" for check in checks)


def test_claim_gate_is_inconclusive_when_required_controls_are_missing() -> None:
    candidate = {
        "task": "DoorKey",
        "evaluation": {"path_key": "external_policy_diagnostics", "episodes": 64},
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": ["recovered_token_dense", "kl_lss_sare"],
        "metrics": {},
    }
    verdict, checks = evaluate_claim_gate(_manifest(), candidate)
    assert verdict == "INCONCLUSIVE: missing prerequisites"
    assert any(check.name == "fairness_controls" and check.status == "INCONCLUSIVE" for check in checks)


def test_claim_gate_can_pass_when_candidate_clears_all_bars() -> None:
    candidate = {
        "task": "DoorKey",
        "evaluation": {"path_key": "external_policy_diagnostics", "episodes": 64},
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": [
            "recovered_token_dense",
            "kl_lss_token_dense",
            "kl_lss_single_expert",
            "baseline_sare",
            "kl_lss_sare",
        ],
        "metrics": {
            "retry_block": {
                "kl_lss_sare": {"mean": 0.6, "complete_seed_failures": 0},
                "kl_lss_single_expert": {"mean": 0.5, "complete_seed_failures": 0},
            },
            "combined": {
                "kl_lss_sare": {"mean": 0.75, "complete_seed_failures": 0},
                "kl_lss_single_expert": {"mean": 0.68, "complete_seed_failures": 1},
            },
        },
    }
    verdict, _checks = evaluate_claim_gate(_manifest(), candidate)
    assert verdict == "PASS: thaw consideration allowed"
