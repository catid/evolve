from psmn_rl.analysis.lss_expansion_mega_program import (
    _decision_status,
    _overall_exploratory_boundary,
    _exploratory_summary,
    _verification_pass,
)


def test_verification_pass_requires_consistent_reruns() -> None:
    campaign = {"selection": {"verification_tolerance": 0.05, "verification_margin_floor": -0.02}}
    original = {
        "candidate_mean": 0.84,
        "round6_mean": 0.83,
        "delta_vs_round6": 0.01,
        "candidate_failures": 0.0,
    }
    rerun_a = {"candidate_mean": 0.83, "candidate_failures": 0.0}
    rerun_b = {"candidate_mean": 0.82, "candidate_failures": 0.0}
    assert _verification_pass(campaign, original, rerun_a, rerun_b, 0.0)

    rerun_b["candidate_mean"] = 0.74
    assert not _verification_pass(campaign, original, rerun_a, rerun_b, 0.0)


def test_overall_exploratory_boundary_prefers_foothold_then_negative() -> None:
    assert _overall_exploratory_boundary([{"boundary": "clearly negative"}, {"boundary": "clearly negative"}]) == "clearly negative"
    assert _overall_exploratory_boundary([{"boundary": "still DoorKey-only"}, {"boundary": "clearly negative"}]) == "still DoorKey-only"
    assert _overall_exploratory_boundary([{"boundary": "exploratory foothold only"}]) == "exploratory foothold only"


def test_exploratory_summary_respects_single_expert_control() -> None:
    rows = [
        {"candidate": "round6", "label": "kl_lss_sare", "final_greedy_success": 0.30},
        {"candidate": "round6", "label": "kl_lss_sare", "final_greedy_success": 0.30},
        {"candidate": "round6", "label": "kl_lss_token_dense", "final_greedy_success": 0.28},
        {"candidate": "round6", "label": "kl_lss_token_dense", "final_greedy_success": 0.28},
        {"candidate": "round6", "label": "kl_lss_single_expert", "final_greedy_success": 0.40},
        {"candidate": "round6", "label": "kl_lss_single_expert", "final_greedy_success": 0.40},
    ]
    summary = _exploratory_summary(rows, "round6")
    assert summary["single_mean"] == 0.40
    assert summary["boundary"] == "still DoorKey-only"


def test_exploratory_summary_marks_missing_single_expert_as_unavailable() -> None:
    rows = [
        {"candidate": "round6", "label": "kl_lss_sare", "final_greedy_success": 0.30},
        {"candidate": "round6", "label": "kl_lss_sare", "final_greedy_success": 0.30},
        {"candidate": "round6", "label": "kl_lss_token_dense", "final_greedy_success": 0.28},
        {"candidate": "round6", "label": "kl_lss_token_dense", "final_greedy_success": 0.28},
    ]
    summary = _exploratory_summary(rows, "round6")
    assert summary["single_available"] is False
    assert summary["single_mean"] is None
    assert summary["boundary"] == "exploratory foothold only"


def test_decision_status_confirms_round6_when_broader_program_strengthens_it() -> None:
    campaign = {
        "current_canonical_name": "round6",
        "selection": {
            "pack_gate_required_verdict": "PASS: thaw consideration allowed",
            "control_eps": 0.02,
        },
    }
    holdout = {
        "best_candidate": None,
        "round6_summary": {"sare_mean": 0.89, "token_mean": 0.88, "single_mean": 0.88},
    }
    anti_regression = {
        "challenger_pass": False,
        "round6_summary": {"sare_mean": 1.0, "token_mean": 0.88, "single_mean": 0.89},
    }
    route = {"round6_pass": True, "challenger_pass": False}
    stability = {"round6_pass": True, "challenger_pass": False}
    exploratory = {"overall_boundary": "still DoorKey-only"}
    gate_payload = {"verdict": "PASS: thaw consideration allowed"}
    assert (
        _decision_status(campaign, holdout, anti_regression, route, stability, exploratory, gate_payload)
        == "round6 confirmed as active DoorKey benchmark and internal DoorKey envelope strengthened"
    )


def test_decision_status_promotes_challenger_only_when_full_path_clears() -> None:
    campaign = {
        "current_canonical_name": "round6",
        "selection": {
            "pack_gate_required_verdict": "PASS: thaw consideration allowed",
            "control_eps": 0.02,
        },
    }
    holdout = {
        "best_candidate": "round7",
        "round6_summary": {"sare_mean": 0.88, "token_mean": 0.88, "single_mean": 0.88},
    }
    anti_regression = {
        "challenger_pass": True,
        "round6_summary": {"sare_mean": 1.0, "token_mean": 0.88, "single_mean": 0.89},
    }
    route = {"round6_pass": True, "challenger_pass": True}
    stability = {"round6_pass": True, "challenger_pass": True}
    exploratory = {"overall_boundary": "still DoorKey-only"}
    gate_payload = {"verdict": "PASS: thaw consideration allowed"}
    assert (
        _decision_status(campaign, holdout, anti_regression, route, stability, exploratory, gate_payload)
        == "challenger replaces round6 as active DoorKey benchmark"
    )

    gate_payload = {"verdict": "FAIL: claim remains frozen"}
    assert (
        _decision_status(campaign, holdout, anti_regression, route, stability, exploratory, gate_payload)
        == "benchmark state needs narrowing"
    )
