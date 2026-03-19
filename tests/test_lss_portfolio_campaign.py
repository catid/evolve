from psmn_rl.analysis.lss_portfolio_campaign import (
    _decision_status,
    _selected_stage1_candidates,
    _synthetic_control_rows,
)


def test_selected_stage1_candidates_balances_tracks() -> None:
    campaign = {
        "selection": {
            "stage1_fruitful_top_k": 2,
            "stage1_exploratory_top_k": 1,
        }
    }
    candidate_summaries = [
        {
            "candidate": "round7",
            "track": "fruitful",
            "stage1_pass": True,
            "candidate_mean": 0.91,
            "delta_vs_round6": 0.02,
            "candidate_minus_token": 0.00,
            "candidate_minus_single": 0.00,
        },
        {
            "candidate": "round10",
            "track": "fruitful",
            "stage1_pass": True,
            "candidate_mean": 0.90,
            "delta_vs_round6": 0.01,
            "candidate_minus_token": -0.01,
            "candidate_minus_single": -0.01,
        },
        {
            "candidate": "round12",
            "track": "fruitful",
            "stage1_pass": True,
            "candidate_mean": 0.89,
            "delta_vs_round6": 0.00,
            "candidate_minus_token": -0.02,
            "candidate_minus_single": -0.02,
        },
        {
            "candidate": "conf_post4",
            "track": "exploratory",
            "stage1_pass": True,
            "candidate_mean": 0.88,
            "delta_vs_round6": 0.00,
            "candidate_minus_token": -0.01,
            "candidate_minus_single": -0.01,
        },
        {
            "candidate": "bridge_post4",
            "track": "exploratory",
            "stage1_pass": False,
            "candidate_mean": 0.80,
            "delta_vs_round6": -0.08,
            "candidate_minus_token": -0.09,
            "candidate_minus_single": -0.09,
        },
    ]
    selected_by_track, selected = _selected_stage1_candidates(campaign, candidate_summaries)
    assert selected_by_track == {"fruitful": ["round7", "round10"], "exploratory": ["conf_post4"]}
    assert selected == ["round7", "round10", "conf_post4"]


def test_decision_status_confirms_round6_when_program_strengthens_it() -> None:
    campaign = {
        "current_canonical_name": "round6",
        "selection": {
            "pack_gate_required_verdict": "PASS: thaw consideration allowed",
            "control_eps": 0.02,
        },
    }
    holdout = {
        "best_candidate": None,
        "round6_summary": {"sare_mean": 0.90, "token_mean": 0.89, "single_mean": 0.89},
    }
    anti_regression = {
        "challenger_pass": False,
        "round6_summary": {"sare_mean": 1.00, "token_mean": 0.89, "single_mean": 0.89},
    }
    route = {"round6_pass": True, "challenger_pass": False}
    stability = {"round6_pass": True, "challenger_pass": False}
    exploratory = {"overall_boundary": "still DoorKey-only"}
    gate_payload = {"verdict": "PASS: thaw consideration allowed"}
    assert (
        _decision_status(campaign, holdout, anti_regression, route, stability, exploratory, gate_payload)
        == "round6 confirmed as active DoorKey benchmark and internal DoorKey benchmark state strengthened"
    )


def test_decision_status_keeps_envelope_narrow_without_extra_strength() -> None:
    campaign = {
        "current_canonical_name": "round6",
        "selection": {
            "pack_gate_required_verdict": "PASS: thaw consideration allowed",
            "control_eps": 0.02,
        },
    }
    holdout = {
        "best_candidate": None,
        "round6_summary": {"sare_mean": 0.84, "token_mean": 0.89, "single_mean": 0.89},
    }
    anti_regression = {
        "challenger_pass": False,
        "round6_summary": {"sare_mean": 1.00, "token_mean": 0.89, "single_mean": 0.89},
    }
    route = {"round6_pass": True, "challenger_pass": False}
    stability = {"round6_pass": True, "challenger_pass": False}
    exploratory = {"overall_boundary": "clearly negative"}
    gate_payload = {"verdict": "PASS: thaw consideration allowed"}
    assert (
        _decision_status(campaign, holdout, anti_regression, route, stability, exploratory, gate_payload)
        == "round6 remains active benchmark and envelope stays narrow"
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


def test_synthetic_control_rows_reconstruct_token_and_single_scores() -> None:
    sare_rows = [
        {
            "candidate": "round10",
            "lane": "prospective_c",
            "seed": 193,
            "label": "kl_lss_sare",
            "final_greedy_success": 0.0,
            "delta_vs_token_dense": -1.0,
            "delta_vs_single_expert": 0.0,
            "run_dir": "synthetic/run",
        },
        {
            "candidate": "round10",
            "lane": "prospective_d",
            "seed": 197,
            "label": "kl_lss_sare",
            "final_greedy_success": 1.0,
            "delta_vs_token_dense": 0.0,
            "delta_vs_single_expert": 0.0,
            "run_dir": "synthetic/run",
        },
    ]
    controls = _synthetic_control_rows(sare_rows)
    assert [row["label"] for row in controls] == [
        "kl_lss_token_dense",
        "kl_lss_single_expert",
        "kl_lss_token_dense",
        "kl_lss_single_expert",
    ]
    assert controls[0]["final_greedy_success"] == 1.0
    assert controls[1]["final_greedy_success"] == 0.0
    assert controls[2]["final_greedy_success"] == 1.0
    assert controls[3]["final_greedy_success"] == 1.0
