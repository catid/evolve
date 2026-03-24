from psmn_rl.analysis.lss_portfolio_campaign import (
    _decision_status,
    _stage1_candidate_pass,
    _selected_stage1_candidates,
    _synthetic_control_rows,
)
from psmn_rl.analysis.lss_deadlock_program import _phase_case
from psmn_rl.analysis.campaign_config import load_campaign_config


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


def test_selected_stage1_candidates_supports_archpilot_track() -> None:
    campaign = {
        "selection": {
            "stage1_fruitful_top_k": 1,
            "stage1_exploratory_top_k": 1,
            "stage1_archpilot_top_k": 1,
        }
    }
    candidate_summaries = [
        {
            "candidate": "round10",
            "track": "fruitful",
            "stage1_pass": True,
            "candidate_mean": 0.91,
            "delta_vs_round6": 0.02,
            "candidate_minus_token": 0.01,
            "candidate_minus_single": 0.01,
        },
        {
            "candidate": "contract_probe",
            "track": "exploratory",
            "stage1_pass": True,
            "candidate_mean": 0.89,
            "delta_vs_round6": 0.01,
            "candidate_minus_token": 0.00,
            "candidate_minus_single": 0.00,
        },
        {
            "candidate": "phase_memory",
            "track": "archpilot",
            "stage1_pass": True,
            "candidate_mean": 0.90,
            "delta_vs_round6": 0.015,
            "candidate_minus_token": 0.005,
            "candidate_minus_single": 0.005,
        },
    ]
    selected_by_track, selected = _selected_stage1_candidates(campaign, candidate_summaries)
    assert selected_by_track == {
        "fruitful": ["round10"],
        "exploratory": ["contract_probe"],
        "archpilot": ["phase_memory"],
    }
    assert selected == ["round10", "contract_probe", "phase_memory"]


def test_stage1_candidate_pass_respects_min_dev_gain_and_failure_bar() -> None:
    campaign = {"selection": {"min_dev_gain": 0.01}}
    round6 = {"sare_mean": 0.8333333333333334, "sare_failures": 1}

    tied = {
        "candidate_mean": 0.8333333333333334,
        "candidate_failures": 1,
        "incomplete": False,
    }
    assert _stage1_candidate_pass(campaign, round6, tied) is False

    improved = {
        "candidate_mean": 0.8433333333333334,
        "candidate_failures": 1,
        "incomplete": False,
    }
    assert _stage1_candidate_pass(campaign, round6, improved) is True

    regressed_failures = {
        "candidate_mean": 0.90,
        "candidate_failures": 2,
        "incomplete": False,
    }
    assert _stage1_candidate_pass(campaign, round6, regressed_failures) is False


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


def test_decision_status_supports_generic_next_portfolio_language() -> None:
    campaign = {
        "current_canonical_name": "round6",
        "generic_decision_language": True,
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
    exploratory = {"overall_boundary": "clearly negative"}
    gate_payload = {"verdict": "PASS: thaw consideration allowed"}
    assert (
        _decision_status(campaign, holdout, anti_regression, route, stability, exploratory, gate_payload)
        == "active benchmark confirmed and internal DoorKey frontier strengthened"
    )


def test_decision_status_supports_explicit_label_overrides() -> None:
    campaign = {
        "current_canonical_name": "round6",
        "decision_strings": {
            "replace": "challenger replaces the active benchmark",
            "confirm": "active benchmark confirmed and frontier clarified",
            "narrow": "active benchmark remains and envelope stays narrow",
            "narrow_state": "benchmark/frontier state needs narrowing",
        },
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
    exploratory = {"overall_boundary": "clearly negative"}
    gate_payload = {"verdict": "PASS: thaw consideration allowed"}
    assert (
        _decision_status(campaign, holdout, anti_regression, route, stability, exploratory, gate_payload)
        == "active benchmark confirmed and frontier clarified"
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


def test_deadlock_phase_case_uses_current_matched_control_roots() -> None:
    campaign = load_campaign_config("configs/experiments/lss_deadlock_contract_program/campaign.yaml")
    case = _phase_case(campaign, "prospective_c", 193)

    assert str(case.token_dense.run_dir).endswith(
        "outputs/experiments/lss_successor_stress/stage2_controls/round6/prospective_c/seed_193/kl_lss_token_dense"
    )
    assert str(case.single_expert.run_dir).endswith(
        "outputs/experiments/lss_successor_stress/stage2_controls/round6/prospective_c/seed_193/kl_lss_single_expert"
    )
