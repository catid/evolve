from collections import Counter

from psmn_rl.analysis.campaign_config import load_campaign_config


def test_next_mega_portfolio_campaign_extends_base_config() -> None:
    campaign = load_campaign_config("configs/experiments/lss_next_mega_portfolio_campaign/campaign.yaml")
    track_counts = Counter(str(meta["track"]) for meta in campaign["candidates"].values())

    assert campaign["name"] == "lss_next_mega_portfolio_campaign"
    assert track_counts == {"fruitful": 40, "exploratory": 40}
    assert len(campaign["blocks"]["dev"]) == 4
    assert len(campaign["blocks"]["holdout"]) == 4
    assert len(campaign["blocks"]["healthy"]) == 4
    assert len(campaign["blocks"]["hard_seed"]) == 4
    assert tuple(campaign["route_cases"].keys()) == ("dev", "holdout", "healthy", "hard")
    assert tuple(campaign["stability_cases"].keys()) == ("dev", "holdout", "healthy", "hard")
    assert campaign["reports"]["decision_memo"] == "outputs/reports/next_mega_portfolio_decision_memo.md"
    assert "outputs/experiments/lss_next_portfolio_campaign/stage1_screening" in campaign["reuse_roots"]["stage1_sare"]


def test_next_round_campaign_uses_subset_and_current_gate_reference_pack() -> None:
    campaign = load_campaign_config("configs/experiments/lss_next_round_campaign/campaign.yaml")
    track_counts = Counter(str(meta["track"]) for meta in campaign["candidates"].values())

    assert campaign["name"] == "lss_next_round_campaign"
    assert track_counts == {"fruitful": 24, "exploratory": 24}
    assert len(campaign["candidates"]) == 48
    assert campaign["frozen_pack"] == "outputs/reports/round6_current_benchmark_pack.json"
    assert campaign["current_decision_memo"] == "outputs/reports/next_mega_portfolio_decision_memo.md"
    assert campaign["reports"]["registration"] == "outputs/reports/next_round_campaign_registration.md"
    assert campaign["reports"]["stage1_fruitful_report"] == "outputs/reports/next_round_stage2_screening_exploit.md"
    assert campaign["reports"]["stage1_exploratory_report"] == "outputs/reports/next_round_stage2_screening_explore.md"
    assert campaign["reports"]["decision_memo"] == "outputs/reports/next_round_decision_memo.md"
    assert tuple(campaign["route_cases"].keys()) == ("dev", "holdout", "healthy", "hard")
    assert tuple(campaign["stability_cases"].keys()) == ("dev", "holdout", "healthy", "hard")
    assert "round10_warm2_last_step" in campaign["candidates"]
    assert "round10_stoch25_last_two" in campaign["candidates"]


def test_mechanism_program_campaign_reuses_next_round_and_narrows_to_targeted_subset() -> None:
    campaign = load_campaign_config("configs/experiments/lss_mechanism_program/campaign.yaml")
    track_counts = Counter(str(meta["track"]) for meta in campaign["candidates"].values())

    assert campaign["name"] == "lss_mechanism_program"
    assert track_counts == {"fruitful": 12, "exploratory": 12}
    assert len(campaign["candidates"]) == 24
    assert campaign["current_decision_memo"] == "outputs/reports/next_round_decision_memo.md"
    assert campaign["decision_strings"]["confirm"] == "active benchmark confirmed and frontier clarified"
    assert campaign["reports"]["stageA1_report"] == "outputs/reports/mechanism_stage1_round_differential.md"
    assert campaign["reports"]["stage1_fruitful_report"] == "outputs/reports/mechanism_stageB1_screening_exploit.md"
    assert campaign["reports"]["stage1_exploratory_report"] == "outputs/reports/mechanism_stageB1_screening_explore.md"
    assert campaign["reports"]["decision_memo"] == "outputs/reports/mechanism_next_decision_memo.md"
    assert campaign["analysis"]["phase_candidate"] == "round10_conf_post4"
    assert "outputs/experiments/lss_next_round_campaign/stage2_screening" in campaign["reuse_roots"]["stage1_sare"]


def test_deadlock_program_campaign_targets_deadlock_blocks_and_candidate_budget() -> None:
    campaign = load_campaign_config("configs/experiments/lss_deadlock_program/campaign.yaml")
    track_counts = Counter(str(meta["track"]) for meta in campaign["candidates"].values())

    assert campaign["name"] == "lss_deadlock_program"
    assert track_counts == {"fruitful": 18, "exploratory": 12}
    assert len(campaign["candidates"]) == 30
    assert campaign["current_decision_memo"] == "outputs/reports/mechanism_next_decision_memo.md"
    assert campaign["decision_strings"]["confirm"] == "active benchmark confirmed and deadlock frontier clarified"
    assert campaign["reports"]["family_definition"] == "outputs/reports/deadlock_family_definition.md"
    assert campaign["reports"]["casebook_report"] == "outputs/reports/deadlock_casebook.md"
    assert campaign["reports"]["shortlist_report"] == "outputs/reports/deadlock_mechanism_shortlist.md"
    assert campaign["reports"]["decision_memo"] == "outputs/reports/deadlock_next_decision_memo.md"
    assert campaign["blocks"]["dev"][0]["lane"] == "prospective_c"
    assert campaign["blocks"]["holdout"][0]["lane"] == "prospective_g"
    assert campaign["analysis"]["parity_candidate"] == "round7"
    assert "round10_search_x4" in campaign["candidates"]
    assert "round10_conf_search_x4" in campaign["candidates"]


def test_deadlock_contract_program_campaign_targets_teacher_and_distribution_repairs() -> None:
    campaign = load_campaign_config("configs/experiments/lss_deadlock_contract_program/campaign.yaml")
    track_counts = Counter(str(meta["track"]) for meta in campaign["candidates"].values())

    assert campaign["name"] == "lss_deadlock_contract_program"
    assert track_counts == {"fruitful": 17, "exploratory": 7}
    assert len(campaign["candidates"]) == 24
    assert campaign["current_decision_memo"] == "outputs/reports/deadlock_next_decision_memo.md"
    assert campaign["decision_strings"]["confirm"] == "active benchmark confirmed and deadlock/data-contract frontier clarified"
    assert campaign["reports"]["teacher_audit_report"] == "outputs/reports/deadlock_contract_teacher_audit.md"
    assert campaign["reports"]["distribution_audit_report"] == "outputs/reports/deadlock_contract_distribution_audit.md"
    assert campaign["reports"]["decision_memo"] == "outputs/reports/deadlock_contract_decision_memo.md"
    assert campaign["analysis"]["program_label"] == "Deadlock/Data-Contract Program"
    assert campaign["analysis"]["shortlist_max_directions"] == 4
    assert "round10_search_x4_temp105" in campaign["candidates"]
    assert "round10_prekey_phase_balanced_4096" in campaign["candidates"]


def test_deadlock_arch_program_campaign_adds_archpilot_track_and_budget() -> None:
    campaign = load_campaign_config("configs/experiments/lss_deadlock_arch_program/campaign.yaml")
    track_counts = Counter(str(meta["track"]) for meta in campaign["candidates"].values())

    assert campaign["name"] == "lss_deadlock_arch_program"
    assert track_counts == {"fruitful": 22, "exploratory": 10, "archpilot": 4}
    assert len(campaign["candidates"]) == 36
    assert campaign["current_decision_memo"] == "outputs/reports/deadlock_contract_decision_memo.md"
    assert campaign["selection"]["stage1_archpilot_top_k"] == 1
    assert campaign["reports"]["family_definition"] == "outputs/reports/deadlock_arch_family_definition.md"
    assert campaign["reports"]["stage1_archpilot_report"] == "outputs/reports/deadlock_arch_stageB1_screening_archpilot.md"
    assert campaign["reports"]["decision_memo"] == "outputs/reports/deadlock_arch_decision_memo.md"
    assert campaign["analysis"]["program_label"] == "Deadlock/Data-Contract + Architecture-Adjacent Program"
    assert campaign["analysis"]["shortlist_max_directions"] == 6
    assert campaign["analysis"]["transition_state_targets"] == ["carry_key", "at_locked_door", "post_unlock"]
    assert "round10_phase_memory_mix050_search_x4" in campaign["candidates"]
    assert "outputs/experiments/lss_deadlock_contract_program/stageB1_screening" in campaign["reuse_roots"]["stage1_sare"]


def test_deadlock_plus_program_campaign_scales_to_seventy_two_candidates() -> None:
    campaign = load_campaign_config("configs/experiments/lss_deadlock_plus_program/campaign.yaml")
    track_counts = Counter(str(meta["track"]) for meta in campaign["candidates"].values())

    assert campaign["name"] == "lss_deadlock_plus_program"
    assert track_counts == {"fruitful": 50, "exploratory": 14, "archpilot": 8}
    assert len(campaign["candidates"]) == 72
    assert campaign["current_decision_memo"] == "outputs/reports/deadlock_arch_decision_memo.md"
    assert campaign["selection"]["stage1_fruitful_top_k"] == 5
    assert campaign["selection"]["stage1_archpilot_top_k"] == 2
    assert len(campaign["blocks"]["dev"]) == 3
    assert len(campaign["blocks"]["holdout"]) == 3
    assert len(campaign["blocks"]["healthy"]) == 3
    assert campaign["reports"]["decision_memo"] == "outputs/reports/deadlock_plus_decision_memo.md"
    assert campaign["analysis"]["shortlist_max_directions"] == 8
    assert "round10_phase_memory_mix075_search_x4" in campaign["candidates"]
    assert "outputs/experiments/lss_deadlock_arch_program/stageB1_screening" in campaign["reuse_roots"]["stage1_sare"]


def test_deadlock_oracle_program_campaign_separates_oracle_and_practical_lanes() -> None:
    campaign = load_campaign_config("configs/experiments/lss_deadlock_oracle_program/campaign.yaml")
    track_counts = Counter(str(meta["track"]) for meta in campaign["candidates"].values())

    assert campaign["name"] == "lss_deadlock_oracle_program"
    assert track_counts == {"fruitful": 16, "exploratory": 6, "archpilot": 2}
    assert len(campaign["candidates"]) == 24
    assert campaign["current_decision_memo"] == "outputs/reports/deadlock_arch_decision_memo.md"
    assert campaign["decision_strings"]["arch_future_branch"] == "oracle evidence suggests a future architecture branch but not a benchmark change"
    assert campaign["reports"]["oracle_teacher_target_report"] == "outputs/reports/oracle_stageA2_teacher_target.md"
    assert campaign["reports"]["oracle_synthesis_report"] == "outputs/reports/oracle_stageA6_synthesis.md"
    assert campaign["reports"]["decision_memo"] == "outputs/reports/deadlock_oracle_decision_memo.md"
    assert sorted(campaign["oracle_candidates"].keys()) == [
        "oracle_combined_confclip_transition4096",
        "oracle_combined_temp105_transition4096",
        "oracle_teacher_base_search4",
        "oracle_teacher_confclip_search4",
        "oracle_teacher_temp105_search4",
        "oracle_teacher_temp115_search4",
        "oracle_transition_base_search4",
        "oracle_transition_phase_balanced_2048",
        "oracle_transition_phase_balanced_4096",
        "oracle_transition_phase_balanced_4096_temp105",
    ]
    assert campaign["analysis"]["shortlist_max_directions"] == 5
    assert campaign["analysis"]["oracle_case_order"] == ["teacher_locked", "ambiguous", "guardrail"]
    assert "outputs/experiments/lss_deadlock_arch_program/stageB1_screening" in campaign["reuse_roots"]["stage1_sare"]


def test_deadlock_escape_program_campaign_registers_rescue_and_practicalization_lanes() -> None:
    campaign = load_campaign_config("configs/experiments/lss_deadlock_escape_program/campaign.yaml")
    track_counts = Counter(str(meta["track"]) for meta in campaign["candidates"].values())

    assert campaign["name"] == "lss_deadlock_escape_program"
    assert track_counts == {"fruitful": 15, "exploratory": 4}
    assert len(campaign["candidates"]) == 19
    assert campaign["current_decision_memo"] == "outputs/reports/deadlock_oracle_decision_memo.md"
    assert campaign["decision_strings"]["confirm"] == "active benchmark confirmed and bounded rescue frontier clarified"
    assert campaign["reports"]["subgroup_detector_report"] == "outputs/reports/escape_program_subgroup_detector.md"
    assert campaign["reports"]["rescue_stage1_report"] == "outputs/reports/escape_stageB1_rescue_screening.md"
    assert campaign["reports"]["practicalization_stage2_report"] == "outputs/reports/escape_stageC2_verification_fairness_holdout.md"
    assert campaign["reports"]["decision_memo"] == "outputs/reports/escape_next_decision_memo.md"
    assert len(campaign["rescue_candidates"]) == 24
    assert len(campaign["analysis"]["teacher_locked_dev_groups"]) == 2
    assert len(campaign["analysis"]["teacher_locked_holdout_groups"]) == 2
    assert len(campaign["analysis"]["ambiguous_groups"]) == 2
    assert len(campaign["analysis"]["healthy_groups"]) == 2
