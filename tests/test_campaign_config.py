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
