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
