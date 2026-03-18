from psmn_rl.analysis.lss_successor_stress import (
    _pick_best_challenger,
    _promote_challenger,
    _route_pass,
)


def test_pick_best_challenger_prefers_earlier_candidate_on_tie() -> None:
    campaign = {"selection": {"challenger_order": ["round7", "round8", "round9"]}}
    rows = [
        {"candidate": "round7", "label": "kl_lss_sare", "final_greedy_success": 1.0},
        {"candidate": "round7", "label": "kl_lss_sare", "final_greedy_success": 1.0},
        {"candidate": "round8", "label": "kl_lss_sare", "final_greedy_success": 1.0},
        {"candidate": "round8", "label": "kl_lss_sare", "final_greedy_success": 1.0},
    ]
    assert _pick_best_challenger(campaign, rows) == "round7"


def test_promote_challenger_requires_real_gain() -> None:
    campaign = {"selection": {"min_mean_gain": 0.05, "min_margin_gain": 0.05}}
    round6 = {"sare_mean": 1.0, "token_mean": 1.0, "single_mean": 0.9, "sare_failures": 0}
    challenger = {"sare_mean": 1.0, "token_mean": 1.0, "single_mean": 1.0, "sare_failures": 0}
    assert not _promote_challenger(campaign, round6, challenger)


def test_promote_challenger_allows_mean_improvement() -> None:
    campaign = {"selection": {"min_mean_gain": 0.05, "min_margin_gain": 0.05}}
    round6 = {"sare_mean": 0.9, "token_mean": 1.0, "single_mean": 0.8, "sare_failures": 0}
    challenger = {"sare_mean": 0.97, "token_mean": 0.95, "single_mean": 0.9, "sare_failures": 0}
    assert _promote_challenger(campaign, round6, challenger)


def test_route_pass_respects_thresholds() -> None:
    campaign = {"selection": {"route_drop_min": 0.5, "route_randomization_drop_min": 0.25}}
    assert _route_pass(
        campaign,
        {
            "fixed_router_drop": 1.0,
            "worst_ablation_drop": 0.75,
            "route_randomization_drop": 0.5,
        },
    )
    assert not _route_pass(
        campaign,
        {
            "fixed_router_drop": 1.0,
            "worst_ablation_drop": 0.1,
            "route_randomization_drop": 0.5,
        },
    )
