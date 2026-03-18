from __future__ import annotations

from psmn_rl.analysis.lss_successor_challenge import (
    _promote_challenger,
    _route_pass,
    _select_best_challenger,
)


def test_select_best_challenger_prefers_round7_on_tie() -> None:
    summaries = {
        "round7": {
            "hard_combined_mean": 1.0,
            "hard_holdout_mean": 1.0,
            "healthy_mean": 1.0,
            "hard_combined_failures": 0.0,
            "healthy_failures": 0.0,
        },
        "round8": {
            "hard_combined_mean": 1.0,
            "hard_holdout_mean": 1.0,
            "healthy_mean": 1.0,
            "hard_combined_failures": 0.0,
            "healthy_failures": 0.0,
        },
    }
    assert _select_best_challenger(summaries) == "round7"


def test_promote_challenger_requires_margin_or_better_control_gap() -> None:
    campaign = {"selection": {"equality_eps": 0.0001}}
    incumbent = {
        "hard_combined_sare": 1.0,
        "hard_combined_token": 1.0,
        "hard_combined_single": 0.90,
        "hard_sare_minus_token": 0.0,
        "hard_sare_minus_single": 0.10,
        "healthy_mean": 1.0,
    }
    challenger = {
        "hard_combined_sare": 1.0,
        "hard_combined_token": 1.0,
        "hard_combined_single": 1.0,
        "hard_sare_minus_token": 0.0,
        "hard_sare_minus_single": 0.0,
        "healthy_mean": 1.0,
    }
    assert not _promote_challenger(campaign, incumbent, challenger)


def test_promote_challenger_allows_tied_sare_with_better_token_gap() -> None:
    campaign = {"selection": {"equality_eps": 0.0001}}
    incumbent = {
        "hard_combined_sare": 1.0,
        "hard_combined_token": 1.0,
        "hard_combined_single": 1.0,
        "hard_sare_minus_token": 0.0,
        "hard_sare_minus_single": 0.0,
        "healthy_mean": 1.0,
    }
    challenger = {
        "hard_combined_sare": 1.0,
        "hard_combined_token": 0.95,
        "hard_combined_single": 0.95,
        "hard_sare_minus_token": 0.05,
        "hard_sare_minus_single": 0.05,
        "healthy_mean": 1.0,
    }
    assert _promote_challenger(campaign, incumbent, challenger)


def test_route_pass_checks_all_required_drops() -> None:
    campaign = {"selection": {"route_drop_min": 0.5, "route_randomization_drop_min": 0.25}}
    summary = {
        "fixed_router_drop": 1.0,
        "worst_ablation_drop": 1.0,
        "route_randomization_drop": 0.5,
    }
    assert _route_pass(campaign, summary)
