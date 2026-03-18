from __future__ import annotations

from psmn_rl.analysis.lss_successor_validation import _route_pass, _select_validation_candidate


def test_select_validation_candidate_prefers_round7_only_for_clear_win() -> None:
    campaign = {"selection": {"round7_min_gain": 0.05}}
    summaries = {
        "round6": {"sare_mean": 0.90, "token_mean": 0.85, "single_mean": 0.80},
        "round7": {"sare_mean": 0.96, "token_mean": 0.95, "single_mean": 0.94},
    }
    assert _select_validation_candidate(campaign, summaries) == "round7"


def test_select_validation_candidate_keeps_round6_for_small_or_unfair_round7_gain() -> None:
    campaign = {"selection": {"round7_min_gain": 0.05}}
    summaries = {
        "round6": {"sare_mean": 0.92, "token_mean": 0.90, "single_mean": 0.88},
        "round7": {"sare_mean": 0.95, "token_mean": 0.97, "single_mean": 0.90},
    }
    assert _select_validation_candidate(campaign, summaries) == "round6"


def test_route_pass_requires_fixed_ablation_and_randomization_drops() -> None:
    campaign = {"selection": {"route_drop_min": 0.5, "route_randomization_drop_min": 0.25}}
    summary = {"fixed_router_drop": 1.0, "worst_ablation_drop": 0.75, "route_randomization_drop": 0.50}
    assert _route_pass(campaign, summary) is True
    summary["route_randomization_drop"] = 0.10
    assert _route_pass(campaign, summary) is False
