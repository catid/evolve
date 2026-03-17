from __future__ import annotations

from psmn_rl.analysis.lss_multi_expert_hardening import (
    _build_final_combined_doorkey_report,
    _build_fresh_single_expert_report,
    _build_seed29_forensics_report,
)


def _row(lane: str, seed: int, label: str, success: float) -> dict[str, object]:
    return {
        "lane": lane,
        "seed": seed,
        "label": label,
        "mode": "greedy",
        "eval_success_rate": success,
    }


def test_fresh_single_expert_report_mentions_sare_ahead() -> None:
    rows = [
        _row("original", 7, "recovered_token_dense", 0.6),
        _row("original", 7, "kl_lss_token_dense", 0.7),
        _row("original", 7, "kl_lss_single_expert", 0.6),
        _row("original", 7, "baseline_sare", 0.0),
        _row("original", 7, "kl_lss_sare", 1.0),
        _row("fresh", 23, "recovered_token_dense", 0.0),
        _row("fresh", 23, "kl_lss_token_dense", 0.4),
        _row("fresh", 23, "kl_lss_single_expert", 0.5),
        _row("fresh", 23, "baseline_sare", 0.0),
        _row("fresh", 23, "kl_lss_sare", 1.0),
    ]
    report = _build_fresh_single_expert_report(rows, episodes=64)
    assert "still stays ahead of both" in report


def test_seed29_forensics_report_mentions_narrow_probe_exception() -> None:
    reference_rows = [
        {
            "lane": "original",
            "seed": 7,
            "probe": "baseline",
            "detail": "-",
            "eval_success_rate": 1.0,
            "route_entropy": 1.38,
            "path_entropy": 1.1,
            "expert_load_0": 0.25,
            "expert_load_1": 0.25,
            "expert_load_2": 0.25,
            "expert_load_3": 0.25,
        },
        {
            "lane": "fresh",
            "seed": 29,
            "probe": "baseline",
            "detail": "-",
            "eval_success_rate": 1.0,
            "route_entropy": 1.37,
            "path_entropy": 1.1,
            "expert_load_0": 0.25,
            "expert_load_1": 0.25,
            "expert_load_2": 0.25,
            "expert_load_3": 0.25,
        },
    ]
    forensic_rows = [
        {"lane": "fresh", "seed": 29, "probe": "baseline", "detail": "-", "eval_success_rate": 1.0, "route_entropy": 1.37, "active_compute_proxy": 0.5, "expert_count": 4.0, "top_k": 2.0},
        {"lane": "fresh", "seed": 29, "probe": "expert_ablation", "detail": "0", "eval_success_rate": 0.0, "route_entropy": 1.37, "active_compute_proxy": 0.5},
        {"lane": "fresh", "seed": 29, "probe": "router_override", "detail": "pair_0_1", "eval_success_rate": 0.8, "route_entropy": 0.69, "active_compute_proxy": 0.5},
        {"lane": "fresh", "seed": 29, "probe": "router_override", "detail": "pair_0_2", "eval_success_rate": 0.0, "route_entropy": 0.69, "active_compute_proxy": 0.5},
        {"lane": "fresh", "seed": 29, "probe": "router_override", "detail": "most_used_pair", "eval_success_rate": 0.0, "route_entropy": 0.69, "active_compute_proxy": 0.5},
        {"lane": "fresh", "seed": 29, "probe": "route_randomization", "detail": "uniform_topk_random:trial=0", "eval_success_rate": 0.95, "route_entropy": 0.69, "active_compute_proxy": 0.5},
        {"lane": "fresh", "seed": 29, "probe": "route_randomization", "detail": "uniform_topk_random:trial=1", "eval_success_rate": 0.85, "route_entropy": 0.69, "active_compute_proxy": 0.5},
        {"lane": "fresh", "seed": 29, "probe": "route_randomization", "detail": "random_single_expert:trial=0", "eval_success_rate": 0.1, "route_entropy": 0.1, "active_compute_proxy": 0.5},
        {"lane": "fresh", "seed": 29, "probe": "route_randomization", "detail": "random_single_expert:trial=1", "eval_success_rate": 0.0, "route_entropy": 0.1, "active_compute_proxy": 0.5},
    ]
    report = _build_seed29_forensics_report(reference_rows, forensic_rows, episodes=64, trial_count=2)
    assert "narrow probe exception" in report


def test_final_combined_report_mentions_multi_expert_edge() -> None:
    rows = [
        _row("original", 7, "recovered_token_dense", 0.6),
        _row("original", 7, "kl_lss_token_dense", 0.7),
        _row("original", 7, "kl_lss_single_expert", 0.6),
        _row("original", 7, "baseline_sare", 0.0),
        _row("original", 7, "kl_lss_sare", 1.0),
        _row("fresh", 23, "recovered_token_dense", 0.0),
        _row("fresh", 23, "kl_lss_token_dense", 0.5),
        _row("fresh", 23, "kl_lss_single_expert", 0.4),
        _row("fresh", 23, "baseline_sare", 0.0),
        _row("fresh", 23, "kl_lss_sare", 1.0),
        _row("fresh_final", 47, "recovered_token_dense", 0.0),
        _row("fresh_final", 47, "kl_lss_token_dense", 0.4),
        _row("fresh_final", 47, "baseline_sare", 0.0),
        _row("fresh_final", 47, "kl_lss_sare", 0.9),
    ]
    report = _build_final_combined_doorkey_report(rows, episodes=64)
    assert "specifically multi-expert routed edge" in report
