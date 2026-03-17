from __future__ import annotations

from psmn_rl.analysis.lss_route_dependence import _build_report


def test_route_dependence_report_mentions_material_drop() -> None:
    rows = [
        {"lane": "original", "seed": 7, "probe": "baseline", "detail": "-", "eval_success_rate": 1.0, "eval_return": 0.9, "route_entropy": 1.38, "active_compute_proxy": 0.5},
        {"lane": "original", "seed": 7, "probe": "expert_ablation", "detail": "0", "eval_success_rate": 0.4, "eval_return": 0.2, "route_entropy": 1.38, "active_compute_proxy": 0.5},
        {"lane": "original", "seed": 7, "probe": "expert_ablation", "detail": "1", "eval_success_rate": 0.8, "eval_return": 0.6, "route_entropy": 1.38, "active_compute_proxy": 0.5},
        {"lane": "original", "seed": 7, "probe": "expert_ablation", "detail": "2", "eval_success_rate": 0.9, "eval_return": 0.7, "route_entropy": 1.38, "active_compute_proxy": 0.5},
        {"lane": "original", "seed": 7, "probe": "expert_ablation", "detail": "3", "eval_success_rate": 0.7, "eval_return": 0.5, "route_entropy": 1.38, "active_compute_proxy": 0.5},
        {"lane": "original", "seed": 7, "probe": "router_override", "detail": "most_used_pair", "eval_success_rate": 0.1, "eval_return": 0.0, "route_entropy": 0.69, "active_compute_proxy": 0.5},
        {"lane": "original", "seed": 7, "probe": "route_randomization", "detail": "uniform_topk_random", "eval_success_rate": 0.0, "eval_return": 0.0, "route_entropy": 0.69, "active_compute_proxy": 0.5},
    ]
    report = _build_report(rows, {("original", 7): [0, 1]}, episodes=64)
    assert "causally relevant" in report
