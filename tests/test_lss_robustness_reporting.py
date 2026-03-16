from __future__ import annotations

from psmn_rl.analysis.lss_robustness import _build_heterogeneity_report, _build_multiseed_report


def test_build_heterogeneity_report_mentions_seed_split() -> None:
    rows = [
        {
            "seed": 7,
            "round_index": 1,
            "mode": "greedy",
            "eval_success_rate": 0.5,
            "collection/steps": 100.0,
            "collection/unique_state_ratio": 0.7,
            "collection/teacher_confidence_mean": 0.9,
            "collection/disagreement_rate": 0.2,
            "collection/route_entropy": 1.3,
            "route_entropy": 1.2,
        },
        {
            "seed": 7,
            "round_index": 1,
            "mode": "sampled_t1.0",
            "eval_success_rate": 0.6,
        },
        {
            "seed": 11,
            "round_index": 1,
            "mode": "greedy",
            "eval_success_rate": 1.0,
            "collection/steps": 120.0,
            "collection/unique_state_ratio": 0.9,
            "collection/teacher_confidence_mean": 0.95,
            "collection/disagreement_rate": 0.1,
            "collection/route_entropy": 1.35,
            "route_entropy": 1.25,
        },
        {
            "seed": 11,
            "round_index": 1,
            "mode": "sampled_t1.0",
            "eval_success_rate": 1.0,
        },
        {
            "seed": 19,
            "round_index": 1,
            "mode": "greedy",
            "eval_success_rate": 0.0,
            "collection/steps": 90.0,
            "collection/unique_state_ratio": 0.4,
            "collection/teacher_confidence_mean": 0.7,
            "collection/disagreement_rate": 0.5,
            "collection/route_entropy": 1.1,
            "route_entropy": 1.0,
        },
        {
            "seed": 19,
            "round_index": 1,
            "mode": "sampled_t1.0",
            "eval_success_rate": 0.0,
        },
    ]
    report = _build_heterogeneity_report(rows, episodes=64)
    assert "Seed `11` is the successful case" in report
    assert "seed `19` is the failed case" in report.lower()
    assert "lower learner-state coverage" in report


def test_build_multiseed_report_marks_gate_failure_with_zero_seed() -> None:
    rows = [
        {"seed": 7, "label": "flat_dense", "mode": "greedy", "eval_success_rate": 1.0},
        {"seed": 7, "label": "recovered_token_dense", "mode": "greedy", "eval_success_rate": 0.7},
        {"seed": 7, "label": "baseline_sare", "mode": "greedy", "eval_success_rate": 0.0},
        {"seed": 7, "label": "improved_lss_sare", "mode": "greedy", "eval_success_rate": 0.5},
        {"seed": 11, "label": "flat_dense", "mode": "greedy", "eval_success_rate": 1.0},
        {"seed": 11, "label": "recovered_token_dense", "mode": "greedy", "eval_success_rate": 0.4},
        {"seed": 11, "label": "baseline_sare", "mode": "greedy", "eval_success_rate": 0.0},
        {"seed": 11, "label": "improved_lss_sare", "mode": "greedy", "eval_success_rate": 0.8},
        {"seed": 19, "label": "flat_dense", "mode": "greedy", "eval_success_rate": 1.0},
        {"seed": 19, "label": "recovered_token_dense", "mode": "greedy", "eval_success_rate": 0.6},
        {"seed": 19, "label": "baseline_sare", "mode": "greedy", "eval_success_rate": 0.0},
        {"seed": 19, "label": "improved_lss_sare", "mode": "greedy", "eval_success_rate": 0.0},
    ]
    report = _build_multiseed_report(rows, episodes=64)
    assert "fails the repo’s reopen-routed-claim bar" in report
    assert "At least one seed remains a complete greedy failure" in report
