from __future__ import annotations

from psmn_rl.analysis.lss_long_campaign import _dedupe_metric_rows, _stage2_pass, _stage3_pass


def test_stage2_pass_requires_mean_gain_and_non_noisy_seed_pattern() -> None:
    baseline = {47: 0.0, 53: 0.515625, 59: 0.421875}
    good_rows = [
        {"seed": 47, "final_greedy_success": 0.5},
        {"seed": 53, "final_greedy_success": 0.515625},
        {"seed": 59, "final_greedy_success": 0.421875},
    ]
    bad_rows = [
        {"seed": 47, "final_greedy_success": 1.0},
        {"seed": 53, "final_greedy_success": 0.0},
        {"seed": 59, "final_greedy_success": 0.0},
    ]
    assert _stage2_pass(good_rows, baseline) is True
    assert _stage2_pass(bad_rows, baseline) is False


def test_stage3_pass_requires_sare_to_match_single_expert_and_own_the_gain() -> None:
    sare_stats = {"mean": 0.6}
    single_stats = {"mean": 0.5}
    assert _stage3_pass(sare_stats, single_stats, token_delta=0.05, single_delta=0.10, sare_delta=0.30) is True
    assert _stage3_pass(sare_stats, single_stats, token_delta=0.35, single_delta=0.10, sare_delta=0.30) is False


def test_dedupe_metric_rows_keeps_one_entry_per_lane_seed_label() -> None:
    rows = [
        {"lane": "fresh_final", "seed": 47, "label": "kl_lss_sare", "final_greedy_success": 0.453125},
        {"lane": "fresh_final", "seed": 47, "label": "kl_lss_sare", "final_greedy_success": 0.453125},
        {"lane": "fresh_final", "seed": 47, "label": "kl_lss_single_expert", "final_greedy_success": 0.453125},
    ]
    deduped = _dedupe_metric_rows(rows)
    assert len(deduped) == 2
    assert {(row["lane"], row["seed"], row["label"]) for row in deduped} == {
        ("fresh_final", 47, "kl_lss_sare"),
        ("fresh_final", 47, "kl_lss_single_expert"),
    }
