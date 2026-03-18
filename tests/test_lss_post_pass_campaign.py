from __future__ import annotations

import csv

from psmn_rl.analysis.lss_post_pass_campaign import _candidate_pack_rows, _new_block_summary, _stability_class


def test_new_block_summary_tracks_competitiveness_vs_single_expert() -> None:
    rows = [
        {"lane": "post_pass_a", "label": "kl_lss_sare", "final_greedy_success": 0.50},
        {"lane": "post_pass_a", "label": "kl_lss_sare", "final_greedy_success": 0.45},
        {"lane": "post_pass_a", "label": "kl_lss_sare", "final_greedy_success": 0.40},
        {"lane": "post_pass_a", "label": "kl_lss_single_expert", "final_greedy_success": 0.47},
        {"lane": "post_pass_a", "label": "kl_lss_single_expert", "final_greedy_success": 0.45},
        {"lane": "post_pass_a", "label": "kl_lss_single_expert", "final_greedy_success": 0.43},
        {"lane": "post_pass_a", "label": "kl_lss_token_dense", "final_greedy_success": 0.60},
        {"lane": "post_pass_a", "label": "kl_lss_token_dense", "final_greedy_success": 0.55},
        {"lane": "post_pass_a", "label": "kl_lss_token_dense", "final_greedy_success": 0.50},
        {"lane": "post_pass_b", "label": "kl_lss_sare", "final_greedy_success": 0.10},
        {"lane": "post_pass_b", "label": "kl_lss_sare", "final_greedy_success": 0.00},
        {"lane": "post_pass_b", "label": "kl_lss_sare", "final_greedy_success": 0.20},
        {"lane": "post_pass_b", "label": "kl_lss_single_expert", "final_greedy_success": 0.35},
        {"lane": "post_pass_b", "label": "kl_lss_single_expert", "final_greedy_success": 0.30},
        {"lane": "post_pass_b", "label": "kl_lss_single_expert", "final_greedy_success": 0.25},
        {"lane": "post_pass_b", "label": "kl_lss_token_dense", "final_greedy_success": 0.40},
        {"lane": "post_pass_b", "label": "kl_lss_token_dense", "final_greedy_success": 0.35},
        {"lane": "post_pass_b", "label": "kl_lss_token_dense", "final_greedy_success": 0.30},
    ]

    summaries = {row["lane"]: row for row in _new_block_summary(rows)}
    assert summaries["post_pass_a"]["competitive_vs_single_expert"] is True
    assert summaries["post_pass_b"]["competitive_vs_single_expert"] is False


def test_stability_class_distinguishes_plateau_from_spike() -> None:
    assert _stability_class([0.52, 0.56, 0.55, 0.53]) == "stable_plateau"
    assert _stability_class([0.05, 0.10, 0.62, 0.18]) == "narrow_spike"
    assert _stability_class([0.25, 0.45, 0.40, 0.28]) == "noisy_brittle"


def test_candidate_pack_rows_keep_gate_combined_set_frozen_comparable(tmp_path) -> None:
    frozen_combined = tmp_path / "frozen_combined.csv"
    historical = tmp_path / "historical.csv"
    stage1 = tmp_path / "stage1.csv"

    with frozen_combined.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "lane", "seed", "label", "eval_success_rate", "run_dir"])
        writer.writeheader()
        writer.writerows(
            [
                {"mode": "greedy", "lane": "original", "seed": 7, "label": "recovered_token_dense", "eval_success_rate": 1.0, "run_dir": "orig/recovered"},
                {"mode": "greedy", "lane": "fresh_final", "seed": 47, "label": "recovered_token_dense", "eval_success_rate": 1.0, "run_dir": "ff/recovered"},
                {"mode": "greedy", "lane": "original", "seed": 7, "label": "baseline_sare", "eval_success_rate": 0.0, "run_dir": "orig/baseline"},
                {"mode": "greedy", "lane": "fresh_final", "seed": 47, "label": "baseline_sare", "eval_success_rate": 0.0, "run_dir": "ff/baseline"},
            ]
        )

    with historical.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate", "lane", "seed", "label", "final_greedy_success", "run_dir"])
        writer.writeheader()
        writer.writerows(
            [
                {"candidate": "post_unlock_weighted", "lane": "original", "seed": 7, "label": "kl_lss_token_dense", "final_greedy_success": 1.0, "run_dir": "orig/token"},
                {"candidate": "post_unlock_weighted", "lane": "original", "seed": 7, "label": "kl_lss_single_expert", "final_greedy_success": 1.0, "run_dir": "orig/single"},
                {"candidate": "post_unlock_weighted", "lane": "original", "seed": 7, "label": "kl_lss_sare", "final_greedy_success": 1.0, "run_dir": "orig/sare"},
                {"candidate": "post_unlock_weighted", "lane": "fresh_final", "seed": 47, "label": "kl_lss_token_dense", "final_greedy_success": 1.0, "run_dir": "ff/token"},
                {"candidate": "post_unlock_weighted", "lane": "fresh_final", "seed": 47, "label": "kl_lss_single_expert", "final_greedy_success": 0.5, "run_dir": "ff/single"},
                {"candidate": "post_unlock_weighted", "lane": "fresh_final", "seed": 47, "label": "kl_lss_sare", "final_greedy_success": 0.5, "run_dir": "ff/sare"},
            ]
        )

    with stage1.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["lane", "block", "seed", "label", "final_greedy_success", "run_dir"])
        writer.writeheader()
        writer.writerows(
            [
                {"lane": "post_pass_a", "block": "post_pass_a", "seed": 61, "label": "recovered_token_dense", "final_greedy_success": 1.0, "run_dir": "ppa/recovered"},
                {"lane": "post_pass_a", "block": "post_pass_a", "seed": 61, "label": "baseline_sare", "final_greedy_success": 0.0, "run_dir": "ppa/baseline"},
                {"lane": "post_pass_a", "block": "post_pass_a", "seed": 61, "label": "kl_lss_token_dense", "final_greedy_success": 1.0, "run_dir": "ppa/token"},
                {"lane": "post_pass_a", "block": "post_pass_a", "seed": 61, "label": "kl_lss_single_expert", "final_greedy_success": 0.0, "run_dir": "ppa/single"},
                {"lane": "post_pass_a", "block": "post_pass_a", "seed": 61, "label": "kl_lss_sare", "final_greedy_success": 1.0, "run_dir": "ppa/sare"},
            ]
        )

    combined_rows, retry_rows, expanded_rows = _candidate_pack_rows(frozen_combined, historical, stage1)

    combined_sets = {(row["lane"], row["seed"]) for row in combined_rows if row["label"] == "kl_lss_sare"}
    retry_sets = {(row["lane"], row["seed"]) for row in retry_rows if row["label"] == "kl_lss_sare"}
    expanded_sets = {(row["lane"], row["seed"]) for row in expanded_rows if row["label"] == "kl_lss_sare"}

    assert combined_sets == {("original", 7), ("fresh_final", 47)}
    assert retry_sets == {("fresh_final", 47)}
    assert expanded_sets == {("original", 7), ("fresh_final", 47), ("post_pass_a", 61)}
