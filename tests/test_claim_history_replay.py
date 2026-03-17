from __future__ import annotations

import csv

from psmn_rl.analysis.claim_history_replay import _build_metric_block, _classify_ledger_row


def test_build_metric_block_aggregates_grouped_csv_sources(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "label", "lane", "seed", "eval_success_rate"])
        writer.writeheader()
        writer.writerows(
            [
                {"mode": "greedy", "label": "kl_lss_sare", "lane": "original", "seed": "7", "eval_success_rate": "1.0"},
                {"mode": "greedy", "label": "kl_lss_sare", "lane": "fresh", "seed": "23", "eval_success_rate": "0.5"},
                {"mode": "sampled_t1.0", "label": "kl_lss_sare", "lane": "fresh", "seed": "23", "eval_success_rate": "1.0"},
            ]
        )

    metrics, actual_set = _build_metric_block(
        {
            "sources": [
                {
                    "kind": "grouped_csv",
                    "path": str(csv_path),
                    "filters": {"mode": "greedy"},
                    "group_field": "label",
                    "value_field": "eval_success_rate",
                    "lane_field": "lane",
                    "seed_field": "seed",
                }
            ]
        }
    )

    assert metrics["kl_lss_sare"]["mean"] == 0.75
    assert metrics["kl_lss_sare"]["min"] == 0.5
    assert metrics["kl_lss_sare"]["max"] == 1.0
    assert metrics["kl_lss_sare"]["complete_seed_failures"] == 0
    assert metrics["kl_lss_sare"]["seed_count"] == 2
    assert actual_set == {("original", 7), ("fresh", 23)}


def test_classify_ledger_row_handles_frozen_and_narrower_cases() -> None:
    classification, detail = _classify_ledger_row("frozen", {"FAIL: claim remains frozen"}, True)
    assert classification == "consistent"
    assert "frozen verdict" in detail

    classification, detail = _classify_ledger_row("bounded positive", {"INCONCLUSIVE: missing prerequisites"}, True)
    assert classification == "consistent but narrower under current gate"
    assert "does not clear the modern frozen gate" in detail
