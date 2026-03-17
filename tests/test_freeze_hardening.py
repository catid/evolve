from psmn_rl.analysis.freeze_hardening import _candidate_validation_rows


def test_validation_rows_fail_when_candidate_drifts_from_manifest() -> None:
    manifest = {
        "thresholds": {
            "combined_means": {"kl_lss_sare": 0.7122},
            "combined_complete_seed_failures": {"kl_lss_sare": 1},
            "retry_block_means": {"kl_lss_sare": 0.3125},
            "retry_block_complete_seed_failures": {"kl_lss_sare": 1},
            "keycorridor_transfer_means": {"kl_lss_sare": 0.0},
        },
        "seed_groups": {
            "combined": {"lanes": ["fresh_final"], "seeds": [47]},
            "retry_block": {"lane": "fresh_final", "seeds": [47]},
        },
    }
    candidate = {
        "metrics": {
            "combined": {"kl_lss_sare": {"mean": 0.70, "complete_seed_failures": 2}},
            "retry_block": {"kl_lss_sare": {"mean": 0.2, "complete_seed_failures": 2}},
            "keycorridor_transfer": {"kl_lss_sare": {"mean": 0.1, "complete_seed_failures": 1}},
        },
        "actual_sets": {
            "combined_lane_seeds": [["fresh_final", 47]],
            "retry_block_lane_seeds": [["fresh_final", 47]],
        },
    }
    rows, verdict = _candidate_validation_rows(manifest, candidate)
    assert verdict == "FAIL: baseline artifacts drifted"
    assert any(row.metric == "mean" and row.section == "combined" and row.status == "FAIL" for row in rows)
    assert any(row.metric == "retry_block_lane_seed_set" and row.status == "PASS" for row in rows)
