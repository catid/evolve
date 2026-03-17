from __future__ import annotations

import json

from psmn_rl.analysis.benchmark_pack import build_frozen_benchmark_pack, sha256_path
from psmn_rl.analysis.claim_gate import evaluate_claim_gate, evaluate_pack_claim_gate, evaluate_pack_claim_gate_from_paths


def _manifest() -> dict:
    return {
        "thaw_gate": {
            "required_controls": [
                "recovered_token_dense",
                "kl_lss_token_dense",
                "kl_lss_single_expert",
                "baseline_sare",
            ],
            "required_evaluation": {
                "path_key": "external_policy_diagnostics",
                "episodes": 64,
                "task": "DoorKey",
            },
            "retry_block": {
                "candidate_sare_mean_must_exceed": 0.3125,
                "candidate_sare_complete_seed_failures_must_be_lte": 1,
            },
            "combined_picture": {
                "candidate_sare_mean_must_be_gte": 0.7122,
                "candidate_sare_complete_seed_failures_must_be_lte": 1,
            },
            "disallowed_claim_keys": [
                "ppo_only_routed_win",
                "specifically_multi_expert_routed_advantage",
            ],
        }
    }


def test_claim_gate_fails_for_current_frozen_baseline_like_candidate() -> None:
    candidate = {
        "task": "DoorKey",
        "evaluation": {"path_key": "external_policy_diagnostics", "episodes": 64},
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": [
            "recovered_token_dense",
            "kl_lss_token_dense",
            "kl_lss_single_expert",
            "baseline_sare",
            "kl_lss_sare",
        ],
        "metrics": {
            "retry_block": {
                "kl_lss_sare": {"mean": 0.3125, "complete_seed_failures": 1},
                "kl_lss_single_expert": {"mean": 0.4635, "complete_seed_failures": 0},
            },
            "combined": {
                "kl_lss_sare": {"mean": 0.7122, "complete_seed_failures": 1},
                "kl_lss_single_expert": {"mean": 0.6862, "complete_seed_failures": 1},
            },
        },
    }
    verdict, checks = evaluate_claim_gate(_manifest(), candidate)
    assert verdict == "FAIL: claim remains frozen"
    assert any(check.name == "retry_block_improvement" and check.status == "FAIL" for check in checks)
    assert any(check.name == "retry_block_vs_single_expert" and check.status == "FAIL" for check in checks)


def test_claim_gate_is_inconclusive_when_required_controls_are_missing() -> None:
    candidate = {
        "task": "DoorKey",
        "evaluation": {"path_key": "external_policy_diagnostics", "episodes": 64},
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": ["recovered_token_dense", "kl_lss_sare"],
        "metrics": {},
    }
    verdict, checks = evaluate_claim_gate(_manifest(), candidate)
    assert verdict == "INCONCLUSIVE: missing prerequisites"
    assert any(check.name == "fairness_controls" and check.status == "INCONCLUSIVE" for check in checks)


def test_claim_gate_fails_overclaim_even_when_controls_are_missing() -> None:
    candidate = {
        "task": "DoorKey",
        "evaluation": {"path_key": "external_policy_diagnostics", "episodes": 64},
        "requested_claims": ["bounded_teacher_guided_doorkey_sare", "specifically_multi_expert_routed_advantage"],
        "controls_present": ["recovered_token_dense", "kl_lss_sare"],
        "metrics": {},
    }
    verdict, checks = evaluate_claim_gate(_manifest(), candidate)
    assert verdict == "FAIL: claim remains frozen"
    assert any(check.name == "claim_scope" and check.status == "FAIL" for check in checks)


def test_claim_gate_can_pass_when_candidate_clears_all_bars() -> None:
    candidate = {
        "task": "DoorKey",
        "evaluation": {"path_key": "external_policy_diagnostics", "episodes": 64},
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": [
            "recovered_token_dense",
            "kl_lss_token_dense",
            "kl_lss_single_expert",
            "baseline_sare",
            "kl_lss_sare",
        ],
        "metrics": {
            "retry_block": {
                "kl_lss_sare": {"mean": 0.6, "complete_seed_failures": 0},
                "kl_lss_single_expert": {"mean": 0.5, "complete_seed_failures": 0},
            },
            "combined": {
                "kl_lss_sare": {"mean": 0.75, "complete_seed_failures": 0},
                "kl_lss_single_expert": {"mean": 0.68, "complete_seed_failures": 1},
            },
        },
    }
    verdict, _checks = evaluate_claim_gate(_manifest(), candidate)
    assert verdict == "PASS: thaw consideration allowed"


def test_pack_claim_gate_is_inconclusive_for_malformed_candidate(tmp_path) -> None:
    artifact_a = tmp_path / "envelope.md"
    artifact_b = tmp_path / "combined.md"
    artifact_a.write_text("# a\n", encoding="utf-8")
    artifact_b.write_text("# b\n", encoding="utf-8")
    manifest = _manifest() | {
        "claim": {
            "id": "doorkey_frozen_claim",
            "status": "frozen",
            "allowed_claim_key": "bounded_teacher_guided_doorkey_sare",
            "allowed_claim_text": "bounded teacher-guided DoorKey SARE result",
            "disallowed_claim_keys": ["ppo_only_routed_win", "specifically_multi_expert_routed_advantage"],
        },
        "canonical_method": {},
        "evaluation": {"task": "DoorKey", "path_key": "external_policy_diagnostics", "episodes": 64},
        "seed_groups": {
            "combined": {"blocks": [{"lane": "original", "seeds": [7]}]},
            "retry_block": {"lane": "fresh_final", "seeds": [47]}},
        "variants": {
            "recovered_token_dense": {},
            "kl_lss_token_dense": {},
            "kl_lss_single_expert": {},
            "baseline_sare": {},
            "kl_lss_sare": {},
        },
        "thresholds": {
            "combined_means": {"kl_lss_sare": 0.7122},
            "combined_complete_seed_failures": {"kl_lss_sare": 1},
            "retry_block_means": {"kl_lss_sare": 0.3125},
            "retry_block_complete_seed_failures": {"kl_lss_sare": 1},
            "keycorridor_transfer_means": {"kl_lss_sare": 0.0},
        },
        "authoritative_reports": {
            "frozen_claim_envelope": str(artifact_a),
            "combined_doorkey_report": str(artifact_b),
        },
        "benchmark_pack": {
            "schema_version": 1,
            "required_artifact_keys": ["frozen_claim_envelope", "combined_doorkey_report"],
        },
        "candidate_pack": {
            "schema_version": 1,
            "required_artifact_roles": [
                "candidate_summary_markdown",
                "candidate_metrics_json",
                "combined_report_markdown",
                "combined_report_csv",
                "retry_block_report_markdown",
                "retry_block_report_csv",
            ],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    frozen_pack = build_frozen_benchmark_pack(manifest_path, manifest, source_commit="deadbeef", source_dirty=False)
    frozen_pack_path = tmp_path / "frozen_pack.json"
    frozen_pack_path.write_text(json.dumps(frozen_pack, indent=2, sort_keys=True), encoding="utf-8")
    candidate = {
        "schema_version": 1,
        "pack_type": "candidate_result_pack",
        "candidate_name": "broken",
        "frozen_pack_reference": {
            "path": str(frozen_pack_path),
            "sha256": sha256_path(frozen_pack_path),
            "claim_id": "doorkey_frozen_claim",
        },
        "task": "DoorKey",
        "evaluation": {"path_key": "external_policy_diagnostics", "episodes": 64},
        "requested_claims": ["bounded_teacher_guided_doorkey_sare"],
        "controls_present": ["kl_lss_sare"],
        "metrics": {},
        "actual_sets": {"combined_lane_seeds": [], "retry_block_lane_seeds": []},
        "artifacts": [],
        "provenance": {"git_commit": "abc123", "git_dirty": False},
    }
    verdict, checks, _frozen_checks, _candidate_checks = evaluate_pack_claim_gate(
        frozen_pack_path,
        frozen_pack,
        tmp_path / "candidate.json",
        candidate,
    )
    assert verdict == "INCONCLUSIVE: missing prerequisites"
    assert any(check.name.startswith("candidate_pack::controls_present") for check in checks)


def test_pack_claim_gate_from_paths_is_inconclusive_for_invalid_json(tmp_path) -> None:
    artifact_a = tmp_path / "envelope.md"
    artifact_b = tmp_path / "combined.md"
    artifact_a.write_text("# a\n", encoding="utf-8")
    artifact_b.write_text("# b\n", encoding="utf-8")
    manifest = _manifest() | {
        "claim": {
            "id": "doorkey_frozen_claim",
            "status": "frozen",
            "allowed_claim_key": "bounded_teacher_guided_doorkey_sare",
            "allowed_claim_text": "bounded teacher-guided DoorKey SARE result",
            "disallowed_claim_keys": ["ppo_only_routed_win", "specifically_multi_expert_routed_advantage"],
        },
        "canonical_method": {},
        "evaluation": {"task": "DoorKey", "path_key": "external_policy_diagnostics", "episodes": 64},
        "seed_groups": {
            "combined": {"blocks": [{"lane": "original", "seeds": [7]}]},
            "retry_block": {"lane": "fresh_final", "seeds": [47]}},
        "variants": {
            "recovered_token_dense": {},
            "kl_lss_token_dense": {},
            "kl_lss_single_expert": {},
            "baseline_sare": {},
            "kl_lss_sare": {},
        },
        "thresholds": {
            "combined_means": {"kl_lss_sare": 0.7122},
            "combined_complete_seed_failures": {"kl_lss_sare": 1},
            "retry_block_means": {"kl_lss_sare": 0.3125},
            "retry_block_complete_seed_failures": {"kl_lss_sare": 1},
            "keycorridor_transfer_means": {"kl_lss_sare": 0.0},
        },
        "authoritative_reports": {
            "frozen_claim_envelope": str(artifact_a),
            "combined_doorkey_report": str(artifact_b),
        },
        "benchmark_pack": {
            "schema_version": 1,
            "required_artifact_keys": ["frozen_claim_envelope", "combined_doorkey_report"],
        },
        "candidate_pack": {
            "schema_version": 1,
            "required_artifact_roles": [
                "candidate_summary_markdown",
                "candidate_metrics_json",
                "combined_report_markdown",
                "combined_report_csv",
                "retry_block_report_markdown",
                "retry_block_report_csv",
            ],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    frozen_pack = build_frozen_benchmark_pack(manifest_path, manifest, source_commit="deadbeef", source_dirty=False)
    frozen_pack_path = tmp_path / "frozen_pack.json"
    frozen_pack_path.write_text(json.dumps(frozen_pack, indent=2, sort_keys=True), encoding="utf-8")
    candidate_path = tmp_path / "broken.json"
    candidate_path.write_text('{"schema_version": 1, "pack_type": "candidate_result_pack", ', encoding="utf-8")
    verdict, checks, _frozen_checks, _candidate_checks = evaluate_pack_claim_gate_from_paths(frozen_pack_path, candidate_path)
    assert verdict == "INCONCLUSIVE: missing prerequisites"
    assert any(check.name == "candidate_pack_load" and check.status == "INCONCLUSIVE" for check in checks)
