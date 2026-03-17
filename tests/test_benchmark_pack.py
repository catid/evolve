from __future__ import annotations

import json
from pathlib import Path

import yaml

from psmn_rl.analysis.benchmark_pack import (
    build_frozen_benchmark_pack,
    sha256_path,
    validate_candidate_result_pack,
    validate_frozen_benchmark_pack,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _manifest(tmp_path: Path) -> tuple[Path, dict]:
    artifact_a = _write(tmp_path / "outputs" / "envelope.md", "# envelope\n")
    artifact_b = _write(tmp_path / "outputs" / "combined.md", "# combined\n")
    manifest = {
        "claim": {
            "id": "doorkey_frozen_claim",
            "status": "frozen",
            "allowed_claim_key": "bounded_teacher_guided_doorkey_sare",
            "allowed_claim_text": "bounded teacher-guided DoorKey SARE result",
            "disallowed_claim_keys": ["ppo_only_routed_win"],
        },
        "canonical_method": {
            "teacher": "flat_dense",
            "supervision": "teacher_logit_kl",
            "learner_state_collection": "append_all",
            "student": "sare",
        },
        "evaluation": {
            "task": "DoorKey",
            "path_key": "external_policy_diagnostics",
            "episodes": 64,
        },
        "seed_groups": {
            "combined": {"blocks": [{"lane": "original", "seeds": [7]}]},
            "retry_block": {"lane": "fresh_final", "seeds": [47]},
        },
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
        "thaw_gate": {
            "required_controls": [
                "recovered_token_dense",
                "kl_lss_token_dense",
                "kl_lss_single_expert",
                "baseline_sare",
            ],
            "required_evaluation": {
                "task": "DoorKey",
                "path_key": "external_policy_diagnostics",
                "episodes": 64,
            },
            "retry_block": {
                "candidate_sare_mean_must_exceed": 0.3125,
                "candidate_sare_complete_seed_failures_must_be_lte": 1,
            },
            "combined_picture": {
                "candidate_sare_mean_must_be_gte": 0.7122,
                "candidate_sare_complete_seed_failures_must_be_lte": 1,
            },
            "disallowed_claim_keys": ["ppo_only_routed_win"],
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
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path, manifest


def _candidate_pack(tmp_path: Path, frozen_pack_path: Path, frozen_pack: dict) -> dict:
    artifact_paths = {}
    for role in frozen_pack["candidate_pack"]["required_artifact_roles"]:
        suffix = ".csv" if role.endswith("_csv") else ".md"
        if role == "candidate_metrics_json":
            suffix = ".json"
        path = _write(tmp_path / f"{role}{suffix}", "{}\n")
        artifact_paths[role] = path

    metrics = {}
    for section in ["combined", "retry_block"]:
        metrics[section] = {}
        for variant in frozen_pack["variants"]:
            metrics[section][variant] = {
                "mean": 0.8 if variant == "kl_lss_sare" else 0.7,
                "min": 0.0,
                "max": 1.0,
                "complete_seed_failures": 0,
                "seed_count": 1,
            }
    metrics["retry_block"]["kl_lss_single_expert"]["mean"] = 0.6
    metrics["combined"]["kl_lss_single_expert"]["mean"] = 0.68

    return {
        "schema_version": 1,
        "pack_type": "candidate_result_pack",
        "candidate_name": "candidate",
        "frozen_pack_reference": {
            "path": str(frozen_pack_path),
            "sha256": sha256_path(frozen_pack_path),
            "claim_id": frozen_pack["claim"]["id"],
        },
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
        "metrics": metrics,
        "actual_sets": {
            "combined_lane_seeds": [["original", 7]],
            "retry_block_lane_seeds": [["fresh_final", 47]],
        },
        "artifacts": [
            {
                "role": role,
                "path": str(path),
                "sha256": sha256_path(path),
                "size_bytes": path.stat().st_size,
            }
            for role, path in artifact_paths.items()
        ],
        "provenance": {"git_commit": "deadbeef", "git_dirty": False},
    }


def test_validate_frozen_benchmark_pack_passes_for_matching_manifest_and_hashes(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    pack = build_frozen_benchmark_pack(manifest_path, manifest, source_commit="deadbeef", source_dirty=False)
    verdict, checks = validate_frozen_benchmark_pack(pack)
    assert verdict == "PASS: frozen benchmark pack validated"
    assert any(check.name == "manifest_hash" and check.status == "PASS" for check in checks)


def test_validate_candidate_result_pack_fails_for_missing_controls(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    frozen_pack = build_frozen_benchmark_pack(manifest_path, manifest, source_commit="deadbeef", source_dirty=False)
    frozen_pack_path = tmp_path / "frozen_pack.json"
    frozen_pack_path.write_text(json.dumps(frozen_pack, indent=2, sort_keys=True), encoding="utf-8")
    candidate = _candidate_pack(tmp_path, frozen_pack_path, frozen_pack)
    metrics_artifact_path = Path(candidate["artifacts"][1]["path"])
    metrics_artifact_path.write_text(
        json.dumps(
            {
                "schema_version": candidate["schema_version"],
                "candidate_name": candidate["candidate_name"],
                "task": candidate["task"],
                "evaluation": candidate["evaluation"],
                "requested_claims": candidate["requested_claims"],
                "controls_present": candidate["controls_present"],
                "metrics": candidate["metrics"],
                "actual_sets": candidate["actual_sets"],
                "provenance": candidate["provenance"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    candidate["artifacts"][1]["sha256"] = sha256_path(metrics_artifact_path)
    candidate["artifacts"][1]["size_bytes"] = metrics_artifact_path.stat().st_size
    candidate["controls_present"] = ["kl_lss_sare"]
    verdict, checks = validate_candidate_result_pack(frozen_pack, frozen_pack_path, candidate)
    assert verdict == "FAIL: candidate result pack invalid"
    assert any(check.name == "controls_present" and check.status == "FAIL" for check in checks)


def test_validate_candidate_result_pack_fails_when_metrics_artifact_diverges(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    frozen_pack = build_frozen_benchmark_pack(manifest_path, manifest, source_commit="deadbeef", source_dirty=False)
    frozen_pack_path = tmp_path / "frozen_pack.json"
    frozen_pack_path.write_text(json.dumps(frozen_pack, indent=2, sort_keys=True), encoding="utf-8")
    candidate = _candidate_pack(tmp_path, frozen_pack_path, frozen_pack)
    metrics_artifact_path = Path(candidate["artifacts"][1]["path"])
    metrics_artifact_path.write_text(
        json.dumps(
            {
                "schema_version": candidate["schema_version"],
                "candidate_name": candidate["candidate_name"],
                "task": candidate["task"],
                "evaluation": candidate["evaluation"],
                "requested_claims": candidate["requested_claims"],
                "controls_present": candidate["controls_present"],
                "metrics": candidate["metrics"],
                "actual_sets": candidate["actual_sets"],
                "provenance": candidate["provenance"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    candidate["artifacts"][1]["sha256"] = sha256_path(metrics_artifact_path)
    candidate["artifacts"][1]["size_bytes"] = metrics_artifact_path.stat().st_size
    candidate["metrics"]["retry_block"]["kl_lss_sare"]["mean"] = 0.95
    verdict, checks = validate_candidate_result_pack(frozen_pack, frozen_pack_path, candidate)
    assert verdict == "FAIL: candidate result pack invalid"
    assert any(check.name == "artifact_consistency::candidate_metrics_json" and check.status == "FAIL" for check in checks)
