from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from psmn_rl.utils.io import get_git_commit, get_git_dirty


FROZEN_PACK_TYPE = "frozen_benchmark_pack"
CANDIDATE_PACK_TYPE = "candidate_result_pack"
DEFAULT_CANDIDATE_ARTIFACT_ROLES = [
    "candidate_summary_markdown",
    "candidate_metrics_json",
    "combined_report_markdown",
    "combined_report_csv",
    "retry_block_report_markdown",
    "retry_block_report_csv",
]
METRIC_FIELDS = ["mean", "min", "max", "complete_seed_failures", "seed_count"]


@dataclass(slots=True)
class PackCheck:
    name: str
    status: str
    detail: str


def load_structured_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raise ValueError(f"Unsupported structured file: {path}")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _expected_lane_seed_set(seed_group: dict[str, Any]) -> set[tuple[str, int]]:
    blocks = seed_group.get("blocks")
    if blocks:
        return {
            (str(block["lane"]), int(seed))
            for block in blocks
            for seed in block.get("seeds", [])
        }
    lanes = seed_group.get("lanes")
    seeds = seed_group.get("seeds", [])
    if lanes is None:
        lanes = [seed_group.get("lane")]
    return {(str(lane), int(seed)) for lane in lanes for seed in seeds}


def _lane_seed_set_from_payload(payload: dict[str, Any], key: str) -> set[tuple[str, int]]:
    actual_sets = payload.get("actual_sets", {})
    values = actual_sets.get(key, [])
    return {(str(lane), int(seed)) for lane, seed in values}


def _required_benchmark_artifact_keys(manifest: dict[str, Any]) -> list[str]:
    benchmark = manifest.get("benchmark_pack", {})
    keys = benchmark.get("required_artifact_keys")
    if keys:
        return [str(key) for key in keys]
    return list(manifest.get("authoritative_reports", {}).keys())


def _required_candidate_artifact_roles(manifest_or_pack: dict[str, Any]) -> list[str]:
    candidate_pack = manifest_or_pack.get("candidate_pack", {})
    roles = candidate_pack.get("required_artifact_roles")
    if roles:
        return [str(role) for role in roles]
    return list(DEFAULT_CANDIDATE_ARTIFACT_ROLES)


def _artifact_record(key: str, path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    exists = path.exists()
    return {
        "key": key,
        "path": str(path),
        "exists": exists,
        "sha256": sha256_path(path) if exists else None,
        "size_bytes": path.stat().st_size if exists else None,
    }


def _candidate_artifact_record(role: str, path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    exists = path.exists()
    return {
        "role": role,
        "path": str(path),
        "exists": exists,
        "sha256": sha256_path(path) if exists else None,
        "size_bytes": path.stat().st_size if exists else None,
    }


def build_frozen_benchmark_pack(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    source_commit: str | None = None,
    source_dirty: bool | None = None,
) -> dict[str, Any]:
    required_keys = _required_benchmark_artifact_keys(manifest)
    reports = manifest.get("authoritative_reports", {})
    artifacts = [_artifact_record(key, reports[key]) for key in required_keys]
    return {
        "schema_version": int(manifest.get("benchmark_pack", {}).get("schema_version", 1)),
        "pack_type": FROZEN_PACK_TYPE,
        "claim": manifest.get("claim", {}),
        "canonical_method": manifest.get("canonical_method", {}),
        "evaluation": manifest.get("evaluation", {}),
        "seed_groups": manifest.get("seed_groups", {}),
        "variants": manifest.get("variants", {}),
        "thresholds": manifest.get("thresholds", {}),
        "thaw_gate": manifest.get("thaw_gate", {}),
        "candidate_pack": {
            "schema_version": int(manifest.get("candidate_pack", {}).get("schema_version", 1)),
            "required_artifact_roles": _required_candidate_artifact_roles(manifest),
        },
        "manifest_reference": {
            "path": str(manifest_path),
            "sha256": sha256_path(manifest_path),
            "required_artifact_keys": required_keys,
        },
        "authoritative_artifacts": artifacts,
        "provenance": {
            "sealed_source_commit": source_commit or get_git_commit(),
            "sealed_source_dirty": get_git_dirty() if source_dirty is None else source_dirty,
            "pack_builder": "psmn_rl.analysis.benchmark_pack",
        },
    }


def validate_frozen_benchmark_pack(pack: dict[str, Any]) -> tuple[str, list[PackCheck]]:
    checks: list[PackCheck] = []
    required_fields = [
        "schema_version",
        "pack_type",
        "claim",
        "evaluation",
        "seed_groups",
        "variants",
        "thresholds",
        "thaw_gate",
        "manifest_reference",
        "authoritative_artifacts",
        "provenance",
    ]
    missing = [field for field in required_fields if field not in pack]
    if missing:
        checks.append(PackCheck("required_fields", "FAIL", "missing required frozen-pack fields: " + ", ".join(f"`{field}`" for field in missing)))
        return "FAIL: frozen benchmark pack invalid", checks
    if pack.get("pack_type") != FROZEN_PACK_TYPE:
        checks.append(PackCheck("pack_type", "FAIL", f"expected `{FROZEN_PACK_TYPE}`, found `{pack.get('pack_type')}`"))
    else:
        checks.append(PackCheck("pack_type", "PASS", f"pack type is `{FROZEN_PACK_TYPE}`"))

    manifest_reference = pack.get("manifest_reference", {})
    manifest_path = Path(str(manifest_reference.get("path", "")))
    if not manifest_path.exists():
        checks.append(PackCheck("manifest_reference", "FAIL", f"manifest path `{manifest_path}` is missing"))
        return "FAIL: frozen benchmark pack invalid", checks

    manifest_hash = sha256_path(manifest_path)
    if manifest_hash != manifest_reference.get("sha256"):
        checks.append(
            PackCheck(
                "manifest_hash",
                "FAIL",
                f"manifest hash `{manifest_hash}` does not match sealed `{manifest_reference.get('sha256')}`",
            )
        )
    else:
        checks.append(PackCheck("manifest_hash", "PASS", f"manifest hash matches `{manifest_hash}`"))

    manifest = load_structured_file(manifest_path)
    expected_schema = int(manifest.get("benchmark_pack", {}).get("schema_version", 1))
    if int(pack.get("schema_version", 0)) != expected_schema:
        checks.append(PackCheck("schema_version", "FAIL", f"pack schema `{pack.get('schema_version')}` does not match manifest `{expected_schema}`"))
    else:
        checks.append(PackCheck("schema_version", "PASS", f"pack schema version `{expected_schema}` is recognized"))

    for section in ["claim", "canonical_method", "evaluation", "seed_groups", "variants", "thresholds", "thaw_gate"]:
        if pack.get(section) != manifest.get(section):
            checks.append(PackCheck(section, "FAIL", f"sealed `{section}` does not match current manifest `{manifest_path}`"))
        else:
            checks.append(PackCheck(section, "PASS", f"sealed `{section}` matches the manifest"))

    required_keys = set(_required_benchmark_artifact_keys(manifest))
    artifact_records = {str(item.get("key")): item for item in pack.get("authoritative_artifacts", [])}
    missing_keys = sorted(required_keys - set(artifact_records))
    if missing_keys:
        checks.append(PackCheck("authoritative_artifacts", "FAIL", "missing artifact keys: " + ", ".join(f"`{key}`" for key in missing_keys)))
    else:
        checks.append(PackCheck("authoritative_artifacts", "PASS", "all required artifact keys are present"))

    reports = manifest.get("authoritative_reports", {})
    for key in sorted(required_keys):
        artifact = artifact_records.get(key)
        if artifact is None:
            continue
        expected_path = reports.get(key)
        sealed_path = str(artifact.get("path", ""))
        if sealed_path != expected_path:
            checks.append(PackCheck(f"artifact_path::{key}", "FAIL", f"sealed path `{sealed_path}` does not match manifest path `{expected_path}`"))
            continue
        path = Path(sealed_path)
        if not path.exists():
            checks.append(PackCheck(f"artifact_hash::{key}", "FAIL", f"required artifact `{sealed_path}` is missing"))
            continue
        current_hash = sha256_path(path)
        if current_hash != artifact.get("sha256"):
            checks.append(PackCheck(f"artifact_hash::{key}", "FAIL", f"artifact `{sealed_path}` hash drifted from `{artifact.get('sha256')}` to `{current_hash}`"))
        else:
            checks.append(PackCheck(f"artifact_hash::{key}", "PASS", f"artifact `{sealed_path}` hash matches `{current_hash}`"))

    verdict = "PASS: frozen benchmark pack validated" if all(check.status == "PASS" for check in checks) else "FAIL: frozen benchmark pack invalid"
    return verdict, checks


def _metric_block_has_required_shape(block: dict[str, Any], variants: list[str]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for variant in variants:
        if variant not in block:
            missing.append(f"{variant}:<variant>")
            continue
        for field in METRIC_FIELDS:
            if field not in block[variant]:
                missing.append(f"{variant}:{field}")
    return (not missing), missing


def validate_candidate_result_pack(
    frozen_pack: dict[str, Any],
    frozen_pack_path: Path,
    candidate_pack: dict[str, Any],
) -> tuple[str, list[PackCheck]]:
    checks: list[PackCheck] = []
    required_fields = [
        "schema_version",
        "pack_type",
        "candidate_name",
        "frozen_pack_reference",
        "task",
        "evaluation",
        "requested_claims",
        "controls_present",
        "metrics",
        "actual_sets",
        "artifacts",
        "provenance",
    ]
    missing = [field for field in required_fields if field not in candidate_pack]
    if missing:
        checks.append(PackCheck("required_fields", "FAIL", "missing required candidate-pack fields: " + ", ".join(f"`{field}`" for field in missing)))
        return "FAIL: candidate result pack invalid", checks

    expected_schema = int(frozen_pack.get("candidate_pack", {}).get("schema_version", 1))
    if candidate_pack.get("pack_type") != CANDIDATE_PACK_TYPE:
        checks.append(PackCheck("pack_type", "FAIL", f"expected `{CANDIDATE_PACK_TYPE}`, found `{candidate_pack.get('pack_type')}`"))
    else:
        checks.append(PackCheck("pack_type", "PASS", f"pack type is `{CANDIDATE_PACK_TYPE}`"))
    if int(candidate_pack.get("schema_version", 0)) != expected_schema:
        checks.append(PackCheck("schema_version", "FAIL", f"candidate schema `{candidate_pack.get('schema_version')}` does not match frozen requirement `{expected_schema}`"))
    else:
        checks.append(PackCheck("schema_version", "PASS", f"candidate schema version `{expected_schema}` is recognized"))

    frozen_reference = candidate_pack.get("frozen_pack_reference", {})
    frozen_hash = sha256_path(frozen_pack_path) if frozen_pack_path.exists() else None
    if str(frozen_reference.get("path", "")) != str(frozen_pack_path):
        checks.append(PackCheck("frozen_pack_reference.path", "FAIL", f"candidate references `{frozen_reference.get('path')}` instead of `{frozen_pack_path}`"))
    else:
        checks.append(PackCheck("frozen_pack_reference.path", "PASS", f"candidate references `{frozen_pack_path}`"))
    if frozen_hash != frozen_reference.get("sha256"):
        checks.append(PackCheck("frozen_pack_reference.sha256", "FAIL", f"candidate references frozen-pack hash `{frozen_reference.get('sha256')}` instead of current `{frozen_hash}`"))
    else:
        checks.append(PackCheck("frozen_pack_reference.sha256", "PASS", f"candidate references frozen-pack hash `{frozen_hash}`"))
    claim_id = str(frozen_pack.get("claim", {}).get("id", ""))
    if frozen_reference.get("claim_id") != claim_id:
        checks.append(PackCheck("frozen_pack_reference.claim_id", "FAIL", f"candidate claim id `{frozen_reference.get('claim_id')}` does not match frozen `{claim_id}`"))
    else:
        checks.append(PackCheck("frozen_pack_reference.claim_id", "PASS", f"candidate claim id matches `{claim_id}`"))

    evaluation = candidate_pack.get("evaluation", {})
    frozen_eval = frozen_pack.get("evaluation", {})
    task = str(candidate_pack.get("task", ""))
    if task != str(frozen_eval.get("task", "")):
        checks.append(PackCheck("task", "FAIL", f"candidate task `{task or '-'}` does not match frozen `{frozen_eval.get('task')}`"))
    else:
        checks.append(PackCheck("task", "PASS", f"candidate task matches `{task}`"))
    if str(evaluation.get("path_key", "")) != str(frozen_eval.get("path_key", "")) or int(evaluation.get("episodes", 0)) != int(frozen_eval.get("episodes", 0)):
        checks.append(
            PackCheck(
                "evaluation",
                "FAIL",
                f"candidate evaluation `{evaluation}` does not match frozen `{frozen_eval}`",
            )
        )
    else:
        checks.append(PackCheck("evaluation", "PASS", "candidate uses the required DoorKey external-64 evaluation path"))

    required_controls = set(frozen_pack.get("thaw_gate", {}).get("required_controls", []))
    controls_present = set(candidate_pack.get("controls_present", []))
    missing_controls = sorted(required_controls - controls_present)
    if missing_controls:
        checks.append(PackCheck("controls_present", "FAIL", "candidate is missing required controls: " + ", ".join(f"`{control}`" for control in missing_controls)))
    else:
        checks.append(PackCheck("controls_present", "PASS", "candidate includes all required fairness controls"))

    variants = list(frozen_pack.get("variants", {}).keys())
    metrics = candidate_pack.get("metrics", {})
    combined = metrics.get("combined", {})
    retry_block = metrics.get("retry_block", {})
    ok_combined, missing_combined = _metric_block_has_required_shape(combined, variants)
    ok_retry, missing_retry = _metric_block_has_required_shape(retry_block, variants)
    if not ok_combined:
        checks.append(PackCheck("metrics.combined", "FAIL", "combined metrics are missing fields: " + ", ".join(f"`{item}`" for item in missing_combined)))
    else:
        checks.append(PackCheck("metrics.combined", "PASS", "combined metrics expose every required variant field"))
    if not ok_retry:
        checks.append(PackCheck("metrics.retry_block", "FAIL", "retry-block metrics are missing fields: " + ", ".join(f"`{item}`" for item in missing_retry)))
    else:
        checks.append(PackCheck("metrics.retry_block", "PASS", "retry-block metrics expose every required variant field"))

    expected_combined_set = _expected_lane_seed_set(frozen_pack.get("seed_groups", {}).get("combined", {}))
    actual_combined_set = _lane_seed_set_from_payload(candidate_pack, "combined_lane_seeds")
    if actual_combined_set != expected_combined_set:
        checks.append(PackCheck("actual_sets.combined", "FAIL", f"candidate combined lane/seed set `{sorted(actual_combined_set)}` does not match frozen `{sorted(expected_combined_set)}`"))
    else:
        checks.append(PackCheck("actual_sets.combined", "PASS", "candidate combined lane/seed set matches the frozen benchmark"))
    expected_retry_set = _expected_lane_seed_set(frozen_pack.get("seed_groups", {}).get("retry_block", {}))
    actual_retry_set = _lane_seed_set_from_payload(candidate_pack, "retry_block_lane_seeds")
    if actual_retry_set != expected_retry_set:
        checks.append(PackCheck("actual_sets.retry_block", "FAIL", f"candidate retry-block lane/seed set `{sorted(actual_retry_set)}` does not match frozen `{sorted(expected_retry_set)}`"))
    else:
        checks.append(PackCheck("actual_sets.retry_block", "PASS", "candidate retry-block lane/seed set matches the frozen benchmark"))

    required_roles = set(_required_candidate_artifact_roles(frozen_pack))
    artifacts = {str(item.get("role")): item for item in candidate_pack.get("artifacts", [])}
    missing_roles = sorted(required_roles - set(artifacts))
    if missing_roles:
        checks.append(PackCheck("artifacts", "FAIL", "candidate pack is missing required artifact roles: " + ", ".join(f"`{role}`" for role in missing_roles)))
    else:
        checks.append(PackCheck("artifacts", "PASS", "candidate pack exposes all required artifact roles"))
    for role in sorted(required_roles):
        artifact = artifacts.get(role)
        if artifact is None:
            continue
        path = Path(str(artifact.get("path", "")))
        if not path.exists():
            checks.append(PackCheck(f"artifact_hash::{role}", "FAIL", f"candidate artifact `{path}` is missing"))
            continue
        current_hash = sha256_path(path)
        if current_hash != artifact.get("sha256"):
            checks.append(PackCheck(f"artifact_hash::{role}", "FAIL", f"candidate artifact `{path}` hash drifted from `{artifact.get('sha256')}` to `{current_hash}`"))
        else:
            checks.append(PackCheck(f"artifact_hash::{role}", "PASS", f"candidate artifact `{path}` hash matches `{current_hash}`"))

    verdict = "PASS: candidate result pack validated" if all(check.status == "PASS" for check in checks) else "FAIL: candidate result pack invalid"
    return verdict, checks


def render_frozen_benchmark_pack_report(pack_path: Path, pack: dict[str, Any]) -> str:
    lines = [
        "# Frozen Benchmark Pack",
        "",
        f"- pack json: `{pack_path}`",
        f"- claim id: `{pack.get('claim', {}).get('id', '-')}`",
        f"- frozen status: `{pack.get('claim', {}).get('status', '-')}`",
        f"- sealed source commit: `{pack.get('provenance', {}).get('sealed_source_commit', '-')}`",
        f"- sealed source dirty: `{pack.get('provenance', {}).get('sealed_source_dirty', '-')}`",
        f"- schema version: `{pack.get('schema_version', '-')}`",
        f"- manifest: `{pack.get('manifest_reference', {}).get('path', '-')}`",
        "",
        "## Claim Envelope",
        "",
        f"- allowed: `{pack.get('claim', {}).get('allowed_claim_text', '-')}`",
        "- disallowed:",
    ]
    for item in pack.get("claim", {}).get("disallowed_claim_keys", []):
        lines.append(f"  - `{item}`")
    lines.extend(
        [
            "",
            "## Frozen Thresholds",
            "",
            f"- combined KL learner-state `SARE` mean: `{pack.get('thresholds', {}).get('combined_means', {}).get('kl_lss_sare', 0.0):.4f}`",
            f"- retry-block KL learner-state `SARE` mean: `{pack.get('thresholds', {}).get('retry_block_means', {}).get('kl_lss_sare', 0.0):.4f}`",
            f"- retry-block KL learner-state `single_expert` mean: `{pack.get('thresholds', {}).get('retry_block_means', {}).get('kl_lss_single_expert', 0.0):.4f}`",
            "",
            "## Canonical Artifacts",
            "",
            "| Key | Path | SHA256 | Size (bytes) |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for artifact in pack.get("authoritative_artifacts", []):
        lines.append(f"| `{artifact['key']}` | `{artifact['path']}` | `{artifact.get('sha256') or '-'}` | {artifact.get('size_bytes') or 0} |")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "- This pack is the immutable machine-readable baseline unit for the current frozen DoorKey claim.",
            "- It is derived from the frozen manifest and authoritative reports, and it is meant to be validated before any future thaw candidate is discussed.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_pack_validation_report(pack_path: Path, verdict: str, checks: list[PackCheck]) -> str:
    lines = [
        "# Frozen Benchmark Pack Validation",
        "",
        f"- pack: `{pack_path}`",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for check in checks:
        lines.append(f"| {check.name} | `{check.status}` | {check.detail} |")
    lines.extend(["", "## Verdict", "", verdict])
    return "\n".join(lines) + "\n"


def render_benchmark_pack_schema_report(manifest: dict[str, Any]) -> str:
    required_keys = _required_benchmark_artifact_keys(manifest)
    lines = [
        "# Benchmark Pack Schema Report",
        "",
        f"- frozen pack schema version: `{manifest.get('benchmark_pack', {}).get('schema_version', 1)}`",
        f"- candidate pack schema version: `{manifest.get('candidate_pack', {}).get('schema_version', 1)}`",
        "",
        "## Frozen Benchmark Pack Required Fields",
        "",
        "- `schema_version`",
        "- `pack_type`",
        "- `claim`",
        "- `canonical_method`",
        "- `evaluation`",
        "- `seed_groups`",
        "- `variants`",
        "- `thresholds`",
        "- `thaw_gate`",
        "- `candidate_pack`",
        "- `manifest_reference`",
        "- `authoritative_artifacts`",
        "- `provenance`",
        "",
        "## Frozen Benchmark Artifact Keys",
        "",
    ]
    for key in required_keys:
        lines.append(f"- `{key}`")
    lines.extend(
        [
            "",
            "## Validator Checks",
            "",
            "- manifest path exists and hash matches the sealed manifest reference",
            "- pack schema version is recognized",
            "- sealed claim, evaluation, variants, thresholds, and thaw gate still match the manifest",
            "- every required authoritative artifact exists and still matches its sealed hash",
            "",
            "A pack passes only if every required field, manifest check, and artifact hash check passes.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_candidate_result_pack_schema(manifest_or_pack: dict[str, Any], frozen_pack_path: Path | None = None) -> str:
    required_roles = _required_candidate_artifact_roles(manifest_or_pack)
    lines = [
        "# Candidate Result Pack Schema",
        "",
        f"- schema version: `{manifest_or_pack.get('candidate_pack', {}).get('schema_version', 1)}`",
        f"- frozen pack reference: `{frozen_pack_path}`" if frozen_pack_path is not None else "- frozen pack reference: fill in the sealed frozen benchmark pack path",
        "",
        "## Required Fields",
        "",
        "- `schema_version`",
        "- `pack_type`",
        "- `candidate_name`",
        "- `frozen_pack_reference`",
        "- `task`",
        "- `evaluation`",
        "- `requested_claims`",
        "- `controls_present`",
        "- `metrics`",
        "- `actual_sets`",
        "- `artifacts`",
        "- `provenance`",
        "",
        "## Required Metrics",
        "",
        "- both `combined` and `retry_block` must include every canonical variant",
        f"- each variant must expose: `{', '.join(METRIC_FIELDS)}`",
        "",
        "## Required Artifact Roles",
        "",
    ]
    for role in required_roles:
        lines.append(f"- `{role}`")
    lines.extend(
        [
            "",
            "## Validator Checks",
            "",
            "- pack references the current sealed frozen benchmark pack path, claim id, and hash",
            "- evaluation stays on external 64-episode DoorKey",
            "- required fairness controls are present",
            "- combined and retry-block lane/seed coverage matches the frozen benchmark",
            "- required artifact roles exist and their hashes match the filesystem",
            "",
            "Malformed or incomplete candidate packs are treated as `INCONCLUSIVE: missing prerequisites` by the pack-based claim gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_candidate_result_pack_template(frozen_pack_path: Path, frozen_pack: dict[str, Any]) -> dict[str, Any]:
    variants = list(frozen_pack.get("variants", {}).keys())
    metric_template = {
        variant: {field: "<fill-me>" for field in METRIC_FIELDS}
        for variant in variants
    }
    return {
        "schema_version": int(frozen_pack.get("candidate_pack", {}).get("schema_version", 1)),
        "pack_type": CANDIDATE_PACK_TYPE,
        "candidate_name": "<fill-me>",
        "frozen_pack_reference": {
            "path": str(frozen_pack_path),
            "sha256": sha256_path(frozen_pack_path),
            "claim_id": frozen_pack.get("claim", {}).get("id", ""),
        },
        "task": frozen_pack.get("evaluation", {}).get("task", "DoorKey"),
        "evaluation": {
            "path_key": frozen_pack.get("evaluation", {}).get("path_key", "external_policy_diagnostics"),
            "episodes": int(frozen_pack.get("evaluation", {}).get("episodes", 64)),
        },
        "requested_claims": [frozen_pack.get("claim", {}).get("allowed_claim_key", "bounded_teacher_guided_doorkey_sare")],
        "controls_present": list(frozen_pack.get("variants", {}).keys()),
        "metrics": {
            "combined": metric_template,
            "retry_block": metric_template,
        },
        "actual_sets": {
            "combined_lane_seeds": [[lane, seed] for lane, seed in sorted(_expected_lane_seed_set(frozen_pack.get("seed_groups", {}).get("combined", {})))],
            "retry_block_lane_seeds": [[lane, seed] for lane, seed in sorted(_expected_lane_seed_set(frozen_pack.get("seed_groups", {}).get("retry_block", {})))],
        },
        "artifacts": [
            {"role": role, "path": "<fill-me>", "sha256": "<fill-me>", "size_bytes": "<fill-me>"}
            for role in _required_candidate_artifact_roles(frozen_pack)
        ],
        "provenance": {
            "git_commit": "<fill-me>",
            "git_dirty": "<fill-me>",
            "notes": "<fill-me>",
        },
    }


def build_current_frozen_candidate_pack(
    manifest: dict[str, Any],
    frozen_pack_path: Path,
    frozen_pack: dict[str, Any],
    candidate_source_path: Path,
) -> dict[str, Any]:
    candidate = load_structured_file(candidate_source_path)
    reports = manifest.get("authoritative_reports", {})
    artifact_paths = {
        "candidate_summary_markdown": reports["frozen_validation_report"],
        "candidate_metrics_json": str(candidate_source_path),
        "combined_report_markdown": reports["combined_doorkey_report"],
        "combined_report_csv": reports["combined_doorkey_csv"],
        "retry_block_report_markdown": reports["final_block_report"],
        "retry_block_report_csv": reports["final_block_csv"],
    }
    return {
        "schema_version": int(frozen_pack.get("candidate_pack", {}).get("schema_version", 1)),
        "pack_type": CANDIDATE_PACK_TYPE,
        "candidate_name": "current_frozen_candidate_pack",
        "frozen_pack_reference": {
            "path": str(frozen_pack_path),
            "sha256": sha256_path(frozen_pack_path),
            "claim_id": frozen_pack.get("claim", {}).get("id", ""),
        },
        "task": candidate.get("task", frozen_pack.get("evaluation", {}).get("task", "DoorKey")),
        "evaluation": candidate.get("evaluation", {}),
        "requested_claims": candidate.get("requested_claims", []),
        "controls_present": candidate.get("controls_present", []),
        "metrics": candidate.get("metrics", {}),
        "actual_sets": candidate.get("actual_sets", {}),
        "artifacts": [_candidate_artifact_record(role, path) for role, path in artifact_paths.items()],
        "provenance": {
            **candidate.get("provenance", {}),
            "candidate_source_json": str(candidate_source_path),
        },
    }


def build_incomplete_candidate_pack(frozen_pack_path: Path, frozen_pack: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": int(frozen_pack.get("candidate_pack", {}).get("schema_version", 1)),
        "pack_type": CANDIDATE_PACK_TYPE,
        "candidate_name": "incomplete_candidate_pack_example",
        "frozen_pack_reference": {
            "path": str(frozen_pack_path),
            "sha256": sha256_path(frozen_pack_path),
            "claim_id": frozen_pack.get("claim", {}).get("id", ""),
        },
        "task": frozen_pack.get("evaluation", {}).get("task", "DoorKey"),
        "evaluation": {
            "path_key": frozen_pack.get("evaluation", {}).get("path_key", "external_policy_diagnostics"),
            "episodes": int(frozen_pack.get("evaluation", {}).get("episodes", 64)),
        },
        "requested_claims": [frozen_pack.get("claim", {}).get("allowed_claim_key", "bounded_teacher_guided_doorkey_sare")],
        "controls_present": ["recovered_token_dense", "kl_lss_sare"],
        "metrics": {
            "combined": {
                "kl_lss_sare": {"mean": 0.7122, "min": 0.0, "max": 1.0, "complete_seed_failures": 1, "seed_count": 12}
            }
        },
        "actual_sets": {
            "combined_lane_seeds": [],
            "retry_block_lane_seeds": [],
        },
        "artifacts": [
            {
                "role": "candidate_metrics_json",
                "path": "outputs/reports/does_not_exist.json",
                "sha256": "missing",
                "size_bytes": 0,
            }
        ],
        "provenance": {
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark-pack sealing and schema validation for the frozen DoorKey claim.")
    sub = parser.add_subparsers(dest="command", required=True)

    seal = sub.add_parser("seal-frozen-pack")
    seal.add_argument("--manifest", required=True)
    seal.add_argument("--output-markdown", required=True)
    seal.add_argument("--output-json", required=True)
    seal.add_argument("--source-commit")
    seal.add_argument("--source-dirty")

    validate = sub.add_parser("validate-frozen-pack")
    validate.add_argument("--pack", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--json-output", required=True)

    schema = sub.add_parser("benchmark-pack-schema-report")
    schema.add_argument("--manifest", required=True)
    schema.add_argument("--output", required=True)

    candidate_schema = sub.add_parser("candidate-pack-schema")
    candidate_schema.add_argument("--frozen-pack", required=True)
    candidate_schema.add_argument("--output-markdown", required=True)
    candidate_schema.add_argument("--output-json-template", required=True)

    current_candidate = sub.add_parser("build-current-frozen-candidate-pack")
    current_candidate.add_argument("--manifest", required=True)
    current_candidate.add_argument("--frozen-pack", required=True)
    current_candidate.add_argument("--candidate-json", required=True)
    current_candidate.add_argument("--output", required=True)

    incomplete = sub.add_parser("build-incomplete-candidate-pack")
    incomplete.add_argument("--frozen-pack", required=True)
    incomplete.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = args.command

    if command == "seal-frozen-pack":
        manifest_path = Path(args.manifest)
        manifest = load_structured_file(manifest_path)
        source_dirty = None
        if args.source_dirty is not None:
            source_dirty = str(args.source_dirty).lower() in {"1", "true", "yes"}
        pack = build_frozen_benchmark_pack(
            manifest_path,
            manifest,
            source_commit=args.source_commit,
            source_dirty=source_dirty,
        )
        output_json = Path(args.output_json)
        output_md = Path(args.output_markdown)
        _write_json(output_json, pack)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_frozen_benchmark_pack_report(output_json, pack), encoding="utf-8")
        return

    if command == "validate-frozen-pack":
        pack_path = Path(args.pack)
        pack = load_structured_file(pack_path)
        verdict, checks = validate_frozen_benchmark_pack(pack)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_pack_validation_report(pack_path, verdict, checks), encoding="utf-8")
        _write_json(
            Path(args.json_output),
            {
                "pack": str(pack_path),
                "verdict": verdict,
                "checks": [asdict(check) for check in checks],
            },
        )
        return

    if command == "benchmark-pack-schema-report":
        manifest = load_structured_file(Path(args.manifest))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_benchmark_pack_schema_report(manifest), encoding="utf-8")
        return

    if command == "candidate-pack-schema":
        frozen_pack_path = Path(args.frozen_pack)
        frozen_pack = load_structured_file(frozen_pack_path)
        output_md = Path(args.output_markdown)
        output_json = Path(args.output_json_template)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_candidate_result_pack_schema(frozen_pack, frozen_pack_path), encoding="utf-8")
        _write_json(output_json, build_candidate_result_pack_template(frozen_pack_path, frozen_pack))
        return

    if command == "build-current-frozen-candidate-pack":
        manifest = load_structured_file(Path(args.manifest))
        frozen_pack_path = Path(args.frozen_pack)
        frozen_pack = load_structured_file(frozen_pack_path)
        candidate = build_current_frozen_candidate_pack(
            manifest,
            frozen_pack_path,
            frozen_pack,
            Path(args.candidate_json),
        )
        _write_json(Path(args.output), candidate)
        return

    if command == "build-incomplete-candidate-pack":
        frozen_pack_path = Path(args.frozen_pack)
        frozen_pack = load_structured_file(frozen_pack_path)
        _write_json(Path(args.output), build_incomplete_candidate_pack(frozen_pack_path, frozen_pack))
        return


if __name__ == "__main__":
    main()
