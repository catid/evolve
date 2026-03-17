from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from psmn_rl.analysis.benchmark_pack import (
    load_structured_file,
    safe_load_structured_file,
    validate_candidate_result_pack,
    validate_frozen_benchmark_pack,
)


@dataclass(slots=True)
class GateCheck:
    name: str
    status: str
    detail: str


def load_manifest(path: Path) -> dict[str, Any]:
    return load_structured_file(path)


def load_candidate(path: Path) -> dict[str, Any]:
    return load_structured_file(path)


def _as_mapping(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _as_string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return list(value)


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def evaluate_claim_gate(manifest: dict[str, Any], candidate: dict[str, Any]) -> tuple[str, list[GateCheck]]:
    checks: list[GateCheck] = []
    thaw_gate = manifest.get("thaw_gate", {})
    required_eval = thaw_gate.get("required_evaluation", {})
    expected_task = str(required_eval.get("task", "DoorKey"))
    expected_path = str(required_eval.get("path_key", "external_policy_diagnostics"))
    expected_episodes = int(required_eval.get("episodes", 64))

    task = str(candidate.get("task", ""))
    evaluation = _as_mapping(candidate.get("evaluation"))
    if evaluation is None:
        checks.append(GateCheck("evaluation_shape", "INCONCLUSIVE", "candidate evaluation must be a mapping"))
        evaluation = {}
    else:
        checks.append(GateCheck("evaluation_shape", "PASS", "candidate evaluation is a mapping"))
    if task != expected_task:
        checks.append(GateCheck("task", "INCONCLUSIVE", f"candidate task `{task or '-'}` does not match required `{expected_task}`"))
    else:
        checks.append(GateCheck("task", "PASS", f"candidate task matches `{expected_task}`"))

    candidate_path = str(evaluation.get("path_key", ""))
    candidate_episodes = _coerce_int(evaluation.get("episodes"))
    if candidate_path != expected_path or candidate_episodes != expected_episodes:
        checks.append(
            GateCheck(
                "evaluation_path",
                "INCONCLUSIVE",
                f"candidate path `{candidate_path or '-'}` / episodes `{candidate_episodes if candidate_episodes is not None else '-'}` do not match required `{expected_path}` / `{expected_episodes}`",
            )
        )
    else:
        checks.append(GateCheck("evaluation_path", "PASS", f"candidate uses `{expected_path}` with `{expected_episodes}` episodes"))

    controls_list = _as_string_list(candidate.get("controls_present"))
    if controls_list is None:
        checks.append(GateCheck("fairness_controls_shape", "INCONCLUSIVE", "controls_present must be a list of strings"))
        controls_present: set[str] = set()
    else:
        checks.append(GateCheck("fairness_controls_shape", "PASS", "controls_present is a list of strings"))
        controls_present = set(controls_list)
    required_controls = set(thaw_gate.get("required_controls", []))
    missing_controls = sorted(required_controls - controls_present)
    if missing_controls:
        checks.append(
            GateCheck(
                "fairness_controls",
                "INCONCLUSIVE",
                "missing required controls: " + ", ".join(f"`{control}`" for control in missing_controls),
            )
        )
    else:
        checks.append(GateCheck("fairness_controls", "PASS", "all required structured controls are present"))

    requested_claims_list = _as_string_list(candidate.get("requested_claims"))
    if requested_claims_list is None:
        checks.append(GateCheck("claim_scope_shape", "INCONCLUSIVE", "requested_claims must be a list of strings"))
        requested_claims: set[str] = set()
    else:
        checks.append(GateCheck("claim_scope_shape", "PASS", "requested_claims is a list of strings"))
        requested_claims = set(requested_claims_list)
    disallowed = sorted(requested_claims & set(thaw_gate.get("disallowed_claim_keys", [])))
    if disallowed:
        checks.append(
            GateCheck(
                "claim_scope",
                "FAIL",
                "candidate requests disallowed widening: " + ", ".join(f"`{claim}`" for claim in disallowed),
            )
        )
    else:
        checks.append(GateCheck("claim_scope", "PASS", "candidate stays inside the frozen claim envelope"))

    metrics = _as_mapping(candidate.get("metrics"))
    if metrics is None:
        checks.append(GateCheck("candidate_metrics_shape", "INCONCLUSIVE", "metrics must be a mapping"))
        metrics = {}
    else:
        checks.append(GateCheck("candidate_metrics_shape", "PASS", "metrics is a mapping"))
    combined = _as_mapping(metrics.get("combined"))
    retry_block = _as_mapping(metrics.get("retry_block"))
    required_metric_variants = ["kl_lss_sare", "kl_lss_single_expert"]
    missing_metric_variants = [
        variant
        for variant in required_metric_variants
        if combined is None or retry_block is None or variant not in retry_block or variant not in combined
    ]
    if combined is None:
        checks.append(GateCheck("candidate_metrics_combined_shape", "INCONCLUSIVE", "metrics.combined must be a mapping"))
    else:
        checks.append(GateCheck("candidate_metrics_combined_shape", "PASS", "metrics.combined is a mapping"))
    if retry_block is None:
        checks.append(GateCheck("candidate_metrics_retry_block_shape", "INCONCLUSIVE", "metrics.retry_block must be a mapping"))
    else:
        checks.append(GateCheck("candidate_metrics_retry_block_shape", "PASS", "metrics.retry_block is a mapping"))
    if missing_metric_variants:
        checks.append(
            GateCheck(
                "candidate_metrics",
                "INCONCLUSIVE",
                "missing candidate metric variants: " + ", ".join(f"`{variant}`" for variant in sorted(set(missing_metric_variants))),
            )
        )
    else:
        checks.append(GateCheck("candidate_metrics", "PASS", "candidate exposes retry-block and combined metrics for required variants"))

    if not missing_metric_variants:
        retry_sare = _as_mapping(retry_block["kl_lss_sare"])
        retry_single = _as_mapping(retry_block["kl_lss_single_expert"])
        combined_sare = _as_mapping(combined["kl_lss_sare"])
        if retry_sare is None or retry_single is None or combined_sare is None:
            checks.append(
                GateCheck(
                    "candidate_metric_payload_shape",
                    "INCONCLUSIVE",
                    "required metric payloads must be mappings for `kl_lss_sare` and `kl_lss_single_expert`",
                )
            )
        else:
            candidate_retry_sare_mean = retry_sare.get("mean")
            candidate_retry_single_mean = retry_single.get("mean")
            candidate_retry_failures = retry_sare.get("complete_seed_failures")
            candidate_combined_sare_mean = combined_sare.get("mean")
            candidate_combined_failures = combined_sare.get("complete_seed_failures")
            numeric_values = {
                "retry_block_improvement": candidate_retry_sare_mean,
                "retry_block_vs_single_expert": candidate_retry_single_mean,
                "retry_block_failures": candidate_retry_failures,
                "combined_picture_mean": candidate_combined_sare_mean,
                "combined_picture_failures": candidate_combined_failures,
            }
            invalid_fields = [name for name, value in numeric_values.items() if _coerce_int(value) is None and not isinstance(value, (int, float))]
            invalid_fields.extend(
                name
                for name, value in {
                    "retry_block_improvement": candidate_retry_sare_mean,
                    "retry_block_vs_single_expert": candidate_retry_single_mean,
                    "combined_picture_mean": candidate_combined_sare_mean,
                }.items()
                if _coerce_int(value) is None and not isinstance(value, float)
            )
            if invalid_fields:
                checks.append(
                    GateCheck(
                        "candidate_metric_values",
                        "INCONCLUSIVE",
                        "required metric values are not numeric: " + ", ".join(f"`{field}`" for field in sorted(set(invalid_fields))),
                    )
                )
            else:
                candidate_retry_sare_mean = float(candidate_retry_sare_mean)
                candidate_retry_single_mean = float(candidate_retry_single_mean)
                candidate_retry_failures = int(candidate_retry_failures)
                candidate_combined_sare_mean = float(candidate_combined_sare_mean)
                candidate_combined_failures = int(candidate_combined_failures)
        frozen_retry_sare_mean = float(thaw_gate.get("retry_block", {}).get("candidate_sare_mean_must_exceed", 0.0))
        max_retry_failures = int(thaw_gate.get("retry_block", {}).get("candidate_sare_complete_seed_failures_must_be_lte", 0))
        if not any(check.name == "candidate_metric_values" and check.status == "INCONCLUSIVE" for check in checks) and not any(
            check.name == "candidate_metric_payload_shape" and check.status == "INCONCLUSIVE" for check in checks
        ) and candidate_retry_sare_mean > frozen_retry_sare_mean:
            checks.append(
                GateCheck(
                    "retry_block_improvement",
                    "PASS",
                    f"candidate retry-block SARE mean `{candidate_retry_sare_mean:.4f}` exceeds frozen baseline `{frozen_retry_sare_mean:.4f}`",
                )
            )
        elif not any(check.name == "candidate_metric_values" and check.status == "INCONCLUSIVE" for check in checks) and not any(
            check.name == "candidate_metric_payload_shape" and check.status == "INCONCLUSIVE" for check in checks
        ):
            checks.append(
                GateCheck(
                    "retry_block_improvement",
                    "FAIL",
                    f"candidate retry-block SARE mean `{candidate_retry_sare_mean:.4f}` does not exceed frozen baseline `{frozen_retry_sare_mean:.4f}`",
                )
            )
        if not any(check.name == "candidate_metric_values" and check.status == "INCONCLUSIVE" for check in checks) and not any(
            check.name == "candidate_metric_payload_shape" and check.status == "INCONCLUSIVE" for check in checks
        ) and candidate_retry_sare_mean >= candidate_retry_single_mean:
            checks.append(
                GateCheck(
                    "retry_block_vs_single_expert",
                    "PASS",
                    f"candidate retry-block SARE mean `{candidate_retry_sare_mean:.4f}` matches or beats same-block single_expert `{candidate_retry_single_mean:.4f}`",
                )
            )
        elif not any(check.name == "candidate_metric_values" and check.status == "INCONCLUSIVE" for check in checks) and not any(
            check.name == "candidate_metric_payload_shape" and check.status == "INCONCLUSIVE" for check in checks
        ):
            checks.append(
                GateCheck(
                    "retry_block_vs_single_expert",
                    "FAIL",
                    f"candidate retry-block SARE mean `{candidate_retry_sare_mean:.4f}` trails same-block single_expert `{candidate_retry_single_mean:.4f}`",
                )
            )
        if not any(check.name == "candidate_metric_values" and check.status == "INCONCLUSIVE" for check in checks) and not any(
            check.name == "candidate_metric_payload_shape" and check.status == "INCONCLUSIVE" for check in checks
        ) and candidate_retry_failures <= max_retry_failures:
            checks.append(
                GateCheck(
                    "retry_block_failures",
                    "PASS",
                    f"candidate retry-block SARE complete-seed failures `{candidate_retry_failures}` stay within gate `{max_retry_failures}`",
                )
            )
        elif not any(check.name == "candidate_metric_values" and check.status == "INCONCLUSIVE" for check in checks) and not any(
            check.name == "candidate_metric_payload_shape" and check.status == "INCONCLUSIVE" for check in checks
        ):
            checks.append(
                GateCheck(
                    "retry_block_failures",
                    "FAIL",
                    f"candidate retry-block SARE complete-seed failures `{candidate_retry_failures}` exceed gate `{max_retry_failures}`",
                )
            )
        min_combined_mean = float(thaw_gate.get("combined_picture", {}).get("candidate_sare_mean_must_be_gte", 0.0))
        max_combined_failures = int(thaw_gate.get("combined_picture", {}).get("candidate_sare_complete_seed_failures_must_be_lte", 0))
        if not any(check.name == "candidate_metric_values" and check.status == "INCONCLUSIVE" for check in checks) and not any(
            check.name == "candidate_metric_payload_shape" and check.status == "INCONCLUSIVE" for check in checks
        ) and candidate_combined_sare_mean >= min_combined_mean:
            checks.append(
                GateCheck(
                    "combined_picture_mean",
                    "PASS",
                    f"candidate combined SARE mean `{candidate_combined_sare_mean:.4f}` preserves or improves frozen baseline `{min_combined_mean:.4f}`",
                )
            )
        elif not any(check.name == "candidate_metric_values" and check.status == "INCONCLUSIVE" for check in checks) and not any(
            check.name == "candidate_metric_payload_shape" and check.status == "INCONCLUSIVE" for check in checks
        ):
            checks.append(
                GateCheck(
                    "combined_picture_mean",
                    "FAIL",
                    f"candidate combined SARE mean `{candidate_combined_sare_mean:.4f}` regresses below frozen baseline `{min_combined_mean:.4f}`",
                )
            )
        if not any(check.name == "candidate_metric_values" and check.status == "INCONCLUSIVE" for check in checks) and not any(
            check.name == "candidate_metric_payload_shape" and check.status == "INCONCLUSIVE" for check in checks
        ) and candidate_combined_failures <= max_combined_failures:
            checks.append(
                GateCheck(
                    "combined_picture_failures",
                    "PASS",
                    f"candidate combined SARE complete-seed failures `{candidate_combined_failures}` stay within gate `{max_combined_failures}`",
                )
            )
        elif not any(check.name == "candidate_metric_values" and check.status == "INCONCLUSIVE" for check in checks) and not any(
            check.name == "candidate_metric_payload_shape" and check.status == "INCONCLUSIVE" for check in checks
        ):
            checks.append(
                GateCheck(
                    "combined_picture_failures",
                    "FAIL",
                    f"candidate combined SARE complete-seed failures `{candidate_combined_failures}` exceed gate `{max_combined_failures}`",
                )
            )

    claim_scope_failed = any(check.name == "claim_scope" and check.status == "FAIL" for check in checks)
    if claim_scope_failed:
        verdict = "FAIL: claim remains frozen"
    elif any(check.status == "INCONCLUSIVE" for check in checks):
        verdict = "INCONCLUSIVE: missing prerequisites"
    elif any(check.status == "FAIL" for check in checks):
        verdict = "FAIL: claim remains frozen"
    else:
        verdict = "PASS: thaw consideration allowed"
    return verdict, checks


def _manifest_from_frozen_pack(frozen_pack: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim": frozen_pack.get("claim", {}),
        "evaluation": frozen_pack.get("evaluation", {}),
        "thaw_gate": frozen_pack.get("thaw_gate", {}),
    }


def evaluate_pack_claim_gate(
    frozen_pack_path: Path,
    frozen_pack: dict[str, Any],
    candidate_pack_path: Path,
    candidate_pack: dict[str, Any],
) -> tuple[str, list[GateCheck], list[dict[str, str]], list[dict[str, str]]]:
    frozen_validation_verdict, frozen_checks = validate_frozen_benchmark_pack(frozen_pack)
    candidate_validation_verdict, candidate_checks = validate_candidate_result_pack(frozen_pack, frozen_pack_path, candidate_pack)

    frozen_pack_rows = [asdict(check) for check in frozen_checks]
    candidate_pack_rows = [asdict(check) for check in candidate_checks]
    manifest_like = _manifest_from_frozen_pack(frozen_pack)
    scope_verdict, scope_checks = evaluate_claim_gate(
        manifest_like,
        {
            "task": candidate_pack.get("task"),
            "evaluation": candidate_pack.get("evaluation"),
            "requested_claims": candidate_pack.get("requested_claims", []),
            "controls_present": candidate_pack.get("controls_present", []),
            "metrics": candidate_pack.get("metrics", {}),
        },
    )
    claim_scope_failed = any(check.name == "claim_scope" and check.status == "FAIL" for check in scope_checks)

    if frozen_validation_verdict != "PASS: frozen benchmark pack validated" or candidate_validation_verdict != "PASS: candidate result pack validated":
        mapped_checks = [
            GateCheck(
                f"frozen_pack::{check['name']}",
                "INCONCLUSIVE" if check["status"] != "PASS" else "PASS",
                check["detail"],
            )
            for check in frozen_pack_rows
        ]
        mapped_checks.extend(
            GateCheck(
                f"candidate_pack::{check['name']}",
                "INCONCLUSIVE" if check["status"] != "PASS" else "PASS",
                check["detail"],
            )
            for check in candidate_pack_rows
        )
        if claim_scope_failed:
            mapped_checks.extend(check for check in scope_checks if check.name == "claim_scope")
            return "FAIL: claim remains frozen", mapped_checks, frozen_pack_rows, candidate_pack_rows
        return "INCONCLUSIVE: missing prerequisites", mapped_checks, frozen_pack_rows, candidate_pack_rows

    verdict, checks = evaluate_claim_gate(manifest_like, candidate_pack)
    return verdict, checks, frozen_pack_rows, candidate_pack_rows


def evaluate_pack_claim_gate_from_paths(
    frozen_pack_path: Path,
    candidate_pack_path: Path,
) -> tuple[str, list[GateCheck], list[dict[str, str]], list[dict[str, str]]]:
    frozen_pack, frozen_error = safe_load_structured_file(frozen_pack_path)
    if frozen_error is not None or frozen_pack is None:
        checks = [GateCheck("frozen_pack_load", "INCONCLUSIVE", f"failed to load frozen pack `{frozen_pack_path}`: {frozen_error}")]
        return "INCONCLUSIVE: missing prerequisites", checks, [], []
    candidate_pack, candidate_error = safe_load_structured_file(candidate_pack_path)
    if candidate_error is not None or candidate_pack is None:
        checks = [GateCheck("candidate_pack_load", "INCONCLUSIVE", f"failed to load candidate pack `{candidate_pack_path}`: {candidate_error}")]
        frozen_verdict, frozen_checks = validate_frozen_benchmark_pack(frozen_pack)
        frozen_rows = [asdict(check) for check in frozen_checks]
        if frozen_verdict != "PASS: frozen benchmark pack validated":
            checks = [
                GateCheck(
                    f"frozen_pack::{check['name']}",
                    "INCONCLUSIVE" if check["status"] != "PASS" else "PASS",
                    check["detail"],
                )
                for check in frozen_rows
            ] + checks
        return "INCONCLUSIVE: missing prerequisites", checks, frozen_rows, []
    return evaluate_pack_claim_gate(frozen_pack_path, frozen_pack, candidate_pack_path, candidate_pack)


def render_gate_report(
    manifest_path: Path,
    candidate_path: Path,
    manifest: dict[str, Any],
    verdict: str,
    checks: list[GateCheck],
) -> str:
    lines = [
        "# Claim Gate Dry Run",
        "",
        f"- manifest: `{manifest_path}`",
        f"- candidate: `{candidate_path}`",
        f"- current frozen status: `{manifest.get('claim', {}).get('status', 'unknown')}`",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for check in checks:
        lines.append(f"| {check.name} | `{check.status}` | {check.detail} |")
    lines.extend(["", "## Verdict", "", verdict])
    return "\n".join(lines) + "\n"


def render_pack_gate_report(
    frozen_pack_path: Path,
    candidate_pack_path: Path,
    verdict: str,
    frozen_pack_checks: list[dict[str, str]],
    candidate_pack_checks: list[dict[str, str]],
    checks: list[GateCheck],
) -> str:
    lines = [
        "# Pack-Based Claim Gate Dry Run",
        "",
        f"- frozen benchmark pack: `{frozen_pack_path}`",
        f"- candidate result pack: `{candidate_pack_path}`",
        "",
        "## Frozen Pack Validation",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for check in frozen_pack_checks:
        lines.append(f"| {check['name']} | `{check['status']}` | {check['detail']} |")
    lines.extend(
        [
            "",
            "## Candidate Pack Validation",
            "",
            "| Check | Result | Detail |",
            "| --- | --- | --- |",
        ]
    )
    for check in candidate_pack_checks:
        lines.append(f"| {check['name']} | `{check['status']}` | {check['detail']} |")
    lines.extend(
        [
            "",
            "## Claim Gate Checks",
            "",
            "| Check | Result | Detail |",
            "| --- | --- | --- |",
        ]
    )
    for check in checks:
        lines.append(f"| {check.name} | `{check.status}` | {check.detail} |")
    lines.extend(["", "## Verdict", "", verdict])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automated claim gate for the frozen DoorKey SARE claim.")
    parser.add_argument("--manifest")
    parser.add_argument("--candidate")
    parser.add_argument("--frozen-pack")
    parser.add_argument("--candidate-pack")
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    json_output_path = Path(args.json_output) if args.json_output else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if json_output_path is not None:
        json_output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.frozen_pack or args.candidate_pack:
        if not args.frozen_pack or not args.candidate_pack:
            raise SystemExit("pack mode requires both --frozen-pack and --candidate-pack")
        frozen_pack_path = Path(args.frozen_pack)
        candidate_pack_path = Path(args.candidate_pack)
        verdict, checks, frozen_pack_checks, candidate_pack_checks = evaluate_pack_claim_gate_from_paths(
            frozen_pack_path,
            candidate_pack_path,
        )
        output_path.write_text(
            render_pack_gate_report(
                frozen_pack_path,
                candidate_pack_path,
                verdict,
                frozen_pack_checks,
                candidate_pack_checks,
                checks,
            ),
            encoding="utf-8",
        )
        if json_output_path is not None:
            json_output_path.write_text(
                json.dumps(
                    {
                        "mode": "pack",
                        "frozen_pack": str(frozen_pack_path),
                        "candidate_pack": str(candidate_pack_path),
                        "verdict": verdict,
                        "frozen_pack_validation": frozen_pack_checks,
                        "candidate_pack_validation": candidate_pack_checks,
                        "checks": [asdict(check) for check in checks],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return

    if not args.manifest or not args.candidate:
        raise SystemExit("legacy mode requires both --manifest and --candidate")
    manifest_path = Path(args.manifest)
    candidate_path = Path(args.candidate)
    manifest = load_manifest(manifest_path)
    candidate = load_candidate(candidate_path)
    verdict, checks = evaluate_claim_gate(manifest, candidate)
    output_path.write_text(
        render_gate_report(manifest_path, candidate_path, manifest, verdict, checks),
        encoding="utf-8",
    )
    if json_output_path is not None:
        json_output_path.write_text(
            json.dumps(
                {
                    "mode": "legacy",
                    "manifest": str(manifest_path),
                    "candidate": str(candidate_path),
                    "verdict": verdict,
                    "checks": [asdict(check) for check in checks],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
