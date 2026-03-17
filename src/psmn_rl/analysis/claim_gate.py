from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class GateCheck:
    name: str
    status: str
    detail: str


def load_manifest(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_candidate(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raise ValueError(f"Unsupported candidate format: {path}")


def evaluate_claim_gate(manifest: dict[str, Any], candidate: dict[str, Any]) -> tuple[str, list[GateCheck]]:
    checks: list[GateCheck] = []
    thaw_gate = manifest.get("thaw_gate", {})
    required_eval = thaw_gate.get("required_evaluation", {})
    expected_task = str(required_eval.get("task", "DoorKey"))
    expected_path = str(required_eval.get("path_key", "external_policy_diagnostics"))
    expected_episodes = int(required_eval.get("episodes", 64))

    task = str(candidate.get("task", ""))
    evaluation = candidate.get("evaluation", {})
    if task != expected_task:
        checks.append(GateCheck("task", "INCONCLUSIVE", f"candidate task `{task or '-'}` does not match required `{expected_task}`"))
    else:
        checks.append(GateCheck("task", "PASS", f"candidate task matches `{expected_task}`"))

    candidate_path = str(evaluation.get("path_key", ""))
    candidate_episodes = int(evaluation.get("episodes", 0))
    if candidate_path != expected_path or candidate_episodes != expected_episodes:
        checks.append(
            GateCheck(
                "evaluation_path",
                "INCONCLUSIVE",
                f"candidate path `{candidate_path or '-'}` / episodes `{candidate_episodes}` do not match required `{expected_path}` / `{expected_episodes}`",
            )
        )
    else:
        checks.append(GateCheck("evaluation_path", "PASS", f"candidate uses `{expected_path}` with `{expected_episodes}` episodes"))

    controls_present = set(candidate.get("controls_present", []))
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

    requested_claims = set(candidate.get("requested_claims", []))
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

    metrics = candidate.get("metrics", {})
    combined = metrics.get("combined", {})
    retry_block = metrics.get("retry_block", {})
    required_metric_variants = ["kl_lss_sare", "kl_lss_single_expert"]
    missing_metric_variants = [
        variant
        for variant in required_metric_variants
        if variant not in retry_block or variant not in combined
    ]
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
        candidate_retry_sare_mean = float(retry_block["kl_lss_sare"]["mean"])
        candidate_retry_single_mean = float(retry_block["kl_lss_single_expert"]["mean"])
        candidate_retry_failures = int(retry_block["kl_lss_sare"]["complete_seed_failures"])
        frozen_retry_sare_mean = float(thaw_gate.get("retry_block", {}).get("candidate_sare_mean_must_exceed", 0.0))
        max_retry_failures = int(thaw_gate.get("retry_block", {}).get("candidate_sare_complete_seed_failures_must_be_lte", 0))
        if candidate_retry_sare_mean > frozen_retry_sare_mean:
            checks.append(
                GateCheck(
                    "retry_block_improvement",
                    "PASS",
                    f"candidate retry-block SARE mean `{candidate_retry_sare_mean:.4f}` exceeds frozen baseline `{frozen_retry_sare_mean:.4f}`",
                )
            )
        else:
            checks.append(
                GateCheck(
                    "retry_block_improvement",
                    "FAIL",
                    f"candidate retry-block SARE mean `{candidate_retry_sare_mean:.4f}` does not exceed frozen baseline `{frozen_retry_sare_mean:.4f}`",
                )
            )
        if candidate_retry_sare_mean >= candidate_retry_single_mean:
            checks.append(
                GateCheck(
                    "retry_block_vs_single_expert",
                    "PASS",
                    f"candidate retry-block SARE mean `{candidate_retry_sare_mean:.4f}` matches or beats same-block single_expert `{candidate_retry_single_mean:.4f}`",
                )
            )
        else:
            checks.append(
                GateCheck(
                    "retry_block_vs_single_expert",
                    "FAIL",
                    f"candidate retry-block SARE mean `{candidate_retry_sare_mean:.4f}` trails same-block single_expert `{candidate_retry_single_mean:.4f}`",
                )
            )
        if candidate_retry_failures <= max_retry_failures:
            checks.append(
                GateCheck(
                    "retry_block_failures",
                    "PASS",
                    f"candidate retry-block SARE complete-seed failures `{candidate_retry_failures}` stay within gate `{max_retry_failures}`",
                )
            )
        else:
            checks.append(
                GateCheck(
                    "retry_block_failures",
                    "FAIL",
                    f"candidate retry-block SARE complete-seed failures `{candidate_retry_failures}` exceed gate `{max_retry_failures}`",
                )
            )

        candidate_combined_sare_mean = float(combined["kl_lss_sare"]["mean"])
        candidate_combined_failures = int(combined["kl_lss_sare"]["complete_seed_failures"])
        min_combined_mean = float(thaw_gate.get("combined_picture", {}).get("candidate_sare_mean_must_be_gte", 0.0))
        max_combined_failures = int(thaw_gate.get("combined_picture", {}).get("candidate_sare_complete_seed_failures_must_be_lte", 0))
        if candidate_combined_sare_mean >= min_combined_mean:
            checks.append(
                GateCheck(
                    "combined_picture_mean",
                    "PASS",
                    f"candidate combined SARE mean `{candidate_combined_sare_mean:.4f}` preserves or improves frozen baseline `{min_combined_mean:.4f}`",
                )
            )
        else:
            checks.append(
                GateCheck(
                    "combined_picture_mean",
                    "FAIL",
                    f"candidate combined SARE mean `{candidate_combined_sare_mean:.4f}` regresses below frozen baseline `{min_combined_mean:.4f}`",
                )
            )
        if candidate_combined_failures <= max_combined_failures:
            checks.append(
                GateCheck(
                    "combined_picture_failures",
                    "PASS",
                    f"candidate combined SARE complete-seed failures `{candidate_combined_failures}` stay within gate `{max_combined_failures}`",
                )
            )
        else:
            checks.append(
                GateCheck(
                    "combined_picture_failures",
                    "FAIL",
                    f"candidate combined SARE complete-seed failures `{candidate_combined_failures}` exceed gate `{max_combined_failures}`",
                )
            )

    if any(check.status == "INCONCLUSIVE" for check in checks):
        verdict = "INCONCLUSIVE: missing prerequisites"
    elif any(check.status == "FAIL" for check in checks):
        verdict = "FAIL: claim remains frozen"
    else:
        verdict = "PASS: thaw consideration allowed"
    return verdict, checks


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automated claim gate for the frozen DoorKey SARE claim.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest)
    candidate_path = Path(args.candidate)
    output_path = Path(args.output)
    json_output_path = Path(args.json_output) if args.json_output else None

    manifest = load_manifest(manifest_path)
    candidate = load_candidate(candidate_path)
    verdict, checks = evaluate_claim_gate(manifest, candidate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_gate_report(manifest_path, candidate_path, manifest, verdict, checks),
        encoding="utf-8",
    )
    if json_output_path is not None:
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(
            json.dumps(
                {
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
