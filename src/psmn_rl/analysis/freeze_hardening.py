from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from psmn_rl.analysis.claim_gate import load_manifest
from psmn_rl.utils.io import get_git_commit, get_git_dirty


DISPLAY_NAMES = {
    "recovered_token_dense": "recovered token_dense",
    "kl_lss_token_dense": "KL learner-state token_dense",
    "kl_lss_single_expert": "KL learner-state single_expert",
    "baseline_sare": "baseline PPO SARE",
    "kl_lss_sare": "KL learner-state SARE",
}

LEDGER_ROWS = [
    {
        "family": "PPO tuning / checkpoint dynamics",
        "status": "negative",
        "scope": "PPO-only DoorKey SARE greedy recovery",
        "claim_effect": "did not recover a greedy PPO SARE policy",
        "reports": ["outputs/reports/checkpoint_dynamics_report.md"],
    },
    {
        "family": "Entropy schedules",
        "status": "negative",
        "scope": "PPO-only DoorKey SARE greedy recovery",
        "claim_effect": "did not recover a greedy PPO SARE policy",
        "reports": ["outputs/reports/entropy_schedule_report.md"],
    },
    {
        "family": "Self-imitation",
        "status": "negative",
        "scope": "DoorKey SARE greedy recovery from successful sampled trajectories",
        "claim_effect": "did not recover a greedy SARE policy",
        "reports": ["outputs/reports/self_imitation_report.md"],
    },
    {
        "family": "Offline teacher distillation",
        "status": "negative",
        "scope": "teacher-guided offline extraction for token_dense and SARE on DoorKey",
        "claim_effect": "did not recover greedy SARE or robust tokenized controls",
        "reports": ["outputs/reports/policy_distillation_report.md"],
    },
    {
        "family": "Learner-state supervision",
        "status": "bounded positive",
        "scope": "DoorKey teacher-guided KL learner-state extraction",
        "claim_effect": "opened the only bounded positive routed DoorKey result",
        "reports": [
            "outputs/reports/lss_robustness_sweep_report.md",
            "outputs/reports/lss_robustness_multiseed_report.md",
        ],
    },
    {
        "family": "Claim hardening",
        "status": "bounded positive",
        "scope": "DoorKey-only replication and matched tokenized controls",
        "claim_effect": "strengthened the DoorKey-only extraction result without broadening it beyond teacher-guided scope",
        "reports": [
            "outputs/reports/lss_additional_seed_report.md",
            "outputs/reports/lss_matched_control_report.md",
            "outputs/reports/lss_claim_hardening_decision_memo.md",
        ],
    },
    {
        "family": "Claim broadening",
        "status": "bounded positive",
        "scope": "DoorKey-only fairness and causal-route checks before final freeze",
        "claim_effect": "showed a stronger within-DoorKey edge, later weakened by the final block",
        "reports": [
            "outputs/reports/lss_single_expert_matched_control_report.md",
            "outputs/reports/lss_extended_route_dependence_report.md",
            "outputs/reports/lss_claim_broadening_decision_memo.md",
        ],
    },
    {
        "family": "Resume gate",
        "status": "frozen",
        "scope": "DoorKey-only retry justification check",
        "claim_effect": "did not justify a bounded retry",
        "reports": [
            "outputs/reports/lss_resume_gate_failure_mechanism_report.md",
            "outputs/reports/lss_resume_gate_decision_memo.md",
        ],
    },
    {
        "family": "Forensic atlas",
        "status": "frozen",
        "scope": "deep DoorKey-only trajectory, round, and route-locality diagnosis",
        "claim_effect": "confirmed mixed weak-block failure modes and blocked bounded retry",
        "reports": [
            "outputs/reports/lss_forensic_casebook.md",
            "outputs/reports/lss_forensic_round_audit.md",
            "outputs/reports/lss_forensic_route_locality.md",
            "outputs/reports/lss_forensic_atlas_decision_memo.md",
        ],
    },
    {
        "family": "Final frozen state",
        "status": "frozen",
        "scope": "current DoorKey teacher-guided claim envelope",
        "claim_effect": "bounded teacher-guided DoorKey SARE result only; no PPO-only, multi-expert, cross-task, or KeyCorridor claim",
        "reports": [
            "outputs/reports/frozen_claim_envelope.md",
            "outputs/reports/frozen_claim_manifest_report.md",
            "outputs/reports/freeze_hardening_decision_memo.md",
        ],
    },
]


@dataclass(slots=True)
class ValidationRow:
    section: str
    metric: str
    variant: str
    expected: str
    actual: str
    status: str


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_greedy_rows(path: Path) -> list[dict[str, str]]:
    rows = _read_csv_rows(path)
    if rows and "mode" in rows[0]:
        rows = [row for row in rows if row.get("mode") == "greedy"]
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _variant_summary(rows: list[dict[str, str]], variants: list[str]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for variant in variants:
        variant_rows = [row for row in rows if row.get("label") == variant]
        rates = [float(row["eval_success_rate"]) for row in variant_rows]
        summary[variant] = {
            "mean": _mean(rates),
            "min": min(rates) if rates else 0.0,
            "max": max(rates) if rates else 0.0,
            "complete_seed_failures": sum(1 for rate in rates if rate == 0.0),
            "seed_count": len(rates),
        }
    return summary


def _lane_seed_set(rows: list[dict[str, str]]) -> set[tuple[str, int]]:
    return {(str(row.get("lane", "")), int(row["seed"])) for row in rows}


def _load_baseline_candidate(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    reports = manifest.get("authoritative_reports", {})
    combined_csv = Path(reports["combined_doorkey_csv"])
    final_block_csv = Path(reports["final_block_csv"])
    keycorridor_csv = Path(reports["keycorridor_transfer_csv"])
    missing = [str(path) for path in (combined_csv, final_block_csv, keycorridor_csv) if not path.exists()]
    variants = list(manifest.get("variants", {}).keys())
    if missing:
        return {}, missing

    combined_rows = _read_greedy_rows(combined_csv)
    final_block_rows = _read_greedy_rows(final_block_csv)
    keycorridor_rows = _read_greedy_rows(keycorridor_csv)

    candidate = {
        "schema_version": 1,
        "candidate_name": "current_frozen_baseline",
        "task": manifest.get("evaluation", {}).get("task", "DoorKey"),
        "evaluation": {
            "path_key": manifest.get("evaluation", {}).get("path_key", "external_policy_diagnostics"),
            "episodes": int(manifest.get("evaluation", {}).get("episodes", 64)),
        },
        "requested_claims": [manifest.get("claim", {}).get("allowed_claim_key", "bounded_teacher_guided_doorkey_sare")],
        "controls_present": variants,
        "provenance": {
            "combined_doorkey_csv": str(combined_csv),
            "final_block_csv": str(final_block_csv),
            "keycorridor_transfer_csv": str(keycorridor_csv),
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
        },
        "metrics": {
            "combined": _variant_summary(combined_rows, variants),
            "retry_block": _variant_summary(final_block_rows, variants),
            "keycorridor_transfer": _variant_summary(
                keycorridor_rows,
                ["recovered_token_dense", "baseline_sare", "kl_lss_sare"],
            ),
        },
        "actual_sets": {
            "combined_lane_seeds": sorted([list(item) for item in _lane_seed_set(combined_rows)]),
            "retry_block_lane_seeds": sorted([list(item) for item in _lane_seed_set(final_block_rows)]),
        },
    }
    return candidate, []


def _expected_lane_seed_set(manifest: dict[str, Any], key: str) -> set[tuple[str, int]]:
    group = manifest.get("seed_groups", {}).get(key, {})
    blocks = group.get("blocks")
    if blocks:
        return {
            (str(block["lane"]), int(seed))
            for block in blocks
            for seed in block.get("seeds", [])
        }
    lanes = group.get("lanes")
    seeds = group.get("seeds", [])
    if lanes is None:
        lanes = [group.get("lane")]
    return {(str(lane), int(seed)) for lane in lanes for seed in seeds}


def _candidate_validation_rows(manifest: dict[str, Any], candidate: dict[str, Any]) -> tuple[list[ValidationRow], str]:
    rows: list[ValidationRow] = []
    thresholds = manifest.get("thresholds", {})
    tolerance = 1e-4

    def add(section: str, metric: str, variant: str, expected: float | int, actual: float | int, ok: bool) -> None:
        rows.append(
            ValidationRow(
                section=section,
                metric=metric,
                variant=variant,
                expected=str(expected if isinstance(expected, int) else _format_float(float(expected))),
                actual=str(actual if isinstance(actual, int) else _format_float(float(actual))),
                status="PASS" if ok else "FAIL",
            )
        )

    actual_combined = candidate["metrics"]["combined"]
    actual_retry = candidate["metrics"]["retry_block"]
    actual_transfer = candidate["metrics"]["keycorridor_transfer"]
    for variant, expected_mean in thresholds.get("combined_means", {}).items():
        actual_mean = float(actual_combined[variant]["mean"])
        add("combined", "mean", variant, float(expected_mean), actual_mean, abs(actual_mean - float(expected_mean)) <= tolerance)
    for variant, expected_failures in thresholds.get("combined_complete_seed_failures", {}).items():
        actual_failures = int(actual_combined[variant]["complete_seed_failures"])
        add("combined", "complete_seed_failures", variant, int(expected_failures), actual_failures, actual_failures == int(expected_failures))
    for variant, expected_mean in thresholds.get("retry_block_means", {}).items():
        actual_mean = float(actual_retry[variant]["mean"])
        add("retry_block", "mean", variant, float(expected_mean), actual_mean, abs(actual_mean - float(expected_mean)) <= tolerance)
    for variant, expected_failures in thresholds.get("retry_block_complete_seed_failures", {}).items():
        actual_failures = int(actual_retry[variant]["complete_seed_failures"])
        add("retry_block", "complete_seed_failures", variant, int(expected_failures), actual_failures, actual_failures == int(expected_failures))
    for variant, expected_mean in thresholds.get("keycorridor_transfer_means", {}).items():
        actual_mean = float(actual_transfer[variant]["mean"])
        add("keycorridor_transfer", "mean", variant, float(expected_mean), actual_mean, abs(actual_mean - float(expected_mean)) <= tolerance)

    actual_combined_set = {tuple(item) for item in candidate.get("actual_sets", {}).get("combined_lane_seeds", [])}
    expected_combined_set = _expected_lane_seed_set(manifest, "combined")
    rows.append(
        ValidationRow(
            section="coverage",
            metric="combined_lane_seed_set",
            variant="-",
            expected=str(sorted(expected_combined_set)),
            actual=str(sorted(actual_combined_set)),
            status="PASS" if actual_combined_set == expected_combined_set else "FAIL",
        )
    )
    actual_retry_set = {tuple(item) for item in candidate.get("actual_sets", {}).get("retry_block_lane_seeds", [])}
    expected_retry_set = _expected_lane_seed_set(manifest, "retry_block")
    rows.append(
        ValidationRow(
            section="coverage",
            metric="retry_block_lane_seed_set",
            variant="-",
            expected=str(sorted(expected_retry_set)),
            actual=str(sorted(actual_retry_set)),
            status="PASS" if actual_retry_set == expected_retry_set else "FAIL",
        )
    )

    verdict = "PASS: frozen baseline validated" if all(row.status == "PASS" for row in rows) else "FAIL: baseline artifacts drifted"
    return rows, verdict


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_envelope_report(manifest: dict[str, Any], manifest_path: Path) -> str:
    thaw = manifest.get("thaw_gate", {})
    retry_block = thaw.get("retry_block", {})
    combined = thaw.get("combined_picture", {})
    lines = [
        "# Frozen Claim Envelope",
        "",
        f"- manifest: `{manifest_path}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
        "## Allowed Claim",
        "",
        f"- `{manifest['claim']['allowed_claim_text']}`",
        "",
        "## Not Allowed",
        "",
    ]
    for item in manifest["claim"]["disallowed_claim_keys"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Future Thaw Bar",
            "",
            f"- Any future DoorKey retry must use the `{manifest['evaluation']['path_text']}`.",
            f"- On seeds `{retry_block['seeds']}`, candidate KL learner-state `SARE` must beat the frozen retry-block mean `{retry_block['candidate_sare_mean_must_exceed']:.4f}`.",
            "- On that same block, candidate KL learner-state `SARE` must at least match the matched KL learner-state `single_expert` result.",
            f"- The candidate must not worsen the combined DoorKey KL learner-state `SARE` mean `{combined['candidate_sare_mean_must_be_gte']:.4f}`.",
            "",
            "## Operational Rule",
            "",
            "- No future DoorKey result should be treated as a thaw candidate until it passes the automated claim gate against this manifest.",
            "- The sealed frozen benchmark pack and pack-based claim gate are the canonical operational entrypoints for future work.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_manifest_report(manifest: dict[str, Any], manifest_path: Path) -> str:
    reports = manifest.get("authoritative_reports", {})
    lines = [
        "# Frozen Claim Manifest Report",
        "",
        f"- manifest: `{manifest_path}`",
        f"- canonical method: teacher `{manifest['canonical_method']['teacher']}` + `{manifest['canonical_method']['supervision']}` + `{manifest['canonical_method']['learner_state_collection']}`",
        f"- evaluation path: `{manifest['evaluation']['path_text']}`",
        "",
        "## Canonical Variants",
        "",
        "| Variant Key | Display | Family |",
        "| --- | --- | --- |",
    ]
    for key, payload in manifest.get("variants", {}).items():
        lines.append(f"| `{key}` | {payload['display']} | `{payload['family']}` |")
    lines.extend(
        [
            "",
            "## Canonical Seed Groups",
            "",
            f"- `structured_slice`: `{manifest['seed_groups']['structured_slice']['seeds']}`",
            f"- `retry_block`: `{manifest['seed_groups']['retry_block']['seeds']}`",
            f"- `combined`: `{[(block['lane'], block['seeds']) for block in manifest['seed_groups']['combined']['blocks']]}`",
            "",
            "## Frozen Thresholds",
            "",
            f"- combined KL learner-state `SARE` mean: `{manifest['thresholds']['combined_means']['kl_lss_sare']:.4f}`",
            f"- combined KL learner-state `single_expert` mean: `{manifest['thresholds']['combined_means']['kl_lss_single_expert']:.4f}`",
            f"- retry-block KL learner-state `SARE` mean: `{manifest['thresholds']['retry_block_means']['kl_lss_sare']:.4f}`",
            f"- retry-block KL learner-state `single_expert` mean: `{manifest['thresholds']['retry_block_means']['kl_lss_single_expert']:.4f}`",
            "",
            "## Pack Schema",
            "",
            f"- frozen benchmark pack schema version: `{manifest.get('benchmark_pack', {}).get('schema_version', 1)}`",
            f"- candidate result pack schema version: `{manifest.get('candidate_pack', {}).get('schema_version', 1)}`",
            f"- required candidate artifact roles: `{manifest.get('candidate_pack', {}).get('required_artifact_roles', [])}`",
            "",
            "## Authoritative Reports",
            "",
        ]
    )
    for key, path in reports.items():
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def _build_validation_report(
    manifest_path: Path,
    manifest: dict[str, Any],
    candidate: dict[str, Any] | None,
    validation_rows: list[ValidationRow],
    verdict: str,
    missing: list[str],
) -> str:
    lines = [
        "# Frozen Baseline Validation",
        "",
        f"- manifest: `{manifest_path}`",
        f"- git commit: `{get_git_commit()}`",
        f"- git dirty: `{get_git_dirty()}`",
        "",
    ]
    if missing:
        lines.extend(
            [
                "## Missing Prerequisites",
                "",
            ]
        )
        for item in missing:
            lines.append(f"- `{item}`")
        lines.extend(["", "## Verdict", "", "INCOMPLETE: required artifact missing"])
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Frozen Summary",
            "",
            "| Variant | Combined Mean | Retry-Block Mean |",
            "| --- | ---: | ---: |",
        ]
    )
    assert candidate is not None
    for variant in manifest.get("variants", {}):
        lines.append(
            f"| {DISPLAY_NAMES[variant]} | {_format_float(candidate['metrics']['combined'][variant]['mean'])} | {_format_float(candidate['metrics']['retry_block'][variant]['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Validation Checks",
            "",
            "| Section | Metric | Variant | Expected | Actual | Status |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in validation_rows:
        lines.append(f"| {row.section} | {row.metric} | {row.variant} | `{row.expected}` | `{row.actual}` | `{row.status}` |")
    lines.extend(["", "## Verdict", "", verdict])
    return "\n".join(lines) + "\n"


def _build_claim_ledger() -> str:
    lines = [
        "# Claim Ledger",
        "",
        "| Experiment Family | Status | Scope | Why It Changed Or Did Not Change The Claim | Authoritative Reports |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in LEDGER_ROWS:
        reports = "<br>".join(f"`{report}`" for report in row["reports"])
        lines.append(f"| {row['family']} | `{row['status']}` | {row['scope']} | {row['claim_effect']} | {reports} |")
    return "\n".join(lines) + "\n"


def _build_future_retry_template(manifest: dict[str, Any], manifest_path: Path) -> str:
    retry_block = manifest.get("thaw_gate", {}).get("retry_block", {})
    combined = manifest.get("thaw_gate", {}).get("combined_picture", {})
    lines = [
        "# Future Retry Template",
        "",
        "Use this template before any future DoorKey retry is run. Do not skip any field.",
        "",
        "## Mechanism Hypothesis",
        "",
        "- hypothesis:",
        "- evidence source report(s):",
        "- why this is one mechanism rather than a broad search:",
        "",
        "## Exact Intervention",
        "",
        "- intervention family (must stay inside current KL learner-state family):",
        "- exact config path(s):",
        "- exact CLI:",
        "",
        "## Fixed Seeds And Controls",
        "",
        f"- retry-block seeds: `{retry_block.get('seeds', [])}`",
        "- required controls on the same block:",
        "  - recovered token_dense",
        "  - KL learner-state token_dense",
        "  - KL learner-state single_expert",
        "  - baseline PPO SARE",
        "",
        "## Success Bar",
        "",
        f"- candidate KL learner-state SARE retry-block mean must exceed `{retry_block.get('candidate_sare_mean_must_exceed', 0.0):.4f}`",
        "- candidate KL learner-state SARE must at least match matched KL learner-state single_expert on the same block",
        f"- candidate combined DoorKey KL learner-state SARE mean must stay at or above `{combined.get('candidate_sare_mean_must_be_gte', 0.0):.4f}`",
        "",
        "## Failure Bar",
        "",
        "- any missing fairness control",
        "- any non-external-64 evaluation path",
        "- any disallowed claim widening",
        "- any regression below the frozen combined DoorKey picture",
        "",
        "## Required Outputs",
        "",
        "- retry report markdown + csv",
        "- updated combined DoorKey report markdown + csv",
        "- claim gate report markdown + json",
        "",
        "## Claim-Gate Invocation",
        "",
        f"Run the candidate through the automated gate using the manifest `{manifest_path}` before any public claim update.",
        "",
        "```bash",
        "./scripts/run_claim_gate.sh \\",
        f"  {manifest_path} \\",
        "  <candidate-json> \\",
        "  outputs/reports/claim_gate_retry_check.md \\",
        "  outputs/reports/claim_gate_retry_check.json",
        "```",
    ]
    return "\n".join(lines) + "\n"


def _build_decision_memo(manifest: dict[str, Any], validation_verdict: str, gate_verdict: str) -> str:
    thaw = manifest.get("thaw_gate", {})
    lines = [
        "# Freeze-Hardening Decision Memo",
        "",
        "## Answers",
        "",
        f"1. Current allowed claim envelope: `{manifest['claim']['allowed_claim_text']}` only. Not allowed: `{', '.join(manifest['claim']['disallowed_claim_keys'])}`. See [frozen_claim_envelope.md](frozen_claim_envelope.md).",
        f"2. The frozen state is defined by the manifest `{manifest['claim']['id']}` plus the authoritative combined and retry-block reports. The key thresholds are combined KL learner-state `SARE` mean `{manifest['thresholds']['combined_means']['kl_lss_sare']:.4f}` and retry-block KL learner-state `SARE` mean `{manifest['thresholds']['retry_block_means']['kl_lss_sare']:.4f}`. See [frozen_claim_manifest_report.md](frozen_claim_manifest_report.md).",
        f"3. Any future thaw candidate must use the external 64-episode DoorKey path, include the required fairness controls, beat the frozen retry-block KL learner-state `SARE` mean `{manifest['thaw_gate']['retry_block']['candidate_sare_mean_must_exceed']:.4f}`, at least match same-block KL learner-state `single_expert`, and preserve the combined DoorKey KL learner-state `SARE` mean `{manifest['thaw_gate']['combined_picture']['candidate_sare_mean_must_be_gte']:.4f}`. See [future_retry_template.md](future_retry_template.md).",
        f"4. The project is now set up to resist overclaiming: the frozen baseline validation currently returns `{validation_verdict}`, and the automated dry-run claim gate returns `{gate_verdict}` on the current frozen artifacts. See [frozen_baseline_validation.md](frozen_baseline_validation.md) and [claim_gate_dry_run.md](claim_gate_dry_run.md).",
        "5. Recommendation: stay frozen until a preregistered retry clears the automated gate.",
    ]
    return "\n".join(lines) + "\n"


def _build_ci_claim_gate_report(
    frozen_pack_validation: dict[str, Any],
    pack_gate: dict[str, Any],
    incomplete_gate: dict[str, Any],
) -> str:
    lines = [
        "# CI Claim Gate Report",
        "",
        "## Workflow",
        "",
        "- GitHub Actions workflow: `.github/workflows/claim-gate.yml`",
        "- Pull request template: `.github/pull_request_template.md`",
        "",
        "## Local Entry Points",
        "",
        "- `./scripts/run_frozen_benchmark_pack_validation.sh`",
        "- `./scripts/run_claim_gate.sh`",
        "- `./scripts/run_freeze_hardening_finalize.sh`",
        "",
        "## Current Dry Runs",
        "",
        f"- frozen benchmark pack validation: `{frozen_pack_validation.get('verdict', '-')}`",
        f"- pack-based claim gate on the current frozen candidate: `{pack_gate.get('verdict', '-')}`",
        f"- malformed/incomplete candidate pack gate: `{incomplete_gate.get('verdict', '-')}`",
        "",
        "## Enforcement Rule",
        "",
        "- Any future thaw candidate should include a candidate result pack path, a pack-based gate report, and a retry-template reference before claim language is discussed in review.",
    ]
    return "\n".join(lines) + "\n"


def _build_operational_memo(
    frozen_pack: dict[str, Any],
    frozen_pack_path: Path,
    frozen_pack_validation: dict[str, Any],
    pack_gate: dict[str, Any],
    incomplete_gate: dict[str, Any],
) -> str:
    thaw = frozen_pack.get("thaw_gate", {})
    retry = thaw.get("retry_block", {})
    combined = thaw.get("combined_picture", {})
    lines = [
        "# Freeze-Hardening Operational Memo",
        "",
        "## Answers",
        "",
        f"1. The frozen benchmark pack is `{frozen_pack_path}`, schema version `{frozen_pack.get('schema_version', '-')}`, sealing the claim `{frozen_pack.get('claim', {}).get('allowed_claim_text', '-')}` against the canonical DoorKey artifacts and manifest.",
        f"2. The pack is validated by hash and schema. The current dry run returns `{frozen_pack_validation.get('verdict', '-')}`. See [frozen_benchmark_pack_validation.md](frozen_benchmark_pack_validation.md).",
        "3. Any future candidate must be packaged as a candidate result pack with the required controls, metrics, actual lane/seed sets, and hashed artifacts. See [candidate_result_pack_schema.md](candidate_result_pack_schema.md) and [candidate_result_pack_template.json](candidate_result_pack_template.json).",
        f"4. The automated gate validates both packs first, then applies the frozen DoorKey thresholds: retry-block KL learner-state `SARE` must beat `{retry.get('candidate_sare_mean_must_exceed', 0.0):.4f}`, match or beat same-block `single_expert`, and preserve combined KL learner-state `SARE` mean `{combined.get('candidate_sare_mean_must_be_gte', 0.0):.4f}`. The current dry run returns `{pack_gate.get('verdict', '-')}` and malformed candidates return `{incomplete_gate.get('verdict', '-')}`. See [claim_gate_pack_dry_run.md](claim_gate_pack_dry_run.md).",
        "5. Repo workflow now includes a claim-gate GitHub Action and a PR template that require claim scope, candidate pack path, gate result, and retry-template reference for claim-sensitive changes. See [ci_claim_gate_report.md](ci_claim_gate_report.md).",
        "6. Yes. The project is now operationally frozen until a preregistered retry clears the pack-based gate.",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze-hardening utilities for the frozen DoorKey SARE claim.")
    sub = parser.add_subparsers(dest="command", required=True)

    envelope = sub.add_parser("claim-envelope-report")
    envelope.add_argument("--manifest", required=True)
    envelope.add_argument("--output", required=True)

    manifest_report = sub.add_parser("manifest-report")
    manifest_report.add_argument("--manifest", required=True)
    manifest_report.add_argument("--output", required=True)

    validate = sub.add_parser("validate-frozen-baseline")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--csv", required=True)
    validate.add_argument("--json-output", required=True)

    ledger = sub.add_parser("claim-ledger")
    ledger.add_argument("--output", required=True)

    retry_template = sub.add_parser("future-retry-template")
    retry_template.add_argument("--manifest", required=True)
    retry_template.add_argument("--output", required=True)

    memo = sub.add_parser("decision-memo")
    memo.add_argument("--manifest", required=True)
    memo.add_argument("--validation-json", required=True)
    memo.add_argument("--claim-gate-json", required=True)
    memo.add_argument("--output", required=True)

    ci_report = sub.add_parser("ci-claim-gate-report")
    ci_report.add_argument("--frozen-pack-validation-json", required=True)
    ci_report.add_argument("--claim-gate-pack-json", required=True)
    ci_report.add_argument("--claim-gate-incomplete-json", required=True)
    ci_report.add_argument("--output", required=True)

    operational = sub.add_parser("operational-memo")
    operational.add_argument("--frozen-pack", required=True)
    operational.add_argument("--frozen-pack-validation-json", required=True)
    operational.add_argument("--claim-gate-pack-json", required=True)
    operational.add_argument("--claim-gate-incomplete-json", required=True)
    operational.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = args.command

    if command == "claim-envelope-report":
        manifest_path = Path(args.manifest)
        manifest = load_manifest(manifest_path)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_build_envelope_report(manifest, manifest_path), encoding="utf-8")
        return

    if command == "manifest-report":
        manifest_path = Path(args.manifest)
        manifest = load_manifest(manifest_path)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_build_manifest_report(manifest, manifest_path), encoding="utf-8")
        return

    if command == "validate-frozen-baseline":
        manifest_path = Path(args.manifest)
        manifest = load_manifest(manifest_path)
        output = Path(args.output)
        csv_output = Path(args.csv)
        json_output = Path(args.json_output)
        candidate, missing = _load_baseline_candidate(manifest)
        if missing:
            validation_rows: list[ValidationRow] = []
            verdict = "INCOMPLETE: required artifact missing"
        else:
            validation_rows, verdict = _candidate_validation_rows(manifest, candidate)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            _build_validation_report(manifest_path, manifest, candidate if candidate else None, validation_rows, verdict, missing),
            encoding="utf-8",
        )
        _write_csv(
            csv_output,
            ["section", "metric", "variant", "expected", "actual", "status"],
            [asdict(row) for row in validation_rows],
        )
        json_output.parent.mkdir(parents=True, exist_ok=True)
        candidate_payload = candidate or {
            "schema_version": 1,
            "candidate_name": "missing_frozen_baseline",
            "task": manifest.get("evaluation", {}).get("task", "DoorKey"),
            "evaluation": {
                "path_key": manifest.get("evaluation", {}).get("path_key", "external_policy_diagnostics"),
                "episodes": int(manifest.get("evaluation", {}).get("episodes", 64)),
            },
            "requested_claims": [manifest.get("claim", {}).get("allowed_claim_key", "bounded_teacher_guided_doorkey_sare")],
            "controls_present": [],
            "metrics": {},
        }
        candidate_payload["_validation"] = {
            "manifest": str(manifest_path),
            "verdict": verdict,
            "missing": missing,
            "checks": [asdict(row) for row in validation_rows],
        }
        json_output.write_text(
            json.dumps(candidate_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return

    if command == "claim-ledger":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_build_claim_ledger(), encoding="utf-8")
        return

    if command == "future-retry-template":
        manifest_path = Path(args.manifest)
        manifest = load_manifest(manifest_path)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_build_future_retry_template(manifest, manifest_path), encoding="utf-8")
        return

    if command == "decision-memo":
        manifest_path = Path(args.manifest)
        manifest = load_manifest(manifest_path)
        validation_json = json.loads(Path(args.validation_json).read_text(encoding="utf-8"))
        claim_gate_json = json.loads(Path(args.claim_gate_json).read_text(encoding="utf-8"))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            _build_decision_memo(
                manifest,
                str(validation_json.get("_validation", {}).get("verdict", "INCOMPLETE: required artifact missing")),
                str(claim_gate_json["verdict"]),
            ),
            encoding="utf-8",
        )
        return

    if command == "ci-claim-gate-report":
        frozen_pack_validation = json.loads(Path(args.frozen_pack_validation_json).read_text(encoding="utf-8"))
        pack_gate = json.loads(Path(args.claim_gate_pack_json).read_text(encoding="utf-8"))
        incomplete_gate = json.loads(Path(args.claim_gate_incomplete_json).read_text(encoding="utf-8"))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            _build_ci_claim_gate_report(
                frozen_pack_validation,
                pack_gate,
                incomplete_gate,
            ),
            encoding="utf-8",
        )
        return

    if command == "operational-memo":
        frozen_pack_path = Path(args.frozen_pack)
        frozen_pack = json.loads(frozen_pack_path.read_text(encoding="utf-8"))
        frozen_pack_validation = json.loads(Path(args.frozen_pack_validation_json).read_text(encoding="utf-8"))
        pack_gate = json.loads(Path(args.claim_gate_pack_json).read_text(encoding="utf-8"))
        incomplete_gate = json.loads(Path(args.claim_gate_incomplete_json).read_text(encoding="utf-8"))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            _build_operational_memo(
                frozen_pack,
                frozen_pack_path,
                frozen_pack_validation,
                pack_gate,
                incomplete_gate,
            ),
            encoding="utf-8",
        )
        return

    raise ValueError(f"Unsupported command: {command}")


if __name__ == "__main__":
    main()
