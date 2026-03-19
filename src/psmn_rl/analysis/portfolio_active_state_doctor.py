from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _write_json
from psmn_rl.analysis.portfolio_candidate_pack_loader import load_portfolio_candidate_pack
from psmn_rl.analysis.portfolio_frontier_contract_loader import load_frontier_contract
from psmn_rl.analysis.portfolio_frontier_doctor_loader import load_frontier_doctor_report
from psmn_rl.analysis.portfolio_gate_report_loader import load_portfolio_gate_report
from psmn_rl.utils.io import get_git_commit, get_git_dirty


EXPECTED_ACTIVE_PACK = "outputs/reports/portfolio_candidate_pack.json"
EXPECTED_ARCHIVED_PACK = "outputs/reports/frozen_benchmark_pack.json"
EXPECTED_GATE_VERDICT = "PASS: thaw consideration allowed"


def evaluate_active_state() -> dict[str, Any]:
    candidate_pack = load_portfolio_candidate_pack()
    gate_report = load_portfolio_gate_report()
    contract = load_frontier_contract()
    frontier_doctor = load_frontier_doctor_report()

    checks = [
        {
            "label": "candidate_round6",
            "status": "pass" if candidate_pack.candidate_name == "round6" else "fail",
            "detail": f"candidate_name={candidate_pack.candidate_name}",
        },
        {
            "label": "candidate_pack_current",
            "status": "pass" if candidate_pack.portfolio_campaign.active_canonical_pack == EXPECTED_ACTIVE_PACK else "fail",
            "detail": f"active_canonical_pack={candidate_pack.portfolio_campaign.active_canonical_pack}",
        },
        {
            "label": "candidate_archived_pack_frozen",
            "status": (
                "pass"
                if candidate_pack.portfolio_campaign.archived_legacy_pack == EXPECTED_ARCHIVED_PACK
                and candidate_pack.frozen_pack_reference.path == EXPECTED_ARCHIVED_PACK
                else "fail"
            ),
            "detail": (
                "archived_legacy_pack="
                f"{candidate_pack.portfolio_campaign.archived_legacy_pack}, frozen_pack_reference={candidate_pack.frozen_pack_reference.path}"
            ),
        },
        {
            "label": "candidate_eval_doorkey_external64",
            "status": (
                "pass"
                if candidate_pack.task == "DoorKey"
                and candidate_pack.evaluation.task == "DoorKey"
                and candidate_pack.evaluation.path_key == "external_policy_diagnostics"
                and candidate_pack.evaluation.episodes == 64
                else "fail"
            ),
            "detail": (
                f"task={candidate_pack.task}, evaluation.task={candidate_pack.evaluation.task}, "
                f"path_key={candidate_pack.evaluation.path_key}, episodes={candidate_pack.evaluation.episodes}"
            ),
        },
        {
            "label": "gate_targets_current_active_pack",
            "status": (
                "pass"
                if gate_report.candidate_pack == EXPECTED_ACTIVE_PACK and gate_report.frozen_pack == EXPECTED_ARCHIVED_PACK
                else "fail"
            ),
            "detail": f"candidate_pack={gate_report.candidate_pack}, frozen_pack={gate_report.frozen_pack}",
        },
        {
            "label": "gate_pack_mode_pass",
            "status": (
                "pass" if gate_report.mode == "pack" and gate_report.verdict == EXPECTED_GATE_VERDICT else "fail"
            ),
            "detail": f"mode={gate_report.mode}, verdict={gate_report.verdict}",
        },
        {
            "label": "gate_combined_picture_pass",
            "status": (
                "pass"
                if gate_report.check_by_name("combined_picture_mean").status == "PASS"
                and gate_report.check_by_name("combined_picture_failures").status == "PASS"
                else "fail"
            ),
            "detail": (
                f"combined_picture_mean={gate_report.check_by_name('combined_picture_mean').status}, "
                f"combined_picture_failures={gate_report.check_by_name('combined_picture_failures').status}"
            ),
        },
        {
            "label": "contract_matches_active_roles",
            "status": (
                "pass"
                if contract.benchmark.active_candidate == "round6"
                and contract.benchmark.active_candidate_pack == EXPECTED_ACTIVE_PACK
                and contract.benchmark.archived_frozen_pack == EXPECTED_ARCHIVED_PACK
                and contract.frontier_roles.default_restart_prior == "round7"
                and contract.frontier_roles.replay_validated_alternate == "round10"
                else "fail"
            ),
            "detail": (
                f"active_candidate={contract.benchmark.active_candidate}, "
                f"default_restart_prior={contract.frontier_roles.default_restart_prior}, "
                f"replay_validated_alternate={contract.frontier_roles.replay_validated_alternate}"
            ),
        },
        {
            "label": "frontier_doctor_pass",
            "status": "pass" if frontier_doctor.overall == "pass" else "fail",
            "detail": f"frontier_doctor.overall={frontier_doctor.overall}",
        },
    ]
    overall = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {"overall": overall, "checks": checks}


def render_active_state_doctor(output: Path | None, json_output: Path | None) -> dict[str, Any]:
    result = evaluate_active_state()

    if output is not None:
        lines = [
            "# Portfolio Active-State Doctor",
            "",
            f"- git commit: `{get_git_commit()}`",
            f"- git dirty: `{get_git_dirty()}`",
            f"- overall: `{result['overall']}`",
            "",
            "## Checks",
            "",
            "| Check | Status | Detail |",
            "| --- | --- | --- |",
        ]
        for check in result["checks"]:
            lines.append(f"| `{check['label']}` | `{check['status']}` | {check['detail']} |")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if json_output is not None:
        _write_json(json_output, result)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Doctor check for the active portfolio benchmark state")
    parser.add_argument("--output", required=False)
    parser.add_argument("--json", required=False)
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = render_active_state_doctor(
        Path(args.output) if args.output else None,
        Path(args.json) if args.json else None,
    )
    if args.fail_on_drift and result["overall"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
