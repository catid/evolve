from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _read_json, _write_json
from psmn_rl.analysis.portfolio_frontier_contract_loader import load_frontier_contract
from psmn_rl.utils.io import get_git_commit, get_git_dirty

CONSISTENCY_PATH = Path("outputs/reports/portfolio_frontier_consistency.json")
DOCS_AUDIT_PATH = Path("outputs/reports/portfolio_frontier_docs_audit.json")
DOCTOR_PATH = Path("outputs/reports/portfolio_frontier_doctor.json")
SEED_PACK_DOCTOR_PATH = Path("outputs/reports/portfolio_seed_pack_doctor.json")


def evaluate_guard_stack(
    consistency_overall: str,
    docs_audit_overall: str,
    doctor_overall: str,
    seed_pack_doctor_overall: str,
) -> dict[str, Any]:
    checks = [
        {
            "label": "consistency_pass",
            "status": "pass" if consistency_overall == "pass" else "fail",
            "detail": f"consistency_overall={consistency_overall}",
        },
        {
            "label": "docs_audit_pass",
            "status": "pass" if docs_audit_overall == "pass" else "fail",
            "detail": f"docs_audit_overall={docs_audit_overall}",
        },
        {
            "label": "doctor_pass",
            "status": "pass" if doctor_overall == "pass" else "fail",
            "detail": f"doctor_overall={doctor_overall}",
        },
        {
            "label": "seed_pack_doctor_pass",
            "status": "pass" if seed_pack_doctor_overall == "pass" else "fail",
            "detail": f"seed_pack_doctor_overall={seed_pack_doctor_overall}",
        },
    ]
    overall = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {"overall": overall, "checks": checks}


def render_guard_report(output: Path | None, json_output: Path | None) -> dict[str, Any]:
    contract = load_frontier_contract()
    consistency = _read_json(CONSISTENCY_PATH)
    docs_audit = _read_json(DOCS_AUDIT_PATH)
    doctor = _read_json(DOCTOR_PATH)
    seed_pack_doctor = _read_json(SEED_PACK_DOCTOR_PATH)
    result = evaluate_guard_stack(
        consistency_overall=str(consistency["overall"]),
        docs_audit_overall=str(docs_audit["overall"]),
        doctor_overall=str(doctor["overall"]),
        seed_pack_doctor_overall=str(seed_pack_doctor["overall"]),
    )
    result["active_candidate"] = contract.benchmark.active_candidate
    result["active_candidate_pack"] = contract.benchmark.active_candidate_pack
    result["archived_frozen_pack"] = contract.benchmark.archived_frozen_pack
    result["default_restart_prior"] = contract.frontier_roles.default_restart_prior
    result["replay_validated_alternate"] = contract.frontier_roles.replay_validated_alternate

    if output is not None:
        lines = [
            "# Portfolio Frontier Guard Report",
            "",
            f"- git commit: `{get_git_commit()}`",
            f"- git dirty: `{get_git_dirty()}`",
            f"- overall: `{result['overall']}`",
            f"- active candidate: `{result['active_candidate']}`",
            f"- active candidate pack: `{result['active_candidate_pack']}`",
            f"- archived frozen pack: `{result['archived_frozen_pack']}`",
            f"- default restart prior: `{result['default_restart_prior']}`",
            f"- replay-validated alternate: `{result['replay_validated_alternate']}`",
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
    parser = argparse.ArgumentParser(description="Consolidated guard report for the frozen portfolio frontier")
    parser.add_argument("--output", required=False)
    parser.add_argument("--json", required=False)
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = render_guard_report(
        Path(args.output) if args.output else None,
        Path(args.json) if args.json else None,
    )
    if args.fail_on_drift and result["overall"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
