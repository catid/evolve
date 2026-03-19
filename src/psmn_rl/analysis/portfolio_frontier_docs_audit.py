from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psmn_rl.analysis.lss_post_pass_campaign import _write_json
from psmn_rl.analysis.portfolio_frontier_contract_loader import FrontierContract, load_frontier_contract
from psmn_rl.utils.io import get_git_commit, get_git_dirty

CURRENT_DECISION_REPORT = "outputs/reports/portfolio_decision_memo.md"
CURRENT_GATE_REPORT = "outputs/reports/portfolio_gate_report.md"
FRONTIER_CONTRACT_REPORT = "outputs/reports/portfolio_frontier_contract.json"
FRONTIER_GUARD_REPORT = "outputs/reports/portfolio_frontier_guard_report.md"
FRONTIER_GUARD_WORKFLOW_CONTRACT_REPORT = "outputs/reports/portfolio_frontier_guard_workflow_contract.md"
FRONTIER_GUARD_SCRIPT = "scripts/run_portfolio_frontier_guard.sh"
ACTIVE_STATE_DOCTOR_REPORT = "outputs/reports/portfolio_active_state_doctor.md"
OPERATIONAL_STATE_REPORT = "outputs/reports/portfolio_operational_state.md"
SEED_PACK_REPORT = "outputs/reports/portfolio_seed_pack.json"
SEED_PACK_DOCTOR_REPORT = "outputs/reports/portfolio_seed_pack_doctor.md"
CLAIM_LEDGER_PORTFOLIO_ROW = "50/50 portfolio campaign"


@dataclass(frozen=True)
class FileExpectation:
    path: str
    required_snippets: tuple[str, ...]


def build_expectations(contract: FrontierContract) -> tuple[FileExpectation, ...]:
    round6 = contract.benchmark.active_candidate
    active_pack = contract.benchmark.active_candidate_pack
    round7 = contract.frontier_roles.default_restart_prior
    round10 = contract.frontier_roles.replay_validated_alternate
    return (
        FileExpectation(
            path="README.md",
            required_snippets=(
                round6,
                active_pack,
                CURRENT_DECISION_REPORT,
                CURRENT_GATE_REPORT,
                FRONTIER_CONTRACT_REPORT,
                FRONTIER_GUARD_REPORT,
                FRONTIER_GUARD_WORKFLOW_CONTRACT_REPORT,
                ACTIVE_STATE_DOCTOR_REPORT,
                OPERATIONAL_STATE_REPORT,
                SEED_PACK_REPORT,
                SEED_PACK_DOCTOR_REPORT,
                FRONTIER_GUARD_SCRIPT,
                round7,
                round10,
            ),
        ),
        FileExpectation(
            path="summary.md",
            required_snippets=(
                round6,
                active_pack,
                CURRENT_DECISION_REPORT,
                FRONTIER_CONTRACT_REPORT,
                FRONTIER_GUARD_REPORT,
                FRONTIER_GUARD_WORKFLOW_CONTRACT_REPORT,
                ACTIVE_STATE_DOCTOR_REPORT,
                OPERATIONAL_STATE_REPORT,
                SEED_PACK_REPORT,
                SEED_PACK_DOCTOR_REPORT,
                FRONTIER_GUARD_SCRIPT,
                round7,
                round10,
            ),
        ),
        FileExpectation(
            path="report.md",
            required_snippets=(
                f"`{round6}` is the active canonical benchmark within DoorKey only",
                active_pack,
                CURRENT_DECISION_REPORT,
                CURRENT_GATE_REPORT,
                FRONTIER_GUARD_REPORT,
                FRONTIER_GUARD_WORKFLOW_CONTRACT_REPORT,
                ACTIVE_STATE_DOCTOR_REPORT,
                OPERATIONAL_STATE_REPORT,
                "portfolio_seed_pack.md",
                SEED_PACK_DOCTOR_REPORT,
                round7,
                round10,
            ),
        ),
        FileExpectation(
            path="outputs/reports/claim_ledger.md",
            required_snippets=(
                CLAIM_LEDGER_PORTFOLIO_ROW,
                CURRENT_DECISION_REPORT,
                CURRENT_GATE_REPORT,
                "active canonical benchmark",
            ),
        ),
    )


def audit_expected_file(expectation: FileExpectation, text: str) -> dict[str, Any]:
    missing = [snippet for snippet in expectation.required_snippets if snippet not in text]
    return {
        "path": expectation.path,
        "status": "pass" if not missing else "fail",
        "missing": missing,
    }


def render_docs_audit(output: Path | None, json_output: Path | None) -> dict[str, Any]:
    contract = load_frontier_contract()
    checks = []
    for expectation in build_expectations(contract):
        text = Path(expectation.path).read_text(encoding="utf-8")
        checks.append(audit_expected_file(expectation, text))
    overall = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    result = {
        "overall": overall,
        "active_candidate": contract.benchmark.active_candidate,
        "default_restart_prior": contract.frontier_roles.default_restart_prior,
        "replay_validated_alternate": contract.frontier_roles.replay_validated_alternate,
        "active_candidate_pack": contract.benchmark.active_candidate_pack,
        "checks": checks,
    }

    if output is not None:
        lines = [
            "# Portfolio Frontier Docs Audit",
            "",
            f"- git commit: `{get_git_commit()}`",
            f"- git dirty: `{get_git_dirty()}`",
            f"- overall: `{overall}`",
            f"- active candidate: `{contract.benchmark.active_candidate}`",
            f"- default restart prior: `{contract.frontier_roles.default_restart_prior}`",
            f"- replay-validated alternate: `{contract.frontier_roles.replay_validated_alternate}`",
            f"- active candidate pack: `{contract.benchmark.active_candidate_pack}`",
            "",
            "## Checks",
            "",
            "| File | Status | Missing snippets |",
            "| --- | --- | --- |",
        ]
        for check in checks:
            missing = ", ".join(f"`{snippet}`" for snippet in check["missing"]) or "`none`"
            lines.append(f"| `{check['path']}` | `{check['status']}` | {missing} |")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if json_output is not None:
        _write_json(json_output, result)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit top-level docs against the frozen portfolio frontier contract")
    parser.add_argument("--output", required=False)
    parser.add_argument("--json", required=False)
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = render_docs_audit(
        Path(args.output) if args.output else None,
        Path(args.json) if args.json else None,
    )
    if args.fail_on_drift and result["overall"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
