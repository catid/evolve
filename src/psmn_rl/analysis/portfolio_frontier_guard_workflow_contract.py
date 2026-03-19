from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from psmn_rl.analysis.lss_post_pass_campaign import _write_json
from psmn_rl.utils.io import get_git_commit, get_git_dirty


WORKFLOW_PATH = Path(".github/workflows/portfolio-frontier-guard.yml")
WORKFLOW_NAME = "portfolio-frontier-guard"
JOB_NAME = "portfolio-frontier-guard"
GUARD_SCRIPT_PATH = "scripts/run_portfolio_frontier_guard.sh"
SYNTAX_CHECK_COMMAND = f"bash -n {GUARD_SCRIPT_PATH}"
GUARD_RUN_COMMAND = f"bash ./{GUARD_SCRIPT_PATH}"
UPLOAD_ARTIFACT_NAME = "portfolio-frontier-guard-reports"
REQUIRED_UPLOAD_ARTIFACTS = (
    "outputs/reports/portfolio_frontier_docs_audit.md",
    "outputs/reports/portfolio_frontier_docs_audit.json",
    "outputs/reports/portfolio_frontier_doctor.md",
    "outputs/reports/portfolio_frontier_doctor.json",
    "outputs/reports/portfolio_frontier_guard_report.md",
    "outputs/reports/portfolio_frontier_guard_report.json",
    "outputs/reports/portfolio_frontier_guard_workflow_contract.md",
    "outputs/reports/portfolio_frontier_guard_workflow_contract.json",
    "outputs/reports/portfolio_active_state_doctor.md",
    "outputs/reports/portfolio_active_state_doctor.json",
    "outputs/reports/portfolio_operational_state.md",
    "outputs/reports/portfolio_operational_state.json",
    "outputs/reports/portfolio_seed_pack_doctor.md",
    "outputs/reports/portfolio_seed_pack_doctor.json",
)


def workflow_on_block(workflow: dict[str, Any]) -> dict[str, Any]:
    return workflow.get("on", workflow.get(True, {}))


def _check(label: str, ok: bool, detail: str) -> dict[str, str]:
    return {"label": label, "status": "pass" if ok else "fail", "detail": detail}


def build_guard_workflow_contract() -> dict[str, Any]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    on_block = workflow_on_block(workflow)
    steps = workflow["jobs"][JOB_NAME]["steps"]
    run_steps = [str(step["run"]) for step in steps if "run" in step]
    upload_step = next(
        (step for step in reversed(steps) if step.get("uses") == "actions/upload-artifact@v4"),
        None,
    )
    upload_with = upload_step.get("with", {}) if upload_step else {}
    upload_artifact_name = str(upload_with.get("name", ""))
    upload_artifact_paths = tuple(
        line.strip()
        for line in str(upload_with.get("path", "")).splitlines()
        if line.strip()
    )
    checks = [
        _check(
            "workflow_name_matches",
            str(workflow.get("name", "")) == WORKFLOW_NAME,
            f"workflow_name={workflow.get('name', '')}",
        ),
        _check(
            "workflow_dispatch_enabled",
            "workflow_dispatch" in on_block,
            f"workflow_dispatch_enabled={'workflow_dispatch' in on_block}",
        ),
        _check(
            "syntax_check_present",
            any(SYNTAX_CHECK_COMMAND in run for run in run_steps),
            f"required_command={SYNTAX_CHECK_COMMAND}",
        ),
        _check(
            "guard_run_present",
            any(GUARD_RUN_COMMAND in run for run in run_steps),
            f"required_command={GUARD_RUN_COMMAND}",
        ),
        _check(
            "upload_step_present",
            upload_step is not None,
            "uses=actions/upload-artifact@v4",
        ),
        _check(
            "upload_artifact_name_matches",
            upload_artifact_name == UPLOAD_ARTIFACT_NAME,
            f"upload_artifact_name={upload_artifact_name}",
        ),
        _check(
            "upload_paths_cover_required",
            set(REQUIRED_UPLOAD_ARTIFACTS).issubset(set(upload_artifact_paths)),
            f"required_upload_count={len(REQUIRED_UPLOAD_ARTIFACTS)}",
        ),
    ]
    overall = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {
        "status_type": "portfolio_frontier_guard_workflow_contract",
        "overall": overall,
        "workflow_name": str(workflow.get("name", "")),
        "workflow_dispatch_enabled": "workflow_dispatch" in on_block,
        "pull_request_paths": tuple(str(path) for path in on_block.get("pull_request", {}).get("paths", [])),
        "guard_script": GUARD_SCRIPT_PATH,
        "syntax_check_command": SYNTAX_CHECK_COMMAND,
        "guard_run_command": GUARD_RUN_COMMAND,
        "upload_artifact_name": upload_artifact_name,
        "upload_artifact_paths": upload_artifact_paths,
        "checks": checks,
    }


def render_guard_workflow_contract(output: Path | None, json_output: Path | None) -> dict[str, Any]:
    result = build_guard_workflow_contract()

    if output is not None:
        lines = [
            "# Portfolio Frontier Guard Workflow Contract",
            "",
            f"- git commit: `{get_git_commit()}`",
            f"- git dirty: `{get_git_dirty()}`",
            f"- overall: `{result['overall']}`",
            f"- workflow name: `{result['workflow_name']}`",
            f"- guard script: `{result['guard_script']}`",
            f"- syntax check command: `{result['syntax_check_command']}`",
            f"- guard run command: `{result['guard_run_command']}`",
            f"- upload artifact name: `{result['upload_artifact_name']}`",
            f"- workflow dispatch enabled: `{result['workflow_dispatch_enabled']}`",
            "",
            "## Upload Paths",
            "",
        ]
        lines.extend(f"- `{path}`" for path in result["upload_artifact_paths"])
        lines.extend(
            [
                "",
                "## Checks",
                "",
                "| Check | Status | Detail |",
                "| --- | --- | --- |",
            ]
        )
        for check in result["checks"]:
            lines.append(f"| `{check['label']}` | `{check['status']}` | {check['detail']} |")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if json_output is not None:
        _write_json(json_output, result)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the frozen CI workflow contract for the frontier guard")
    parser.add_argument("--output", required=False)
    parser.add_argument("--json", required=False)
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = render_guard_workflow_contract(
        Path(args.output) if args.output else None,
        Path(args.json) if args.json else None,
    )
    if args.fail_on_drift and result["overall"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
