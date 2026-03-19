from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GUARD_WORKFLOW_CONTRACT_PATH = Path("outputs/reports/portfolio_frontier_guard_workflow_contract.json")


@dataclass(frozen=True)
class GuardWorkflowContractCheck:
    label: str
    status: str
    detail: str


@dataclass(frozen=True)
class PortfolioFrontierGuardWorkflowContract:
    status_type: str
    overall: str
    workflow_name: str
    workflow_dispatch_enabled: bool
    pull_request_paths: tuple[str, ...]
    guard_script: str
    syntax_check_command: str
    guard_run_command: str
    upload_artifact_name: str
    upload_artifact_paths: tuple[str, ...]
    checks: tuple[GuardWorkflowContractCheck, ...]

    def check_by_label(self, label: str) -> GuardWorkflowContractCheck:
        for check in self.checks:
            if check.label == label:
                return check
        raise KeyError(label)


def load_frontier_guard_workflow_contract(
    path: Path | None = None,
) -> PortfolioFrontierGuardWorkflowContract:
    report_path = path or DEFAULT_GUARD_WORKFLOW_CONTRACT_PATH
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return PortfolioFrontierGuardWorkflowContract(
        status_type=str(data["status_type"]),
        overall=str(data["overall"]),
        workflow_name=str(data["workflow_name"]),
        workflow_dispatch_enabled=bool(data["workflow_dispatch_enabled"]),
        pull_request_paths=tuple(str(path) for path in data["pull_request_paths"]),
        guard_script=str(data["guard_script"]),
        syntax_check_command=str(data["syntax_check_command"]),
        guard_run_command=str(data["guard_run_command"]),
        upload_artifact_name=str(data["upload_artifact_name"]),
        upload_artifact_paths=tuple(str(path) for path in data["upload_artifact_paths"]),
        checks=tuple(
            GuardWorkflowContractCheck(
                label=str(row["label"]),
                status=str(row["status"]),
                detail=str(row["detail"]),
            )
            for row in data["checks"]
        ),
    )
