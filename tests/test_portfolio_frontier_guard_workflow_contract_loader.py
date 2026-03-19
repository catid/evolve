from psmn_rl.analysis.portfolio_frontier_guard_workflow_contract import (
    REQUIRED_UPLOAD_ARTIFACTS,
    UPLOAD_ARTIFACT_NAME,
)
from psmn_rl.analysis.portfolio_frontier_guard_workflow_contract_loader import (
    load_frontier_guard_workflow_contract,
)


def test_load_frontier_guard_workflow_contract_snapshot() -> None:
    contract = load_frontier_guard_workflow_contract()
    assert contract.status_type == "portfolio_frontier_guard_workflow_contract"
    assert contract.overall == "pass"
    assert contract.workflow_name == "portfolio-frontier-guard"
    assert contract.workflow_dispatch_enabled is True
    assert contract.upload_artifact_name == UPLOAD_ARTIFACT_NAME
    assert set(REQUIRED_UPLOAD_ARTIFACTS).issubset(set(contract.upload_artifact_paths))
    assert tuple(check.label for check in contract.checks) == (
        "workflow_name_matches",
        "workflow_dispatch_enabled",
        "syntax_check_present",
        "guard_run_present",
        "upload_step_present",
        "upload_artifact_name_matches",
        "upload_paths_cover_required",
    )
    assert all(check.status == "pass" for check in contract.checks)


def test_check_by_label() -> None:
    contract = load_frontier_guard_workflow_contract()
    check = contract.check_by_label("upload_paths_cover_required")
    assert check.status == "pass"
