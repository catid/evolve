from psmn_rl.analysis.portfolio_frontier_guard_workflow_contract import (
    GUARD_RUN_COMMAND,
    GUARD_SCRIPT_PATH,
    REQUIRED_UPLOAD_ARTIFACTS,
    SYNTAX_CHECK_COMMAND,
    UPLOAD_ARTIFACT_NAME,
    WORKFLOW_NAME,
    build_guard_workflow_contract,
)


def test_build_guard_workflow_contract_snapshot() -> None:
    contract = build_guard_workflow_contract()
    assert contract["status_type"] == "portfolio_frontier_guard_workflow_contract"
    assert contract["overall"] == "pass"
    assert contract["workflow_name"] == WORKFLOW_NAME
    assert contract["workflow_dispatch_enabled"] is True
    assert contract["guard_script"] == GUARD_SCRIPT_PATH
    assert contract["syntax_check_command"] == SYNTAX_CHECK_COMMAND
    assert contract["guard_run_command"] == GUARD_RUN_COMMAND
    assert contract["upload_artifact_name"] == UPLOAD_ARTIFACT_NAME
    assert set(REQUIRED_UPLOAD_ARTIFACTS).issubset(set(contract["upload_artifact_paths"]))
    assert [check["label"] for check in contract["checks"]] == [
        "workflow_name_matches",
        "workflow_dispatch_enabled",
        "syntax_check_present",
        "guard_run_present",
        "upload_step_present",
        "upload_artifact_name_matches",
        "upload_paths_cover_required",
    ]
    assert all(check["status"] == "pass" for check in contract["checks"])
