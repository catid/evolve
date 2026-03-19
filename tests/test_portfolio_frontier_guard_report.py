from psmn_rl.analysis.portfolio_frontier_guard_report import evaluate_guard_stack


def test_evaluate_guard_stack_pass() -> None:
    result = evaluate_guard_stack(
        consistency_overall="pass",
        docs_audit_overall="pass",
        workflow_contract_overall="pass",
        doctor_overall="pass",
        seed_pack_doctor_overall="pass",
        active_state_doctor_overall="pass",
    )
    assert result["overall"] == "pass"


def test_evaluate_guard_stack_fail() -> None:
    result = evaluate_guard_stack(
        consistency_overall="pass",
        docs_audit_overall="fail",
        workflow_contract_overall="pass",
        doctor_overall="pass",
        seed_pack_doctor_overall="pass",
        active_state_doctor_overall="pass",
    )
    assert result["overall"] == "fail"
