from psmn_rl.analysis.portfolio_operational_state_loader import load_portfolio_operational_state


def test_load_portfolio_operational_state_snapshot() -> None:
    state = load_portfolio_operational_state()
    assert state.status_type == "portfolio_operational_state"
    assert state.active_candidate == "round6"
    assert state.active_candidate_pack == "outputs/reports/portfolio_candidate_pack.json"
    assert state.archived_frozen_pack == "outputs/reports/frozen_benchmark_pack.json"
    assert state.gate_verdict == "PASS: thaw consideration allowed"
    assert state.gate_mode == "pack"
    assert state.active_pack_role == "narrowed_active_round6"
    assert state.default_restart_prior == "round7"
    assert state.replay_validated_alternate == "round10"
    assert state.evaluation_task == "DoorKey"
    assert state.evaluation_path_key == "external_policy_diagnostics"
    assert state.evaluation_episodes == 64
    assert state.frontier_guard_overall == "pass"
    assert state.active_state_doctor_overall == "pass"
    assert state.ready_for_bounded_restart is True
    assert state.decision_reference == "outputs/reports/next_mega_portfolio_decision_memo.md"
    assert state.gate_report_reference == "outputs/reports/portfolio_gate_report.md"
