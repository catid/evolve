from psmn_rl.analysis.portfolio_candidate_pack_loader import load_portfolio_candidate_pack


def test_load_portfolio_candidate_pack_snapshot() -> None:
    pack = load_portfolio_candidate_pack()
    assert pack.candidate_name == "round6"
    assert pack.pack_type == "candidate_result_pack"
    assert pack.task == "DoorKey"
    assert pack.evaluation.episodes == 64
    assert pack.evaluation.path_key == "external_policy_diagnostics"
    assert pack.evaluation.task == "DoorKey"
    assert pack.active_benchmark_state.active_pack_role == "confirmed_active_round6"
    assert pack.active_benchmark_state.winner == "round6"
    assert pack.active_benchmark_state.challenger_viable_pre_gate is False
    assert (
        pack.active_benchmark_state.current_active_pack.path
        == "outputs/reports/successor_mega_league_candidate_pack.json"
    )
    assert (
        pack.active_benchmark_state.archived_legacy_frozen_pack.path
        == "outputs/reports/frozen_benchmark_pack.json"
    )
    assert pack.portfolio_campaign.winner == "round6"
    assert pack.portfolio_campaign.active_canonical_pack == "outputs/reports/portfolio_candidate_pack.json"
    assert pack.portfolio_campaign.archived_legacy_pack == "outputs/reports/frozen_benchmark_pack.json"
    assert pack.portfolio_campaign.gate_reference_pack == "outputs/reports/frozen_benchmark_pack.json"
    assert pack.requested_claims == ("bounded_teacher_guided_doorkey_sare",)
    assert pack.controls_present == (
        "recovered_token_dense",
        "kl_lss_token_dense",
        "kl_lss_single_expert",
        "baseline_sare",
        "kl_lss_sare",
    )


def test_candidate_pack_artifact_by_role() -> None:
    pack = load_portfolio_candidate_pack()
    artifact = pack.artifact_by_role("candidate_metrics_json")
    assert artifact.path == "outputs/reports/hard_family_saturation_candidate_metrics.json"
    assert artifact.size_bytes == 3321
