from psmn_rl.analysis.portfolio_seed_pack_loader import load_portfolio_seed_pack


def test_load_portfolio_seed_pack_snapshot() -> None:
    pack = load_portfolio_seed_pack()
    assert pack.campaign_name == "lss_portfolio_campaign"
    assert pack.pack_type == "portfolio_seed_pack"
    assert pack.benchmark.active_candidate_name == "round6"
    assert pack.benchmark.active_candidate_pack == "outputs/reports/portfolio_candidate_pack.json"
    assert pack.benchmark.archived_frozen_pack == "outputs/reports/frozen_benchmark_pack.json"
    assert pack.frontier.restart_default == ("round7",)
    assert pack.frontier.reserve_priors == ("round10", "round5")
    assert pack.frontier.retired_priors == ("door3_post5", "post_unlock_x5")
    assert pack.frontier.support_status == "measured_support_regression"
    assert pack.frontier.surviving_candidates == ()
    assert pack.generated_from.restart_policy == "outputs/reports/portfolio_restart_policy.json"
    assert pack.generated_from.screening_spec == "outputs/reports/portfolio_screening_spec.json"
    assert pack.screening_rules.advance_rule == "advance_for_broader_dev"
    assert pack.screening_rules.prune_rules == (
        "prune_support_regression",
        "prune_guardrail_regression",
    )
    assert pack.screening_rules.seed_roles["sentinel"].required_behavior == "track_only"
    assert pack.screening_rules.seed_roles["ranking_support"].required_min_success == 1.0


def test_seed_pack_row_by_candidate() -> None:
    pack = load_portfolio_seed_pack()
    row = pack.row_by_candidate("post_unlock_x5")
    assert row.tier == "retired"
    assert row.policy_bucket == "retire_local_only_fix"
    assert row.screen_rule == "prune_guardrail_regression"
    assert row.support_233 is None
    assert row.guardrail_277 == 0.890625
