from psmn_rl.analysis.lss_portfolio_discriminator_report import _seed_scores


def test_seed_scores_groups_scores_by_seed_and_label() -> None:
    rows = [
        {"candidate": "round10", "lane": "prospective_c", "seed": 193, "label": "kl_lss_sare", "final_greedy_success": 0.0},
        {"candidate": "round10", "lane": "prospective_c", "seed": 193, "label": "kl_lss_token_dense", "final_greedy_success": 0.0},
        {"candidate": "round10", "lane": "prospective_c", "seed": 193, "label": "kl_lss_single_expert", "final_greedy_success": 0.0},
        {"candidate": "round10", "lane": "prospective_f", "seed": 233, "label": "kl_lss_sare", "final_greedy_success": 1.0},
    ]
    grouped = _seed_scores(rows, "round10")
    assert grouped[("prospective_c", 193)] == {
        "kl_lss_sare": 0.0,
        "kl_lss_token_dense": 0.0,
        "kl_lss_single_expert": 0.0,
    }
    assert grouped[("prospective_f", 233)] == {"kl_lss_sare": 1.0}
