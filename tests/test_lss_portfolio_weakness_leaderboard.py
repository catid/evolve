from psmn_rl.analysis.lss_portfolio_weakness_leaderboard import _candidate_entry


def test_candidate_entry_extracts_h_lane_triplet() -> None:
    rows = [
        {"candidate": "round7", "label": "kl_lss_sare", "lane": "prospective_h", "seed": 269, "final_greedy_success": 1.0},
        {"candidate": "round7", "label": "kl_lss_sare", "lane": "prospective_h", "seed": 271, "final_greedy_success": 1.0},
        {"candidate": "round7", "label": "kl_lss_sare", "lane": "prospective_h", "seed": 277, "final_greedy_success": 1.0},
    ]
    entry = _candidate_entry("stress_extended", rows, "round7")
    assert entry is not None
    assert entry["seed_269"] == 1.0
    assert entry["seed_271"] == 1.0
    assert entry["seed_277"] == 1.0


def test_candidate_entry_returns_none_without_seed_269() -> None:
    rows = [
        {"candidate": "round7", "label": "kl_lss_sare", "lane": "prospective_h", "seed": 271, "final_greedy_success": 1.0},
    ]
    assert _candidate_entry("stress_extended", rows, "round7") is None
