from psmn_rl.analysis.lss_portfolio_weakness_shortlist import _aggregate


def test_aggregate_builds_conservative_round_prior() -> None:
    rows = [
        {"candidate": "round7", "source": "stress_extended", "seed_269": 1.0, "seed_277": 1.0, "lane_mean": 1.0},
        {"candidate": "round7", "source": "migration", "seed_269": 1.0, "seed_277": 1.0, "lane_mean": 1.0},
    ]
    aggregated = _aggregate(rows)
    assert aggregated[0]["bucket"] == "conservative_round_prior"
    assert aggregated[0]["source_count"] == 2


def test_aggregate_builds_local_only_fix() -> None:
    rows = [
        {"candidate": "door2_post4", "source": "migration", "seed_269": 1.0, "seed_277": 0.609375, "lane_mean": 0.8697916667},
    ]
    aggregated = _aggregate(rows)
    assert aggregated[0]["bucket"] == "local_only_fix"
