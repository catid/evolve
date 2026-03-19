from psmn_rl.analysis.lss_portfolio_frontier_manifest import _manifest_bucket


def test_manifest_bucket_default_restart() -> None:
    assert (
        _manifest_bucket("round7", "advance_for_broader_dev", None, ["round7"])
        == "default_restart_prior"
    )


def test_manifest_bucket_replay_validated_alternate() -> None:
    assert (
        _manifest_bucket("round10", "advance_for_broader_dev", "advance_for_broader_dev", ["round7"])
        == "replay_validated_alternate"
    )


def test_manifest_bucket_hold_only() -> None:
    assert (
        _manifest_bucket("round5", "hold_seed_clean_but_below_incumbent", None, ["round7"])
        == "hold_only_prior"
    )
