from psmn_rl.analysis.lss_portfolio_seed_pack_validator import _validation_bucket


def test_validation_bucket_restart_default() -> None:
    assert _validation_bucket("restart_default", "advance_for_broader_dev", 0.0) == "validated_restart_default"


def test_validation_bucket_reserve() -> None:
    assert _validation_bucket("reserve", "advance_for_broader_dev", 0.0) == "validated_reserve"


def test_validation_bucket_retired() -> None:
    assert _validation_bucket("retired", "prune_support_regression", -0.6) == "validated_retired"
