from psmn_rl.analysis.lss_portfolio_restart_policy import _policy_bucket


def test_policy_bucket_restart_default() -> None:
    assert (
        _policy_bucket(
            "round7",
            "conservative_clean_prior",
            "recommended_conservative_default",
            "measured_support_regression",
        )
        == "restart_default"
    )


def test_policy_bucket_retire_structural_regression() -> None:
    assert (
        _policy_bucket(
            "door3_post5",
            "support_unmeasured_structural",
            None,
            "measured_support_regression",
        )
        == "retire_structural_regression"
    )


def test_policy_bucket_retire_local_only_fix() -> None:
    assert (
        _policy_bucket(
            "post_unlock_x5",
            "partial_guardrail_loss",
            None,
            "measured_support_regression",
        )
        == "retire_local_only_fix"
    )
