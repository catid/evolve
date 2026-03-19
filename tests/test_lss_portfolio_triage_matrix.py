from psmn_rl.analysis.lss_portfolio_triage_matrix import _prior_bucket


def test_prior_bucket_conservative_clean() -> None:
    assert _prior_bucket(1.0, 1.0, 1.0, 0.984375, structural=False) == "conservative_clean_prior"


def test_prior_bucket_structural_clean() -> None:
    assert _prior_bucket(1.0, 1.0, 1.0, 0.984375, structural=True) == "structural_clean_prior"


def test_prior_bucket_support_unmeasured_structural() -> None:
    assert _prior_bucket(None, 1.0, 1.0, 0.984375, structural=True) == "support_unmeasured_structural"


def test_prior_bucket_partial_guardrail_loss() -> None:
    assert _prior_bucket(1.0, 1.0, 0.890625, 0.984375, structural=True) == "partial_guardrail_loss"


def test_prior_bucket_not_better_than_incumbent() -> None:
    assert _prior_bucket(1.0, 0.984375, 1.0, 0.984375, structural=False) == "not_better_than_incumbent"
